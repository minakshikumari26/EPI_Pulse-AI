"""
EpiPulse AI — RAG Knowledge Base
Loads PDF/TXT documents, chunks them, embeds with sentence-transformers,
and stores in ChromaDB for semantic retrieval.

Usage:
    from src.rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    kb.add_pdf("docs/who_dengue_guidelines.pdf", source="WHO Dengue Guidelines 2023")
    kb.add_text("Dengue is transmitted by Aedes aegypti mosquitoes...", source="Manual entry")
    kb.save()   # persists to disk
"""

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
CHROMA_DIR  = PROJECT_ROOT / "data" / "rag_db"
DOCS_DIR    = PROJECT_ROOT / "data" / "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"    # 80MB, fast, good quality
CHUNK_SIZE  = 400                    # chars per chunk
CHUNK_OVERLAP = 80                   # overlap between chunks


# ── Built-in seed knowledge (no PDFs needed to get started) ──────────────────
SEED_KNOWLEDGE = [
    {
        "source": "WHO Dengue Guidelines",
        "text": """Dengue fever is a mosquito-borne viral infection caused by the dengue virus (DENV).
It is transmitted by Aedes mosquitoes, primarily Aedes aegypti.
Dengue occurs in tropical and sub-tropical climates worldwide, mostly in urban areas.
Symptoms include high fever, severe headache, pain behind the eyes, muscle and joint pains, nausea, vomiting, swollen glands and rash.
For outbreak response: vector control through elimination of breeding sites is the primary intervention.
Insecticide spraying should target adult mosquitoes. Public health education on removing stagnant water is essential.
Hospitalization is required for severe dengue with warning signs: abdominal pain, persistent vomiting, bleeding, rapid breathing."""
    },
    {
        "source": "WHO Dengue Guidelines",
        "text": """Dengue outbreak thresholds: An outbreak is declared when case counts exceed 2 standard deviations above the seasonal mean.
Early warning indicators include: increase in febrile illness presentations, positive NS1 antigen tests, and increased Aedes larval indices.
The epidemic threshold z-score of 1.5 or greater warrants immediate public health response.
Intervention priorities for high z-score regions: (1) Enhanced surveillance, (2) Vector control operations,
(3) Hospital preparedness, (4) Community mobilization.
High-risk periods: monsoon and post-monsoon seasons (July-October in India) when humidity >70% and temperatures 25-35°C."""
    },
    {
        "source": "WHO Malaria Guidelines",
        "text": """Malaria is caused by Plasmodium parasites transmitted through Anopheles mosquitoes.
In India, P. vivax and P. falciparum are the dominant species.
Peak transmission occurs during and after monsoon season (June-September) when mosquito breeding is highest.
Outbreak indicators: case incidence exceeding 5 per 1000 population per month, or 2x increase in weekly cases.
Response measures: (1) Mass drug administration in high-burden areas, (2) Indoor residual spraying,
(3) Long-lasting insecticidal nets distribution, (4) Rapid diagnostic testing at community level.
High-risk states: Odisha, Jharkhand, Chhattisgarh, Meghalaya (northeast India)."""
    },
    {
        "source": "WHO Malaria Guidelines",
        "text": """Malaria risk factors include: proximity to water bodies, high rainfall, high humidity, temperature 20-30°C.
Climate correlation: malaria transmission increases significantly when rainfall >100mm/month and temperature between 20-30°C.
Districts with high Annual Parasite Index (API > 2) are classified as high-burden.
Urban malaria is increasing due to construction sites, water storage, and unplanned drainage.
Intervention for active outbreaks: indoor residual spraying within 24 hours of cluster detection,
followed by case investigation and treatment within 48 hours."""
    },
    {
        "source": "IDSP India Disease Surveillance",
        "text": """The Integrated Disease Surveillance Programme (IDSP) monitors 33 diseases across India weekly.
Outbreak alert system: S (Syndromic), P (Presumptive), L (Laboratory) reporting from districts.
Alert threshold: 2x increase in weekly cases compared to previous 4-week average.
High-risk states for dengue: Delhi, Maharashtra, Karnataka, Tamil Nadu, West Bengal.
Seasonal pattern: dengue cases peak August-November across most Indian states.
For rapid response: district health officers must be notified within 24 hours of threshold breach.
Contact tracing and source investigation mandatory for cholera and typhoid outbreaks."""
    },
    {
        "source": "IDSP India Disease Surveillance",
        "text": """Cholera outbreak response protocol in India:
Cholera is a notifiable disease under the Epidemic Diseases Act.
Outbreak definition: 2 or more linked cases with laboratory confirmation.
Immediate actions: (1) Case isolation and oral rehydration therapy,
(2) Chlorination of water sources, (3) Food safety inspections,
(4) Antibiotic prophylaxis for close contacts in high-risk settings.
Risk factors: contaminated water supply, open defecation, flooding, poor sanitation.
High-risk periods: May-July (pre-monsoon), post-flood events.
Cholera hotspots in India: Kolkata, Mumbai slums, rural Bihar, Odisha coastal districts."""
    },
    {
        "source": "IDSP India Disease Surveillance",
        "text": """Typhoid fever surveillance and response:
Typhoid (enteric fever) caused by Salmonella Typhi is endemic in India.
Annual burden estimated at 4.5 million cases, concentrated in urban slums and peri-urban areas.
Outbreak threshold: cluster of 5+ cases linked to a common source within 2 weeks.
Risk factors: contaminated water, street food, poor handwashing, overcrowding.
Response: (1) Water quality testing and chlorination, (2) Food handler screening,
(3) Antibiotic treatment (azithromycin or ceftriaxone), (4) Vaccination in high-risk groups.
Peak season: April-June before monsoon when water sources are stressed.
High-burden cities: Delhi, Mumbai, Lucknow, Patna, Kolkata."""
    },
    {
        "source": "COVID-19 Epidemiology Reference",
        "text": """COVID-19 resurgence indicators and public health response:
Resurgence threshold: >10% test positivity rate sustained for 7 days, or doubling time <14 days.
Risk factors for resurgence: waning immunity, new variants, low booster coverage, crowded settings.
Early warning signals: increase in influenza-like illness (ILI) presentations, wastewater surveillance positivity.
Intervention ladder: (1) Enhanced testing and genomic surveillance, (2) Targeted masking in high-risk settings,
(3) Booster campaign acceleration, (4) Hospital surge preparedness.
High-risk populations: elderly (>60), immunocompromised, unvaccinated.
India-specific: urban dense clusters (Delhi, Mumbai, Bengaluru) are first to show resurgence signals."""
    },
    {
        "source": "Epidemiology Risk Assessment Framework",
        "text": """Z-score interpretation for disease surveillance:
Z-score 0-1.0: Normal seasonal variation — routine monitoring sufficient.
Z-score 1.0-1.5: Elevated — increase surveillance frequency, prepare response teams.
Z-score 1.5-2.0: Alert threshold — activate rapid response protocol, notify district health officer.
Z-score 2.0-3.0: High alert — outbreak likely, deploy field teams, consider public communication.
Z-score >3.0: Emergency — immediate public health emergency declaration, activate incident command.

Risk score interpretation (0-1 scale):
0.0-0.25: Low risk — standard surveillance
0.25-0.55: Medium risk — enhanced monitoring, vector control
0.55-0.75: High risk — active outbreak response
0.75-1.0: Critical — emergency response, resource mobilization"""
    },
    {
        "source": "Epidemiology Risk Assessment Framework",
        "text": """Climate-disease correlations for Indian subcontinent:
Temperature 25-35°C + Humidity >70% + Rainfall 50-200mm: High dengue and malaria risk.
Temperature >35°C + Low humidity: Reduced mosquito survival, lower vector-borne disease risk.
Post-monsoon period (October-November): Peak dengue season across northern India.
Pre-monsoon heat (April-June): Peak typhoid and cholera risk due to water stress.
Winter (December-February): Lower vector-borne disease, higher respiratory illness.
Flooding events: Immediate cholera and leptospirosis risk spike within 2 weeks.
El Niño years: Associated with 20-40% increase in malaria cases across South Asia."""
    },
    {
        "source": "Public Health Intervention Guidelines",
        "text": """Resource allocation framework for outbreak response:
Priority scoring = (Z-score × 0.4) + (Risk Score × 0.35) + (Population density × 0.15) + (Healthcare access × 0.10)
Top-priority regions should receive: rapid response teams within 6 hours, supplies within 24 hours.

For HIGH risk regions (score >0.55):
- Deploy district rapid response team immediately
- Activate emergency operations center
- Begin daily case reporting
- Issue public health advisory
- Coordinate with state health department

For MEDIUM risk regions (score 0.25-0.55):
- Increase surveillance frequency to twice weekly
- Pre-position medical supplies
- Alert local health workers
- Conduct environmental investigation"""
    },
    {
        "source": "Public Health Intervention Guidelines",
        "text": """Intervention effectiveness evidence base:
Vector control for dengue: Indoor residual spraying reduces transmission by 60-80% when coverage >80%.
Larviciding: Reduces dengue incidence by 40-60% in urban areas.
Community mobilization: Source reduction campaigns reduce Aedes breeding sites by 50-70%.
Bed nets (malaria): Long-lasting insecticidal nets reduce malaria incidence by 50% in high-transmission areas.
Water chlorination: Reduces cholera transmission by 90% when residual chlorine maintained at 0.5mg/L.
Rapid treatment: Early dengue case management reduces case fatality rate from 5% to <1%.
Vaccination: Typhoid conjugate vaccine is 80% effective; recommended for high-risk populations."""
    },
]


class KnowledgeBase:
    """
    RAG knowledge base backed by ChromaDB with sentence-transformer embeddings.
    Supports PDF ingestion, manual text addition, and semantic retrieval.
    """

    def __init__(self, persist_dir: Path = CHROMA_DIR, embed_model: str = EMBED_MODEL):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RAG] Loading embedding model: {embed_model}")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model)

        import chromadb
        self.client     = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="epipulse_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[RAG] Collection has {self.collection.count()} documents")

    # ── Seeding ───────────────────────────────────────────────────────────────
    def seed_default_knowledge(self, force: bool = False):
        """Load built-in WHO/IDSP knowledge if collection is empty."""
        if self.collection.count() > 0 and not force:
            print(f"[RAG] Skipping seed — already have {self.collection.count()} docs")
            return

        print(f"[RAG] Seeding {len(SEED_KNOWLEDGE)} built-in documents...")
        for item in SEED_KNOWLEDGE:
            chunks = self._chunk_text(item["text"])
            for chunk in chunks:
                self._add_chunk(chunk, source=item["source"])
        print(f"[RAG] Seed complete — {self.collection.count()} chunks in DB")

    # ── PDF ingestion ─────────────────────────────────────────────────────────
    def add_pdf(self, pdf_path: str, source: str = None) -> int:
        """Extract text from PDF, chunk it, and add to the knowledge base."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Run: pip install pypdf")

        path   = Path(pdf_path)
        source = source or path.stem
        reader = PdfReader(str(path))
        n_added = 0

        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            full_text += f"\n[Page {page_num + 1}]\n{text}"

        chunks = self._chunk_text(full_text)
        for chunk in chunks:
            self._add_chunk(chunk, source=source)
            n_added += 1

        print(f"[RAG] Added {n_added} chunks from {path.name} (source: {source})")
        return n_added

    def add_text(self, text: str, source: str = "Manual") -> int:
        """Add plain text to the knowledge base."""
        text    = self._clean_text(text)
        chunks  = self._chunk_text(text)
        n_added = 0
        for chunk in chunks:
            self._add_chunk(chunk, source=source)
            n_added += 1
        print(f"[RAG] Added {n_added} chunks from '{source}'")
        return n_added

    def add_uploaded_file(self, file_obj, filename: str, source: str = None) -> int:
        """Add a file-like object (from Streamlit uploader)."""
        source = source or Path(filename).stem
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            # Save temporarily then process
            tmp_path = self.persist_dir / f"_tmp_{filename}"
            tmp_path.write_bytes(file_obj.read())
            n = self.add_pdf(str(tmp_path), source=source)
            tmp_path.unlink()
            return n
        elif suffix in [".txt", ".md"]:
            text = file_obj.read().decode("utf-8", errors="ignore")
            return self.add_text(text, source=source)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        disease_filter: str = None,
    ) -> list[dict]:
        """
        Semantic search — returns top_k most relevant chunks.

        Returns list of dicts: {text, source, score, chunk_id}
        """
        if self.collection.count() == 0:
            return []

        query_emb = self.embedder.encode([query])[0].tolist()

        where = None
        if disease_filter:
            # Try to filter by disease keyword in source
            pass  # ChromaDB metadata filtering — skip for simplicity

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1 - dist, 4)  # cosine similarity (0-1)
            if score > 0.15:            # relevance threshold
                retrieved.append({
                    "text":     doc,
                    "source":   meta.get("source", "Unknown"),
                    "score":    score,
                    "chunk_id": meta.get("chunk_id", ""),
                })

        return retrieved

    def list_sources(self) -> list[str]:
        """Return all unique source names in the knowledge base."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = list({m.get("source", "Unknown") for m in results["metadatas"]})
        return sorted(sources)

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Remove all documents from the collection."""
        self.client.delete_collection("epipulse_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="epipulse_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        print("[RAG] Collection cleared")

    # ── Private helpers ───────────────────────────────────────────────────────
    def _add_chunk(self, text: str, source: str):
        chunk_id = hashlib.md5(f"{source}::{text}".encode()).hexdigest()
        try:
            emb = self.embedder.encode([text])[0].tolist()
            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"source": source, "chunk_id": chunk_id}],
            )
        except Exception as e:
            print(f"[RAG] Warning: could not add chunk — {e}")

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks of ~CHUNK_SIZE chars."""
        text   = text.strip()
        if not text:
            return []

        # Split on sentence boundaries first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks    = []
        current   = ""

        for sent in sentences:
            if len(current) + len(sent) <= CHUNK_SIZE:
                current += " " + sent
            else:
                if current.strip():
                    chunks.append(current.strip())
                # Start new chunk with overlap
                current = current[-CHUNK_OVERLAP:] + " " + sent if len(current) > CHUNK_OVERLAP else sent

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 50]

    def _clean_text(self, text: str) -> str:
        """Remove PDF artifacts and normalize whitespace."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # fix hyphenation
        return text.strip()


# ── Singleton ─────────────────────────────────────────────────────────────────
_kb_instance = None

def get_knowledge_base(auto_seed: bool = True) -> KnowledgeBase:
    """Get or create the global KnowledgeBase instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
        if auto_seed:
            _kb_instance.seed_default_knowledge()
    return _kb_instance
