# test_rag.py — run this first: python test_rag.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Step 1: Testing imports...")
try:
    from src.rag.knowledge_base import get_knowledge_base
    print("  ✅ knowledge_base imported")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

print("Step 2: Loading embedding model (~80MB download)...")
try:
    kb = get_knowledge_base(auto_seed=False)
    print(f"  ✅ KnowledgeBase created at: {kb.persist_dir}")
except Exception as e:
    print(f"  ❌ KnowledgeBase failed: {e}")
    sys.exit(1)

print("Step 3: Seeding WHO/IDSP documents...")
try:
    kb.seed_default_knowledge(force=True)
    print(f"  ✅ Seeded! Total chunks: {kb.count()}")
    print(f"  📚 Sources: {kb.list_sources()}")
except Exception as e:
    print(f"  ❌ Seeding failed: {e}")
    sys.exit(1)

print("Step 4: Testing retrieval...")
try:
    results = kb.retrieve("dengue outbreak intervention", top_k=2)
    for r in results:
        print(f"  ✅ Found: [{r['score']:.0%}] {r['source']}")
except Exception as e:
    print(f"  ❌ Retrieval failed: {e}")
    sys.exit(1)

print("\n✅ RAG is working! Now run: streamlit run dashboard/chat.py")