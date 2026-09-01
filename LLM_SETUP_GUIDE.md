# EpiPulse AI - Ollama LLM Integration Guide

## 🚀 Quick Start

This guide explains how to set up and use the local LLM (Ollama) integration with EpiPulse AI.

### Prerequisites
- Python 3.9+
- 8GB+ RAM (4GB minimum for Mistral model)
- 10GB+ free disk space

---

## 📦 Installation

### Local Ollama Setup

#### Step 1: Install Ollama
Download and install from: https://ollama.ai

#### Step 2: Pull a Model
```bash
# Download Mistral (7B, ~4.1GB) - Fast and accurate
ollama pull mistral

# Or download Llama 2 (7B, ~3.8GB) - General purpose
ollama pull llama2

# Or download Neural Chat (7B, ~4.9GB) - Optimized for chat
ollama pull neural-chat
```

#### Step 3: Start Ollama Server
```bash
ollama serve
```
By default, it runs on `http://localhost:11434`

#### Step 4: Verify Connection
```bash
curl http://localhost:11434/api/tags
```

#### Step 5: Update Configuration (Optional)
Edit `configs/config.yaml`:
```yaml
llm:
  enabled: true
  ollama_url: http://localhost:11434
  model: mistral  # or your chosen model
  timeout: 60
  temperature: 0.7
```

---

## 🎯 Usage

### 1. Chat Interface (Streamlit)

Start the LLM-powered chat interface:
```bash
streamlit run dashboard/chat.py
```

**Features:**
- 💬 Interactive chat with the AI epidemiologist
- 📍 Region-specific context
- 📊 Quick analysis tools (summarize, list high-risk regions, comparisons)
- 🔍 Ask natural language questions about outbreaks

### 2. FastAPI LLM Endpoints

#### Explain Alert
```bash
curl "http://localhost:8000/llm/explain-alert?region=Delhi&cases=150&risk_score=75&z_score=2.1"
```

Response:
```json
{
  "alert": "High outbreak risk",
  "region": "Delhi",
  "explanation": "Delhi is experiencing a significant outbreak with 150 cases and a risk score of 75/100. The z-score of 2.1 indicates this is substantially above normal baseline..."
}
```

#### Ask Question
```bash
curl "http://localhost:8000/llm/question?question=Which+regions+are+at+highest+risk&region=Delhi"
```

Response:
```json
{
  "question": "Which regions are at highest risk",
  "answer": "Based on current data analysis, regions with the highest outbreak risk are..."
}
```

#### Generate Summary
```bash
curl "http://localhost:8000/llm/summarize?region=Mumbai"
```

Response:
```json
{
  "summary": "Current Situation: Mumbai has recorded 200 cases with a moderate to high risk level..."
}
```

### 3. Python Integration

#### In Your Code
```python
from src.llm import get_llm_client
from src.alerting.alert_generator import generate_alert_with_explanation

# Initialize LLM
llm = get_llm_client()

# Generate alert with explanation
alert = generate_alert_with_explanation(
    cases=150,
    risk_score=75,
    z_score=2.1,
    region="Delhi",
    use_llm=True
)

print(f"Alert: {alert['alert']}")
print(f"Explanation: {alert['explanation']}")

# Ask questions
answer = llm.answer_question(
    "What's the trend in Delhi?",
    context={"region": "Delhi"}
)
print(answer)

# Generate reports
report = llm.generate_report({
    "total_cases": 1000,
    "regions_affected": ["Delhi", "Mumbai"],
    "avg_risk_score": 65
})
print(report)
```

---

## 📊 Model Selection

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **Mistral** | 7B (4GB) | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Excellent | **Recommended** - Good balance |
| **Llama 2** | 7B (4GB) | ⚡⚡⚡ Fast | ⭐⭐⭐⭐ Good | General purpose, open source |
| **Neural Chat** | 7B (5GB) | ⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Chat-optimized, longer responses |
| **Orca Mini** | 3B (2GB) | ⚡⚡⚡⚡ Very Fast | ⭐⭐⭐ Good | Resource-constrained environments |
| **Dolphin Mixtral** | 46B (28GB) | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | Best quality (requires GPU/high RAM) |

**Switch models:**
```bash
ollama pull neural-chat  # Download new model
# Update config.yaml: model: neural-chat
# Restart application
```

---

## ✅ Verification

### Check Ollama Connection
```python
from src.llm import get_llm_client

try:
    llm = get_llm_client()
    print("✅ Connected successfully")
    print(f"Model: {llm.model}")
    print(f"URL: {llm.base_url}")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Test LLM Generation
```bash
curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"mistral","prompt":"What is disease surveillance?","stream":false}'
```

### Monitor Ollama Logs
```bash
tail -f ~/.ollama/logs/server.log
```

---

## 🔧 Configuration

### config.yaml Settings

```yaml
llm:
  enabled: true                    # Enable/disable LLM features
  ollama_url: http://localhost:11434  # Ollama server URL
  model: mistral                   # Model name
  timeout: 60                      # Request timeout (seconds)
  temperature: 0.7                 # Sampling temperature (0-1)
```

**Temperature:**
- 0.0-0.3: Deterministic, precise answers
- 0.5-0.7: Balanced creativity and accuracy (recommended)
- 0.8-1.0: Creative, varied responses

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Ollama at http://localhost:11434"

**Solution:**
1. Check if Ollama is running: `ollama serve`
2. Verify URL in config.yaml
3. Check firewall settings
4. Try: `curl http://localhost:11434/api/tags`

### Issue: "Model not found"

**Solution:**
```bash
ollama pull mistral  # or your chosen model
ollama list          # verify installation
```

### Issue: Slow inference / Out of memory

**Solution:**
1. Use smaller model: `ollama pull orca-mini`
2. Reduce timeout in config.yaml
3. Restart Ollama to free memory: `pkill ollama && ollama serve`

---

## 📈 Performance Tips

1. **Use CPU-optimized small models for quick responses:**
   - Orca Mini (3B) - Fastest
   - Mistral (7B) - Best balance (recommended)

2. **Caching for repeated queries:**
   - Ollama caches loaded models in memory
   - First request is slower, subsequent ones are faster

3. **Batch API calls:**
   - Use `/llm/summarize` for bulk analysis
   - Reduces overhead vs individual `/llm/question` calls

---

## 🚀 Next Steps

1. ✅ Run chat interface: `streamlit run dashboard/chat.py`
2. ✅ Test API endpoints: See **Usage** section above
3. ✅ Integrate into Airflow pipeline
4. ✅ Add LLM insights to alerts
5. ✅ Fine-tune model with domain data (advanced)

---

## 📚 Resources

- **Ollama Documentation:** https://github.com/ollama/ollama
- **Model Library:** https://ollama.ai/library
- **LLM Performance Guide:** https://ollama.ai/library
- **EpiPulse AI Docs:** See README.md

---

## 🎓 Learning Resources

- [Understanding LLMs](https://en.wikipedia.org/wiki/Large_language_model)
- [Ollama Getting Started](https://github.com/ollama/ollama#quickstart)
- [Prompt Engineering Tips](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

---

**Happy Epidemiological Insights! 🩺🔬**
