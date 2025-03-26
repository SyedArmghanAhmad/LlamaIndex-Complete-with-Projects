# 🧪 Clinical Trial Intelligence Explorer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://clinical-trial-explorer.streamlit.app)
[![Groq](https://img.shields.io/badge/Groq-00FF00?style=for-the-badge&logo=groq&logoColor=black)](https://groq.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-FFD700?style=for-the-badge)](https://www.llamaindex.ai/)

**A cognitive search interface for exploring clinical trials using natural language queries**, powered by Groq's ultra-fast LLMs and enhanced with medical reasoning capabilities.

## ✨ Features

- **🧠 Natural Language Querying**  
  "Find Phase 2 Alzheimer's trials recruiting patients over 65"
  
- **⚡ Real-time AI Insights**  
  Get clinical significance analysis and trial comparisons

- **🔍 Context-Aware Filtering**  
  Automatic detection of phase, status, and medical conditions

- **📊 Structured & Unstructured Data Fusion**  
  Combines vector search with metadata filtering

- **👩⚕️ Clinician-Friendly Interface**  
  Trial cards with eligibility highlights and safety alerts

- **📈 Temporal Analysis**  
  Automatic detection of recent trials and study timelines

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Groq API key (free tier available)
- Streamlit account

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Configuration

1. Get your Groq API key from [console.groq.com](https://console.groq.com/)
2. Add to `.env`:

   ```env
   GROQ_API_KEY=your_key_here
   ```

### Usage

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

| Technology          | Role                                |
|---------------------|-------------------------------------|
| ![Groq](https://img.shields.io/badge/Groq-LLM_Engine-00FF00?logo=groq&logoColor=black) | Ultra-fast inference engine |
| ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG_Framework-FFD700) | Hybrid search & retrieval |
| ![Streamlit](https://img.shields.io/badge/Streamlit-UI_Framework-FF4B4B) | Interactive web interface |
| ![HuggingFace](https://img.shields.io/badge/BAAI/bge--small--en--v1.5-Embeddings-yellow) | Clinical text embeddings |

## 📚 Data Model

The synthetic dataset includes:

- 10 clinical trials across 3 therapeutic areas
- Metadata-enriched documents with:
  - Trial phases (1-4)
  - Recruitment statuses
  - Eligibility criteria
  - Intervention types
  - Temporal attributes

## 🤖 AI Capabilities

### Intelligent Query Handling

```python
def query_trials(question):
    """Example processing pipeline"""
    return llm(f"""
    Analyze query: {question}
    1. Detect medical condition
    2. Identify phase preference
    3. Determine recruitment status
    4. Extract eligibility parameters
    5. Rank by relevance
    """)
```

### Safety Features

- Phase-appropriate warnings
- Trial termination alerts
- Eligibility conflict detection
- Recency validation

## 🌟 Example Queries

```markdown
- "Show recent Phase 3 Alzheimer's drug trials"
- "What Parkinson's studies are currently recruiting?"
- "Find vaccine trials for neurodegenerative diseases"
- "List cognitive therapy interventions in Phase 2"
```

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙌 Contribution

Contributions welcome! Please see:

1. Fork the repository
2. Create your feature branch
3. Submit a PR with tests

```bash
# Run tests
pytest test/
```

## ⚠️ Disclaimer

> **This is a demonstration system**  
> Uses synthetic data - Not for clinical decision making  
> Always verify trial details at [ClinicalTrials.gov](https://clinicaltrials.gov)

---
