import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import json
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from llama_index.core import (
    StorageContext, VectorStoreIndex, Settings, Document,
    load_index_from_storage, PromptTemplate
)
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
import chromadb
from dotenv import load_dotenv
import random
from faker import Faker
import pandas as pd
import time
import uuid
import streamlit as st
from typing import List
import re
import shutil
from functools import lru_cache
from llama_index.core import QueryBundle

# Initialize Faker for fake data generation
fake = Faker()

# Load environment variables
load_dotenv()

# =============================================
# 1. Persistent Data Storage Paths
# =============================================

DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")
CHROMA_DIR = Path("./chroma_db")
METRICS_FILE = Path("./query_metrics.json")
QUALITY_FILE = Path("./quality_metrics.json")
VERSIONS_FILE = Path("./versions.json")

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# =============================================
# 2. Enhanced Document Loader with PDF Support
# =============================================

class FinancialDataLoader:
    def __init__(self):
        self.parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            include_metadata=True
        )

    def load_pdf(self, file_path: str, additional_metadata: dict = None) -> List[TextNode]:
        """Load and parse PDF financial reports with metadata"""
        reader = PDFReader()
        documents = reader.load_data(file=file_path)
        
        # Add custom metadata if provided
        if additional_metadata:
            for doc in documents:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata.update(additional_metadata)
                
        return self._process_documents(documents, source_type="pdf")

    def _process_documents(self, documents: List[Document], source_type: str) -> List[TextNode]:
        """Properly convert Documents to TextNodes with metadata"""
        nodes = []
        for doc in documents:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            
            doc.metadata.update({
                "source_type": source_type,
                "ingestion_date": datetime.now().isoformat()
            })
            
            new_nodes = self.parser.get_nodes_from_documents([doc])
            for node in new_nodes:
                if not hasattr(node, 'ref_doc_id'):
                    node.ref_doc_id = doc.id_ if hasattr(doc, 'id_') else str(uuid.uuid4())
                nodes.append(node)
        return nodes

# =============================================
# 3. Generate Company Data (Initial Seed Only)
# =============================================

def generate_company():
    """Generate realistic company financial data"""
    return {
        "company": fake.company(),
        "symbol": fake.lexify(text="????").upper(),
        "cik": str(random.randint(10**9, 10**10)),
        "fiscal_year": random.randint(2018, 2023),
        "revenue": round(random.uniform(1e9, 500e9), 2),
        "net_income": round(random.uniform(1e8, 50e9), 2),
        "eps": round(random.uniform(1.0, 20.0), 2),
        "employees": random.randint(1000, 500000),
        "risk_factors": [fake.sentence() for _ in range(20)],
        "mdna": [fake.paragraph() for _ in range(10)]
    }

# Only generate synthetic data if no existing index
if not STORAGE_DIR.exists():
    companies = [generate_company() for _ in range(50)]
else:
    companies = []

# =============================================
# 4. JSON Serialization Handling
# =============================================

class CustomJSONEncoder(json.JSONEncoder):
    """Handles both date and datetime serialization"""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

# =============================================
# 5. Vector Store Setup (Cached)
# =============================================

@st.cache_resource(show_spinner=False)
def initialize_vector_store():
    """Initialize ChromaDB client with caching to prevent reinitialization"""
    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("sec_filings")
    return ChromaVectorStore(chroma_collection=chroma_collection)

vector_store = initialize_vector_store()

# =============================================
# 6. Document Creation
# =============================================

def create_sec_document(company):
    """Convert company data to structured SEC filing document"""
    sections = {
        "Business Overview": fake.paragraphs(nb=5),
        "Risk Factors": company["risk_factors"],
        "Management Discussion": company["mdna"],
        "Financial Statements": [
            f"Revenue: ${company['revenue']/1e9:.2f}B",
            f"Net Income: ${company['net_income']/1e9:.2f}B",
            f"EPS: ${company['eps']}",
            f"Employees: {company['employees']:,}"
        ]
    }

    full_text = "\n\n".join(
        [f"## {section}\n" + "\n".join(content)
         for section, content in sections.items()]
    )

    filed_date = fake.date_between(start_date="-5y", end_date="today")

    return Document(
        text=full_text,
        metadata={
            "company": company["company"],
            "symbol": company["symbol"],
            "filing_type": random.choice(["10-K", "10-Q", "8-K"]),
            "fiscal_year": company["fiscal_year"],
            "filed_date": filed_date.isoformat(),
            "full_text": full_text,
            "source_type": "synthetic"
        }
    )

# =============================================
# 7. Index Creation with Multi-Source Support (Cached)
# =============================================

@st.cache_resource(show_spinner=False)
def create_or_load_index():
    """Create or load index with multi-source data (cached to prevent re-embedding)"""
    print("Initializing or loading index...")
    loader = FinancialDataLoader()

    # Generate synthetic SEC data only if no existing index
    sec_documents = []
    if not STORAGE_DIR.exists():
        sec_documents = [create_sec_document(c) for c in companies]

    # Load any existing external files
    external_nodes = []
    if DATA_DIR.exists():
        for file in DATA_DIR.glob("*.pdf"):
            external_nodes.extend(loader.load_pdf(str(file)))

    # Combine all nodes
    all_documents = sec_documents + external_nodes

    # Initialize settings (only once)
    Settings.llm = Groq(model="llama-3.3-70b-versatile")
    Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200

    # Create or load index
    if not STORAGE_DIR.exists():
        print("Creating new index with multi-source data...")
        try:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            # Initialize fresh storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex(
                all_documents,
                storage_context=storage_context,
                show_progress=True,
                store_nodes_override=True
            )
            index.storage_context.persist(persist_dir=STORAGE_DIR)
            print("Index created successfully")
            return index
        except Exception as e:
            print(f"Error creating index: {e}")
            if STORAGE_DIR.exists():
                shutil.rmtree(STORAGE_DIR, ignore_errors=True)
            raise
    else:
        print("Loading existing index...")
        try:
            # Verify all required files exist
            required_files = ['docstore.json', 'vector_store.json', 'index_store.json']
            missing_files = [f for f in required_files if not (STORAGE_DIR / f).exists()]
            
            if missing_files:
                print(f"Missing index files: {missing_files}, recreating index...")
                shutil.rmtree(STORAGE_DIR, ignore_errors=True)
                STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                # Initialize fresh storage context for recreation
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                index = VectorStoreIndex(
                    all_documents,
                    storage_context=storage_context,
                    show_progress=True,
                    store_nodes_override=True
                )
                index.storage_context.persist(persist_dir=STORAGE_DIR)
                return index
            
            # Load existing storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=STORAGE_DIR,
                vector_store=vector_store
            )
            index = load_index_from_storage(storage_context)
            
            # Add any new documents
            if external_nodes:
                print("Adding new documents to existing index...")
                for node in external_nodes:
                    index.insert_nodes([node])
                index.storage_context.persist()
            
            print("Index loaded successfully")
            return index
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Attempting to recreate index...")
            shutil.rmtree(STORAGE_DIR, ignore_errors=True)
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            # Initialize fresh storage context for recreation
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex(
                all_documents,
                storage_context=storage_context,
                show_progress=True,
                store_nodes_override=True
            )
            index.storage_context.persist(persist_dir=STORAGE_DIR)
            return index

# Create/Load index (cached)
index = create_or_load_index()

# =============================================
# 8. Retrieval and Search Setup (Cached)
# =============================================

@st.cache_resource(show_spinner=False)
def initialize_retrievers(_index):
    """Initialize retrievers with caching to prevent reinitialization"""
    print("Initializing retrievers...")
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.schema import TextNode
    import Stemmer
    
    try:
        # First try to get nodes from the existing docstore
        if hasattr(_index, 'docstore') and hasattr(_index.docstore, 'docs'):
            text_nodes = []
            for node_id, node in _index.docstore.docs.items():
                clean_text = node.text if hasattr(node, "text") else node.content if hasattr(node, "content") else ""
                if clean_text:
                    text_nodes.append(TextNode(
                        text=clean_text,
                        metadata=node.metadata,
                        excluded_embed_metadata_keys=["filed_date", "ingestion_date"],
                        excluded_llm_metadata_keys=["filed_date", "ingestion_date"],
                        id_=node_id
                    ))
            
            # Initialize BM25 retriever with the nodes
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=text_nodes,
                similarity_top_k=10,
                stemmer=Stemmer.Stemmer("english")
            )
        else:
            # Fallback if docstore isn't available
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=_index.docstore,
                similarity_top_k=10,
                stemmer=Stemmer.Stemmer("english")
            )
        
        # Initialize vector retriever
        vector_retriever = _index.as_retriever(similarity_top_k=5)
        
        return bm25_retriever, vector_retriever
    
    except Exception as e:
        print(f"Error initializing retrievers: {e}")
        # Fallback to simple vector retriever if BM25 fails
        return None, _index.as_retriever(similarity_top_k=10)

bm25_retriever, vector_retriever = initialize_retrievers(index)

def hybrid_search(query: str):
    """Production-grade hybrid search with score fusion"""
    query_bundle = QueryBundle(query_str=query)

    # If BM25 retriever failed to initialize, use only vector search
    if bm25_retriever is None:
        return vector_retriever.retrieve(query_bundle)[:7]

    # Retrieve from both systems
    vector_results = vector_retriever.retrieve(query_bundle)
    bm25_results = bm25_retriever.retrieve(query_bundle)

    # RRF fusion
    combined_scores = {}
    for rank, result in enumerate(vector_results):
        combined_scores[result.node.node_id] = combined_scores.get(result.node.node_id, 0) + 1/(rank + 60)

    for rank, result in enumerate(bm25_results):
        combined_scores[result.node.node_id] = combined_scores.get(result.node.node_id, 0) + 1/(rank + 60)

    # Sort and select top results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:7]

    # Preserve order
    final_results = []
    seen_ids = set()
    for result in vector_results + bm25_results:
        if result.node.node_id in dict(sorted_results) and result.node.node_id not in seen_ids:
            final_results.append(result)
            seen_ids.add(result.node.node_id)

    return final_results[:7]

# =============================================
# 9. Persistent Query Monitoring and Versioning
# =============================================

class QueryMonitor:
    def __init__(self):
        self.queries = self._load_metrics()
        
    def _load_metrics(self):
        """Load metrics from file or create new DataFrame"""
        if METRICS_FILE.exists():
            try:
                df = pd.read_json(METRICS_FILE)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except:
                pass
        return pd.DataFrame(columns=["timestamp", "query", "latency", "results"])
    
    def _save_metrics(self):
        """Save metrics to file"""
        self.queries.to_json(METRICS_FILE, orient='records', date_format='iso')

    def log_query(self, query: str, results: list, latency: float):
        new_entry = pd.DataFrame({
            "timestamp": [datetime.now()],
            "query": [query],
            "latency": [latency],
            "results": [len(results)]
        })
        self.queries = pd.concat([self.queries, new_entry], ignore_index=True)
        self._save_metrics()

    def generate_report(self):
        if self.queries.empty:
            return pd.DataFrame()

        report = self.queries.groupby(pd.Grouper(key="timestamp", freq="D")).agg({
            "query": "count",
            "latency": "mean",
            "results": "mean"
        }).rename(columns={
            "query": "query_count",
            "latency": "avg_latency",
            "results": "avg_results"
        })
        return report

class VersionedIndex:
    def __init__(self):
        self.versions = self._load_versions()
        
    def _load_versions(self):
        """Load versions from file or return empty dict"""
        if VERSIONS_FILE.exists():
            try:
                with open(VERSIONS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {}

    def get_active_version(self):
        """Get the currently active version (based on storage directory)"""
        if not STORAGE_DIR.exists():
            return None
        # Find which version matches the current storage
        for version_id, version_data in self.versions.items():
            version_dir = STORAGE_DIR / f"index_{version_id}"
            if version_dir.exists() and version_dir.samefile(STORAGE_DIR):
                return version_id
        return None

    def commit(self, index, description: str):
        version_id = str(uuid.uuid4())
        version_dir = STORAGE_DIR / f"index_{version_id}"
        
        # Create the new version directory
        version_dir.mkdir(exist_ok=True)
        
        # Persist the index to the version directory
        index.storage_context.persist(persist_dir=version_dir)
        
        # Record the version metadata
        cleaned_docs = []
        for doc in index.docstore.docs.values():
            safe_metadata = {
                k: v.isoformat() if isinstance(v, (date, datetime)) else str(v)
                for k, v in doc.metadata.items()
            }
            cleaned_docs.append({"id": doc.id_, "metadata": safe_metadata})

        self.versions[version_id] = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "directory": str(version_dir),
            "stats": {
                "doc_count": len(index.docstore.docs),
                "companies": list(set(
                    d["metadata"]["symbol"] for d in cleaned_docs
                    if "symbol" in d["metadata"]
                ))
            }
        }

        with open(VERSIONS_FILE, "w") as f:
            json.dump(self.versions, f, cls=CustomJSONEncoder, indent=2)

        return version_id

    def delete_version(self, version_id: str):
        """Delete a specific version"""
        if version_id in self.versions:
            # Remove the version directory
            version_dir = Path(self.versions[version_id]["directory"])
            if version_dir.exists():
                import shutil
                shutil.rmtree(version_dir)
            
            # Remove from versions record
            del self.versions[version_id]
            
            # Save updated versions
            with open(VERSIONS_FILE, "w") as f:
                json.dump(self.versions, f, cls=CustomJSONEncoder, indent=2)
            return True
        return False

    def switch_version(self, version_id: str):
        """Switch to a specific version"""
        if version_id in self.versions:
            version_dir = Path(self.versions[version_id]["directory"])
            
            if version_dir.exists():
                # Clear the current storage
                if STORAGE_DIR.exists():
                    import shutil
                    shutil.rmtree(STORAGE_DIR)
                
                # Copy the version to be the active storage
                shutil.copytree(version_dir, STORAGE_DIR)
                return True
        return False

class QualityMetrics:
    def __init__(self):
        self.metrics = self._load_metrics()
        
    def _load_metrics(self):
        """Load quality metrics from file or create new DataFrame"""
        if QUALITY_FILE.exists():
            try:
                df = pd.read_json(QUALITY_FILE)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except:
                pass
        return pd.DataFrame(columns=[
            "timestamp", "query", "quality_score", "complete", 
            "citation_count", "has_risk_score", "recommendations"
        ])
    
    def _save_metrics(self):
        """Save quality metrics to file"""
        self.metrics.to_json(QUALITY_FILE, orient='records', date_format='iso')

    def log_metrics(self, query: str, validation: dict):
        new_entry = pd.DataFrame({
            "timestamp": [datetime.now()],
            "query": [query],
            "quality_score": [validation["quality_score"]],
            "complete": [validation["complete"]],
            "citation_count": [validation["citation_count"]],
            "has_risk_score": [validation["risk_score_present"]],
            "recommendations": [validation["recommendation_count"]]
        })
        self.metrics = pd.concat([self.metrics, new_entry], ignore_index=True)
        self._save_metrics()

# =============================================
# 10. Enhanced Analysis and Query System
# =============================================

FINANCIAL_ANALYSIS_PROMPT = PromptTemplate("""\
**Context**: {context_str}

As a senior financial analyst with 20+ years experience in SEC filings analysis, provide a professional assessment of:

**Query**: {query_str}

Structure your response using these sections:

1. **Executive Summary** (3-5 bullet points)
   - Focus on key governance and succession-related elements
   - Highlight explicit vs implicit succession planning mentions

2. **Leadership Structure Analysis**
   - Current key personnel roles and responsibilities
   - Board composition analysis
   - Any stated succession timelines or age-related policies

3. **Risk Assessment**
   - Concentration risk score (1-5) based on leadership dependence
   - Mitigation strategies mentioned
   - Historical transitions patterns in the company

4. **Documentary Evidence**
   - Specific filing excerpts with dates
   - Cross-reference between different filings
   - Note any discrepancies or omissions

5. **Comparative Analysis** (where applicable)
   - Industry benchmarking
   - Historical precedents within the company
   - Regulatory requirements compliance

6. **Recommendations**
   - Suggested governance improvements
   - Investor considerations
   - Potential timeline implications

**Format Requirements:**
- Use professional analyst tone with appropriate hedging
- Differentiate between stated facts and reasonable inferences
- Highlight any contradictory information
- Cite sources using [Doc#] notation
- Include relevant excerpts with page references when available
- For numerical data, show calculations/derivations
""")

def validate_response(response: str) -> dict:
    """Ensure all required sections are present and score response quality"""
    required_sections = [
        "Executive Summary",
        "Leadership Structure Analysis", 
        "Risk Assessment",
        "Documentary Evidence",
        "Comparative Analysis",
        "Recommendations"
    ]
    
    # Basic validation
    validation = {
        "complete": all(section in response for section in required_sections),
        "missing_sections": [s for s in required_sections if s not in response],
        "citation_count": response.count("[Doc"),
        "risk_score_present": "risk score" in response.lower(),
        "recommendation_count": len([m for m in response.split("\n") if m.strip().startswith("-")])
    }
    
    # Quality scoring (0-1 scale)
    validation["quality_score"] = min(1.0, (
        0.3 * validation["complete"] +
        0.2 * (validation["citation_count"] / 5) +
        0.2 * validation["risk_score_present"] +
        0.2 * (validation["recommendation_count"] / 3) +
        0.1 * ("**" in response)  # Markdown formatting
    ))
    
    return validation

def enhance_response(response: str) -> str:
    """Post-process the response to ensure consistent formatting"""
    # Ensure section headers are properly formatted
    sections = {
        "Executive Summary": "## 🏛️ Executive Summary",
        "Leadership Structure Analysis": "## 👨‍💼 Leadership Structure Analysis",
        "Risk Assessment": "## ⚠️ Risk Assessment",
        "Documentary Evidence": "## 📄 Documentary Evidence",
        "Comparative Analysis": "## 📊 Comparative Analysis",
        "Recommendations": "## 💡 Recommendations"
    }
    
    for old, new in sections.items():
        response = response.replace(f"**{old}**", new)
        response = response.replace(f"{old}", new)
    
    # Add divider between sections
    for header in sections.values():
        response = response.replace(header, f"\n{header}\n")
    
    return response.strip()

def analyze_financials(results: list, query: str) -> dict:
    """Enhanced analysis with validation and quality scoring"""
    context = "\n\n".join([
        f"**Document {i+1}**: {result.node.metadata.get('company', 'Unknown Company')} "
        f"({result.node.metadata.get('symbol', 'N/A')})\n"
        f"• Type: {result.node.metadata.get('filing_type', 'Unknown')} "
        f"({result.node.metadata.get('fiscal_year', 'Unknown')})\n"
        f"• Source: {result.node.metadata.get('source_type', 'Unknown')}\n"
        f"• Excerpt:\n{result.node.text[:500]}{'...' if len(result.node.text) > 500 else ''}"
        for i, result in enumerate(results[:3])
    ])
    
    try:
        # Get raw response from LLM
        response = Settings.llm.complete(
            FINANCIAL_ANALYSIS_PROMPT.format(
                context_str=context,
                query_str=query
            )
        )
        
        formatted_response = str(response).strip()
        validation = validate_response(formatted_response)
        enhanced_response = enhance_response(formatted_response)
        
        return {
            "analysis": enhanced_response,
            "validation": validation,
            "context": context,
            "error": None
        }
        
    except Exception as e:
        return {
            "analysis": f"## Analysis Error\n{str(e)}",
            "validation": {
                "complete": False,
                "error": str(e),
                "quality_score": 0.0
            },
            "context": context,
            "error": str(e)
        }

class FinancialQuerySystem:
    def __init__(self, index, monitor, quality_metrics):
        self.index = index
        self.monitor = monitor
        self.quality_metrics = quality_metrics

    def execute_query(self, query: str) -> dict:
        """Execute a financial query with enhanced validation"""
        if not query.strip():
            return {
                "analysis": "Error: Query cannot be empty.",
                "validation": None,
                "context": None,
                "error": "Empty query"
            }

        try:
            start_time = time.time()
            results = hybrid_search(query)
            latency = time.time() - start_time
            
            analysis_result = analyze_financials(results, query)
            self.monitor.log_query(query, results, latency)
            
            if analysis_result["validation"]:
                self.quality_metrics.log_metrics(query, analysis_result["validation"])
            
            return analysis_result

        except Exception as e:
            return {
                "analysis": f"Error processing query: {str(e)}",
                "validation": None,
                "context": None,
                "error": str(e)
            }

def process_uploaded_file(uploaded_file, metadata=None):
    """Process an uploaded PDF file and add to index"""
    loader = FinancialDataLoader()
    
    # Save the file permanently to data directory
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Process with metadata
        nodes = loader.load_pdf(str(file_path), metadata)
        
        # Insert into index
        for node in nodes:
            index.insert_nodes([node])
        
        # Persist changes
        index.storage_context.persist()
        return len(nodes)
    except Exception as e:
        # Clean up if error occurs
        if file_path.exists():
            file_path.unlink()
        raise e

# =============================================
# 11. Enhanced Streamlit UI with Quality Metrics
# =============================================

def display_results(response: dict, latency: float):
    """Enhanced display with quality indicators"""
    # Header with performance info
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #4a90e2, #6a8eff);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin:0; color: white;">✨ Analysis Results</h2>
                <p style="margin:0; opacity: 0.9;">⚡ Processed in {latency:.2f} seconds</p>
            </div>
            <div style="font-size: 2rem;">📊</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show quality score badge
    if response.get("validation"):
        quality_score = response["validation"]["quality_score"]
        quality_color = "#4CAF50" if quality_score > 0.7 else "#FFC107" if quality_score > 0.4 else "#F44336"
        
        st.markdown(f"""
        <div style="
            background: {quality_color}10;
            border-left: 4px solid {quality_color};
            padding: 1rem;
            margin-bottom: 1.5rem;
            border-radius: 0 8px 8px 0;
        ">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <strong>Analysis Quality Score:</strong> 
                    <span style="color: {quality_color}; font-weight: bold;">{quality_score:.1%}</span>
                </div>
                <div>
                    {'✅ Complete' if response["validation"]["complete"] else '⚠️ Missing Sections'} | 
                    📄 {response["validation"]["citation_count"]} citations |
                    💡 {response["validation"]["recommendation_count"]} recommendations
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Show the analysis content
    st.markdown(response["analysis"])

    # Quality metrics cards
    st.markdown("---")
    st.markdown("""
    <style>
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: 100%;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4a90e2;
            margin: 0.5rem 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">⏱️</div>
            <div class="metric-value">{latency:.2f}s</div>
            <div class="metric-label">Response Time</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        score = response.get("validation", {}).get("quality_score", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">{'⭐' if score > 0.7 else '💫' if score > 0.4 else '⚠️'}</div>
            <div class="metric-value">{score:.0%}</div>
            <div class="metric-label">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        cites = response.get("validation", {}).get("citation_count", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">📑</div>
            <div class="metric-value">{cites}</div>
            <div class="metric-label">Citations</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        recs = response.get("validation", {}).get("recommendation_count", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">💡</div>
            <div class="metric-value">{recs}</div>
            <div class="metric-label">Recommendations</div>
        </div>
        """, unsafe_allow_html=True)

    # Document references
    st.markdown("---")
    st.markdown("### 📚 Source Documents")
    st.caption("These documents were used in the analysis")
    if "context" in response:
        st.text_area("Context Preview", value=response["context"], height=200, disabled=True)

def render_data_upload():
    st.header("📤 Upload Financial Documents")
    
    with st.form("upload_form"):
        # File uploader
        uploaded_file = st.file_uploader(
            "Select PDF file",
            type=["pdf"],
            accept_multiple_files=False
        )
        
        # Metadata collection
        st.markdown("### Document Metadata")
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name")
        with col2:
            symbol = st.text_input("Stock Symbol")
        
        fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=datetime.now().year)
        filing_type = st.selectbox("Filing Type", ["10-K", "10-Q", "8-K", "Other"])
        
        if st.form_submit_button("Upload and Index"):
            if uploaded_file:
                metadata = {
                    "company": company_name,
                    "symbol": symbol,
                    "fiscal_year": fiscal_year,
                    "filing_type": filing_type,
                    "upload_date": datetime.now().isoformat(),
                    "source_type": "user_upload"
                }
                
                with st.spinner("Processing document..."):
                    try:
                        num_nodes = process_uploaded_file(uploaded_file, metadata)
                        st.success(f"Successfully indexed {num_nodes} document sections")
                        st.experimental_rerun()  # Refresh to show new documents
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            else:
                st.warning("Please upload a PDF file")

def render_query_interface(query_system):
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter financial query:", 
                            placeholder="e.g., What is the succession plan for Warren Buffett at Berkshire Hathaway?")
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("Analyze", type="primary")

    if submit_btn and query:
        with st.spinner("Analyzing financial documents..."):
            start_time = time.time()
            response = query_system.execute_query(query)
            latency = time.time() - start_time
            
            display_results(response, latency)

def render_analytics_dashboard(monitor, query_system, quality_metrics):
    st.header("📊 Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Query Metrics", "Quality Trends", "System Health"])
    
    with tab1:
        st.markdown("### Query Performance Metrics")
        
        if not monitor.queries.empty:
            # Convert timestamp to datetime and set as index
            df = monitor.queries.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Calculate daily aggregates
            daily_metrics = df.resample('D').agg({
                'query': 'count',
                'latency': ['mean', 'max'],
                'results': 'mean'
            })
            daily_metrics.columns = ['query_count', 'avg_latency', 'max_latency', 'avg_results']
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", len(df))
            with col2:
                st.metric("Avg Latency", f"{df['latency'].mean():.2f}s")
            with col3:
                st.metric("Avg Results", f"{df['results'].mean():.1f}")
            
            # Time series charts
            st.markdown("#### Daily Trends")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.area_chart(daily_metrics['query_count'], use_container_width=True)
            with chart_col2:
                st.line_chart(daily_metrics[['avg_latency', 'max_latency']], use_container_width=True)
            
            # Top queries
            st.markdown("#### Frequent Queries")
            query_counts = df['query'].value_counts().head(10)
            st.bar_chart(query_counts)
            
            # Raw data
            st.markdown("#### Detailed Query Log")
            st.dataframe(df.sort_index(ascending=False))
            
            # Export button
            csv = df.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Query Data",
                data=csv,
                file_name="query_metrics.csv",
                mime="text/csv"
            )
        else:
            st.info("No query data available yet")
    
    with tab2:
        st.markdown("### Analysis Quality Metrics")
        
        if not quality_metrics.metrics.empty:
            # Convert timestamp to datetime and set as index
            qdf = quality_metrics.metrics.copy()
            qdf['timestamp'] = pd.to_datetime(qdf['timestamp'])
            qdf = qdf.set_index('timestamp')
            
            # Calculate weekly aggregates
            weekly_quality = qdf.resample('W').agg({
                'quality_score': 'mean',
                'citation_count': 'mean',
                'recommendations': 'mean',
                'complete': 'mean'
            })
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Quality Score", f"{qdf['quality_score'].mean():.1%}")
            with col2:
                st.metric("Avg Citations", f"{qdf['citation_count'].mean():.1f}")
            with col3:
                st.metric("Avg Recommendations", f"{qdf['recommendations'].mean():.1f}")
            
            # Quality trends
            st.markdown("#### Quality Over Time")
            st.line_chart(weekly_quality['quality_score'], use_container_width=True)
            
            # Correlation matrix
            st.markdown("#### Metric Correlations")
            corr_matrix = qdf[['quality_score', 'citation_count', 'recommendations']].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='Blues', vmin=-1, vmax=1))
            
            # Quality distribution
            st.markdown("#### Quality Score Distribution")
            hist_values = pd.cut(qdf['quality_score'], 
                                bins=[0, 0.3, 0.6, 0.8, 1.0],
                                labels=['Poor (<30%)', 'Fair (30-60%)', 'Good (60-80%)', 'Excellent (>80%)'])
            st.bar_chart(hist_values.value_counts())
            
            # Export button
            csv = qdf.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Quality Data",
                data=csv,
                file_name="quality_metrics.csv",
                mime="text/csv"
            )
        else:
            st.info("No quality metrics available yet")
    
    with tab3:
        st.markdown("### System Health Metrics")
        
        # Document statistics
        doc_count = len(index.docstore.docs)
        companies = len(set(
            doc.metadata.get('symbol', '') 
            for doc in index.docstore.docs.values() 
            if hasattr(doc, 'metadata') and 'symbol' in doc.metadata
        ))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", doc_count)
        with col2:
            st.metric("Unique Companies", companies)
        
        # Document types
        if doc_count > 0:
            st.markdown("#### Document Types")
            doc_types = [
                doc.metadata.get('filing_type', 'Unknown') 
                for doc in index.docstore.docs.values() 
                if hasattr(doc, 'metadata')
            ]
            type_counts = pd.Series(doc_types).value_counts()
            st.bar_chart(type_counts)
            
            # Document sources
            st.markdown("#### Data Sources")
            sources = [
                doc.metadata.get('source_type', 'Unknown') 
                for doc in index.docstore.docs.values() 
                if hasattr(doc, 'metadata')
            ]
            source_counts = pd.Series(sources).value_counts()
            st.bar_chart(source_counts)
            
            # Document timeline
            st.markdown("#### Document Timeline")
            try:
                dates = [
                    pd.to_datetime(doc.metadata.get('filed_date', None)) 
                    for doc in index.docstore.docs.values() 
                    if hasattr(doc, 'metadata') and 'filed_date' in doc.metadata
                ]
                if dates:
                    date_counts = pd.Series(dates).dropna().dt.to_period('M').value_counts().sort_index()
                    st.line_chart(date_counts)
            except:
                st.warning("Could not parse document dates")

def render_version_management(version_system):
    st.header("🔄 Version Management")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Create New Version")
        with st.form("version_form"):
            description = st.text_input("Version description:")
            if st.form_submit_button("Commit Version"):
                version_id = version_system.commit(index, description)
                st.success(f"Created version: {version_id}")
                st.rerun()  # Refresh to show new version
    
    with col2:
        st.markdown("### Version Control")
        
        if not version_system.versions:
            st.info("No versions available")
            return
            
        active_version = version_system.get_active_version()
        
        # Prepare version data for display
        version_data = []
        for version_id, details in version_system.versions.items():
            version_data.append({
                "Version ID": version_id,
                "Timestamp": details["timestamp"],
                "Description": details["description"],
                "Document Count": details["stats"]["doc_count"],
                "Active": version_id == active_version
            })
        
        # Create DataFrame
        versions_df = pd.DataFrame(version_data)
        
        # Display versions in a table
        st.dataframe(
            versions_df[["Timestamp", "Description", "Document Count", "Active"]],
            column_config={
                "Timestamp": "Timestamp",
                "Description": "Description",
                "Document Count": "Document Count",
                "Active": "Active"
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Version actions
        if len(version_system.versions) > 0:
            selected_version = st.selectbox(
                "Select a version:",
                options=list(version_system.versions.keys()),
                format_func=lambda x: f"{version_system.versions[x]['description']} ({version_system.versions[x]['timestamp']})"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Switch to Version", 
                           disabled=selected_version==active_version,
                           help="Switch to this version of the index"):
                    if version_system.switch_version(selected_version):
                        st.success(f"Switched to version {selected_version}")
                        st.rerun()
                    else:
                        st.error("Failed to switch version")
            
            with col2:
                if st.button("Delete Version", 
                            disabled=selected_version==active_version,
                            help="Permanently delete this version"):
                    if version_system.delete_version(selected_version):
                        st.success(f"Deleted version {selected_version}")
                        st.rerun()
                    else:
                        st.error("Failed to delete version")

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize core system components with caching"""
    print("Initializing system components...")
    monitor = QueryMonitor()
    version_system = VersionedIndex()
    quality_metrics = QualityMetrics()
    query_system = FinancialQuerySystem(index, monitor, quality_metrics)
    return query_system, monitor, version_system, quality_metrics

def main():
    st.set_page_config(
        page_title="Financial Analysis Workstation",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #4a90e2, #6a8eff);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        }
        .sidebar-header {
            background: linear-gradient(135deg, #4a90e2, #6a8eff);
            color: white;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #4a90e2, #6a8eff);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(74, 144, 226, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

    query_system, monitor, version_system, quality_metrics = initialize_system()

    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin:0;">📊 Financial Analysis Workstation</h1>
        <p style="margin:0; opacity: 0.9;">AI-powered insights for financial professionals</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3 style="color: white; margin:0;">⚙️ System Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_tab = st.radio("Navigation", [
            "🔍 Query Interface", 
            "📤 Data Upload", 
            "📊 Analytics Dashboard", 
            "🔄 Version Management"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("**📡 System Status**")
        st.metric("📑 Indexed Documents", len(index.docstore.docs))
        latest_version = list(version_system.versions.keys())[-1] if version_system.versions else "N/A"
        st.metric("🛠️ Active Version", version_system.versions.get(latest_version, {}).get("description", "N/A"))

    if selected_tab == "🔍 Query Interface":
        render_query_interface(query_system)
    elif selected_tab == "📤 Data Upload":
        render_data_upload()
    elif selected_tab == "📊 Analytics Dashboard":
        render_analytics_dashboard(monitor, query_system, quality_metrics)
    elif selected_tab == "🔄 Version Management":
        render_version_management(version_system)

if __name__ == "__main__":
    main()