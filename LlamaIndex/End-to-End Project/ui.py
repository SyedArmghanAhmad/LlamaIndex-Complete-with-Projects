
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import json
from datetime import date
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, Settings, Document
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from dotenv import load_dotenv
import random
from faker import Faker
from llama_index.core import load_index_from_storage

# Initialize Faker for fake data generation
fake = Faker()

# Load environment variables
load_dotenv()

# =============================================
# 1. Generate Company Data
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
        "mdna": [fake.paragraph() for _ in range(10)]  # Management Discussion
    }

# Generate 50 companies (30+ pages of data)
companies = [generate_company() for _ in range(50)]

# =============================================
# 2. Fix JSON Serialization Error (Date Handling)
# =============================================

from datetime import datetime, date
import json

class CustomJSONEncoder(json.JSONEncoder):
    """Handles both date and datetime serialization"""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

# =============================================
# 3. Create Persistent Vector Store
# =============================================

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
chroma_collection = chroma_client.get_or_create_collection("sec_filings")

# Define vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# =============================================
# 4. Modified Document Creation with Proper Serialization
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
    
    # Convert date to string for serialization
    filed_date = fake.date_between(start_date="-5y", end_date="today")
    
    return Document(
        text=full_text,
        metadata={
            "company": company["company"],
            "symbol": company["symbol"],
            "filing_type": random.choice(["10-K", "10-Q", "8-K"]),
            "fiscal_year": company["fiscal_year"],
            "filed_date": filed_date.isoformat(),
            "full_text": full_text  # Critical addition
        }
    )


# =============================================
# 5. Configure Settings with Persistence
# =============================================

# Initialize components
Settings.llm = Groq(model="mixtral-8x7b-32768")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

# =============================================
# 6. FIXED Index Creation and Loading
# =============================================

persist_dir = "./storage"
if not Path(persist_dir).exists():
    print("Creating new index...")
    sec_documents = [create_sec_document(c) for c in companies]
    
    # Create index with explicit docstore
    index = VectorStoreIndex.from_documents(
        sec_documents,
        storage_context=storage_context,
        show_progress=True,
        store_nodes_override=True  # Ensure nodes are stored
    )
    
    # Verify node storage
    if not index.docstore.docs:
        raise ValueError("Failed to store nodes in docstore!")
    
    # Persist properly
    index.storage_context.persist(persist_dir=persist_dir)
    
    # Get nodes directly from docstore
    all_nodes = list(index.docstore.docs.values())
    print(f"Created new index with {len(all_nodes)} nodes")
    print(f"Sample node text: {all_nodes[0].text[:200]}...")
else:
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir,
        vector_store=vector_store
    )
    
    # Load with node verification
    index = load_index_from_storage(storage_context)
    
    # Get nodes with text recovery
    all_nodes = []
    for node_id in storage_context.docstore.docs:
        node = storage_context.docstore.get_node(node_id)
        
        # Ensure text exists
        if not hasattr(node, "text"):
            if hasattr(node, "content"):
                node.text = node.content
            elif "full_text" in node.metadata:
                node.text = node.metadata["full_text"]
            else:
                print(f"Skipping node {node_id} - no text found")
                continue
                
        all_nodes.append(node)
    
    print(f"Loaded {len(all_nodes)} nodes")
    if all_nodes:
        print(f"Sample node text: {all_nodes[0].text[:200]}...")

# Final validation
if not all_nodes:
    raise ValueError("""
    CRITICAL: No nodes loaded!
    Possible fixes:
    1. Delete ./storage and ./chroma_db folders
    2. Ensure create_sec_document() returns valid Documents
    3. Verify storage permissions
    """)

# %%
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
import Stemmer  # PyStemmer library

# Convert documents to TextNodes with validation
text_nodes = []
for node in all_nodes:
    # Handle different node types and legacy formats
    if hasattr(node, "text"):
        clean_text = node.text.strip()
    elif hasattr(node, "content"):  # Backwards compatibility
        clean_text = node.content.strip()
    else:
        continue
    
    if not clean_text:
        continue
    
    text_nodes.append(TextNode(
        text=clean_text,
        metadata=node.metadata,
        excluded_embed_metadata_keys=["filed_date"],
        excluded_llm_metadata_keys=["filed_date"],
        id_=node.node_id
    ))

if not text_nodes:
    # Enhanced error diagnostics
    sample_nodes = [n.__dict__ for n in all_nodes[:3]]
    raise ValueError(f"""
    No valid text nodes found for BM25. 
    First 3 nodes sample: {json.dumps(sample_nodes, indent=2)}
    """)

# Initialize BM25 with explicit parameters
bm25_retriever = BM25Retriever.from_defaults(
    nodes=text_nodes,  # Use validated nodes
    similarity_top_k=10,
    stemmer=Stemmer.Stemmer("english"),
    token_pattern=r"(?u)\b\w+\b",  # Broader token pattern
    skip_stemming=False
)

vector_retriever = index.as_retriever(similarity_top_k=5)

def hybrid_search(query):
    """Production-grade hybrid search with score fusion"""
    query_bundle = QueryBundle(query_str=query)
    
    # Retrieve from both systems
    vector_results = vector_retriever.retrieve(query_bundle)
    bm25_results = bm25_retriever.retrieve(query_bundle)

    # RRF fusion with position weighting
    combined_scores = {}
    for rank, result in enumerate(vector_results):
        combined_scores[result.node.node_id] = combined_scores.get(result.node.node_id, 0) + 1/(rank + 60)
    
    for rank, result in enumerate(bm25_results):
        combined_scores[result.node.node_id] = combined_scores.get(result.node.node_id, 0) + 1/(rank + 60)

    # Sort and select top 7
    sorted_results = sorted(combined_scores.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:7]
    
    # Preserve order from both result sets
    final_results = []
    seen_ids = set()
    for result in vector_results + bm25_results:
        if result.node.node_id in dict(sorted_results) and result.node.node_id not in seen_ids:
            final_results.append(result)
            seen_ids.add(result.node.node_id)
    
    return final_results[:7]


# Example 1: Sector Analysis
response = hybrid_search(
    "Compare risk factors in the technology sector versus healthcare sector "
    "from 2020-2023 10-K filings. Identify common themes and sector-specific risks."
)

# Example 2: Financial Health Check
response = hybrid_search(
    "List companies with negative EPS growth but increasing R&D expenditure "
    "from their latest 10-Q filings. Include financial metrics and CEO commentary."
)

# Example 3: M&A Analysis
response = hybrid_search(
    "Identify all 8-K filings related to mergers and acquisitions in the "
    "past 3 years. Analyze deal sizes and strategic rationales provided."
)


import pandas as pd
import time
from datetime import datetime

class QueryMonitor:
    def __init__(self):
        # Initialize DataFrame with explicit column types
        self.queries = pd.DataFrame(columns=["timestamp", "query", "latency", "results"])
    
    def log_query(self, query, results, latency):
        """Log a query with its results and latency."""
        new_entry = pd.DataFrame({
            "timestamp": [datetime.now()],
            "query": [query],
            "latency": [latency],
            "results": [len(results)]
        })
        
        # Concatenate while preserving column types
        self.queries = pd.concat(
            [self.queries, new_entry],
            ignore_index=True,
            axis=0
        )
    
    def generate_report(self):
        """Generate a daily report of query analytics."""
        if self.queries.empty:
            return pd.DataFrame()  # Return empty DataFrame if no queries logged
        
        # Group by day and aggregate metrics
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
# =============================================
# Final Working VersionedIndex Class
# =============================================
import uuid
from datetime import datetime

class VersionedIndex:
    def __init__(self):
        self.versions = {}
    
    def commit(self, index, description):
        version_id = str(uuid.uuid4())
        index.storage_context.persist(persist_dir=f"./index_{version_id}")
        
        # Clean metadata for all documents
        cleaned_docs = []
        for doc in index.docstore.docs.values():
            # Convert all metadata values to strings
            safe_metadata = {
                k: v.isoformat() if isinstance(v, (date, datetime)) else str(v)
                for k, v in doc.metadata.items()
            }
            cleaned_docs.append({
                "id": doc.id_,
                "metadata": safe_metadata
            })
        
        # Create version entry with safe data
        self.versions[version_id] = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "stats": {
                "doc_count": len(index.docstore.docs),
                "companies": list(set(
                    d["metadata"]["symbol"] for d in cleaned_docs
                    if "symbol" in d["metadata"]
                ))
            }
        }
        
        # Save with custom encoder
        with open("versions.json", "w") as f:
            json.dump(
                self.versions,
                f,
                cls=CustomJSONEncoder,
                indent=2,
                ensure_ascii=False
            )
            
        return version_id

from llama_index.core import PromptTemplate

# Enhanced Analysis Prompt
EXPERT_ANALYSIS_PROMPT = PromptTemplate("""\
**Role**: Senior Financial Analyst at Goldman Sachs  
**Task**: Analyze SEC filings for sector comparison.  
**Context**: {context_str}  
**Query**: {query_str}  

**Response Format**:  
1. **Executive Summary**  
   - Summarize key findings in 2-3 sentences.  
   - Highlight major differences between sectors.  

2. **Sector Performance Overview**  
   - Create a table comparing:  
     - Revenue growth (YoY)  
     - Profit margins  
     - R&D spend (% of revenue)  
     - Cash reserves  
     - Debt ratios  
   - Include specific numbers from filings.  

3. **Risk Analysis**  
   - For each sector, list top 3 risks with:  
     - Risk name  
     - Severity (High/Medium/Low)  
     - Mitigation strategies cited in filings  
   - Use direct quotes where available.  

4. **Strategic Recommendations**  
   - Provide 3 actionable recommendations per sector.  
   - Base recommendations on filing insights.  

5. **Critical Insights from Filings**  
   - Include 2-3 direct quotes per sector.  
   - Explain the significance of each quote.  

6. **Emerging Trends**  
   - Identify 2-3 trends supported by filing data.  
   - Highlight potential future impacts.  

**Tone**: Professional, data-driven, and concise.  
**Style**: Use bullet points, tables, and clear headings.  
**Data**: Always cite specific filings and metrics.  
""")


def analyze_financials(results, query):
    """Generate expert analysis using actual document context"""
    # Build context from search results
    context = "\n\n".join([
        f"Document {i+1}: {result.node.metadata['company']} ({result.node.metadata['symbol']})\n"
        f"Filing Type: {result.node.metadata['filing_type']}\n"
        f"Content: {result.node.text[:1000]}..."
        for i, result in enumerate(results[:5])  # Use top 5 results
    ])
    
    # Generate analysis
    response = Settings.llm.complete(
        EXPERT_ANALYSIS_PROMPT.format(
            context_str=context,
            query_str=query
        )
    )
    return str(response)

class FinancialQuerySystem:
    def __init__(self, index, monitor):
        self.index = index
        self.monitor = monitor

    def execute_query(self, query):
        """Execute a financial query and return expert analysis."""
        if not query.strip():
            return "Error: Query cannot be empty."

        print(f"\nProcessing query: '{query}'")
        
        try:
            # Step 1: Hybrid Search
            start_time = time.time()
            results = hybrid_search(query)
            latency = time.time() - start_time
            
            # Log the query
            self.monitor.log_query(query, results, latency)
            
            # Step 2: Generate Expert Analysis
            analysis = analyze_financials(results, query)
            
            return analysis

        except Exception as e:
            return f"Error processing query: {str(e)}"




#[Streamlit UI - Final Implementation]
import streamlit as st
import pandas as pd
import time

# Initialize system components with proper caching
@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize core system components with caching"""
    monitor = QueryMonitor()
    version_system = VersionedIndex()
    query_system = FinancialQuerySystem(index, monitor)
    return query_system, monitor, version_system

# Streamlit App Configuration
st.set_page_config(
    page_title="Financial Analysis Workstation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #2a3f5f;
    }
    .stTextInput input {
        border-radius: 4px;
    }
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border-radius: 4px;
    }
    .analysis-section {
        border-left: 4px solid #4a90e2;
        padding-left: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# System initialization
query_system, monitor, version_system = initialize_system()

# Main Interface
def main():
    st.title("📊 Financial Analysis Workstation")
    st.markdown("### SEC Filing Analysis Platform")

    # Sidebar Controls
    with st.sidebar:
        st.header("System Controls")
        selected_tab = st.radio("Navigation", [
            "Query Interface", 
            "Analytics Dashboard", 
            "Version Management"
        ])
        
        st.markdown("---")
        st.markdown("**System Status**")
        st.metric("Indexed Documents", len(all_nodes))
        latest_version = list(version_system.versions.keys())[-1] if version_system.versions else "N/A"
        st.metric("Active Version", version_system.versions.get(latest_version, {}).get("description", "N/A"))

    # Tab routing
    if selected_tab == "Query Interface":
        render_query_interface()
    elif selected_tab == "Analytics Dashboard":
        render_analytics_dashboard()
    elif selected_tab == "Version Management":
        render_version_management()

# Query Interface Components
def render_query_interface():
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter financial query:", 
                            placeholder="e.g., Compare tech and healthcare sector risks")
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("Analyze", type="primary")

    if submit_btn and query:
        with st.spinner("Analyzing SEC filings..."):
            start_time = time.time()
            response = query_system.execute_query(query)
            latency = time.time() - start_time
            
            display_results(response, latency)

def display_results(response, latency):
    st.success(f"Analysis completed in {latency:.2f}s")
    
    sections = {
        "Executive Summary": "📌",
        "Sector Performance Overview": "📊",
        "Risk Analysis": "⚠️",
        "Strategic Recommendations": "🚀",
        "Critical Insights from Filings": "💡",
        "Emerging Trends": "📈"
    }
    
    for section, icon in sections.items():
        if f"**{section}**" in response:
            section_content = response.split(f"**{section}**")[1].split("**")[0]
            with st.expander(f"{icon} {section}", expanded=(section == "Executive Summary")):
                st.markdown(section_content)
        else:
            st.warning(f"Section {section} not found in response")

# Analytics Dashboard Components
def render_analytics_dashboard():
    st.header("Query Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Daily Summary")
        report = monitor.generate_report()
        if not report.empty:
            st.dataframe(report.style.format({"avg_latency": "{:.2f}s", "avg_results": "{:.1f}"}))
        else:
            st.info("No query data available")
    
    with col2:
        st.markdown("### Historical Data")
        if not monitor.queries.empty:
            st.dataframe(monitor.queries)
            csv = monitor.queries.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full History",
                data=csv,
                file_name="query_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No historical data available")

# Version Management Components
def render_version_management():
    st.header("Index Version Control")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Create New Version")
        description = st.text_input("Version description:")
        if st.button("Commit Version"):
            version_id = version_system.commit(index, description)
            st.success(f"Created version: {version_id}")
    
    with col2:
        st.markdown("### Existing Versions")
        try:
            with open("versions.json") as f:
                versions = json.load(f)
                st.json(versions)
        except FileNotFoundError:
            st.error("No versions found")

if __name__ == "__main__":
    main()

# Run with: streamlit run your_script.py

