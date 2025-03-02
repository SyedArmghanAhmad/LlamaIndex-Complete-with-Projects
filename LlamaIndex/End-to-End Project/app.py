# %%
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
            "filed_date": filed_date.isoformat()  # Convert to ISO format string
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
# 6. Create/Load Index with Persistence Check
# =============================================
persist_dir = "./storage"
if not Path(persist_dir).exists():
    # Create new index
    sec_documents = [create_sec_document(c) for c in companies]

    index = VectorStoreIndex.from_documents(
        sec_documents,
        storage_context=storage_context,
        show_progress=True
    )

    # Persist index to disk
    index.storage_context.persist(persist_dir=persist_dir)
    all_nodes = [node for node in index.docstore.docs.values()]
else:
    # ⚠️ CORRECTED LOADING MECHANISM ⚠️
    # Load entire storage context including docstore
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir,
        vector_store=vector_store  # Keep our Chroma connection
    )

    # Load index with full storage context
    index = load_index_from_storage(storage_context)

    # Get nodes from properly loaded docstore
    all_nodes = list(storage_context.docstore.docs.values())

print("Index created/loaded successfully!")

# %%
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode
import Stemmer  # PyStemmer library

# Convert documents to TextNodes explicitly
text_nodes = [
    TextNode(
        text=doc.text,
        metadata=doc.metadata,
        excluded_embed_metadata_keys=["filed_date"],  # Exclude dates from embedding
        excluded_llm_metadata_keys=["filed_date"]     # Exclude dates from LLM context
    )
    for doc in sec_documents
]

# Initialize stemmer for BM25
stemmer = Stemmer.Stemmer("english")  # Use English stemmer

# Initialize retrievers
vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=text_nodes,  # Use explicit text nodes
    similarity_top_k=10,
    stemmer=stemmer,   # Use PyStemmer for tokenization
    skip_stemming=False  # Enable stemming for better term matching
)

def hybrid_search(query):
    """Production-grade hybrid search with score fusion"""
    query_bundle = QueryBundle(query_str=query)

    # Retrieve results from both retrievers
    vector_results = vector_retriever.retrieve(query_bundle)
    bm25_results = bm25_retriever.retrieve(query_bundle)

    # Combine using Reciprocal Rank Fusion (RRF)
    combined_results = {}
    for rank, result in enumerate(vector_results + bm25_results):
        doc_id = result.node.node_id
        score = 1 / (rank + 1)
        combined_results[doc_id] = combined_results.get(doc_id, 0) + score

    # Sort and return top 7
    sorted_results = sorted(combined_results.items(),
                          key=lambda x: x[1],
                          reverse=True)[:7]
    return [r for r in vector_results + bm25_results
           if r.node.node_id in dict(sorted_results)]


# %%
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

# %%
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

# Usage Example
monitor = QueryMonitor()

# Simulate a query
start = time.time()
results = hybrid_search("Tech company risks")  # Replace with your hybrid search function
latency = time.time() - start

# Log the query
monitor.log_query("Tech company risks", results, latency)

# Generate and display the report
report = monitor.generate_report()
print(report)

# %%
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

# %%
# Simple search
results = hybrid_search("Companies with revenue over $100B")
print("Top Results:")
for idx, result in enumerate(results[:3]):
    # Extract revenue safely
    revenue_line = result.node.text.split('Revenue: ')[1]
    revenue = revenue_line.split('\n')[0]

    print(f"{idx+1}. {result.node.metadata['company']}")
    print(f"   Revenue: {revenue}")
    print(f"   Score: {result.score:.2f}\n")

# %%
from llama_index.core import PromptTemplate

analyst_prompt = PromptTemplate("""\
You are a senior financial analyst. Analyze these SEC filings:
{context_str}

Query: {query_str}

Format your response with:
1. Key Findings
2. Supporting Data
3. Recommendations
""")

def analyze_financials(query):
    results = hybrid_search(query)
    context = "\n\n".join([n.node.text for n in results])

    response = Settings.llm.complete(
        analyst_prompt.format(context_str=context, query_str=query)
    )

    return str(response)

# Example: Risk Factor Analysis
analysis = analyze_financials(
    "Identify top 3 emerging risks in the tech sector from recent 10-K filings"
)
print(analysis)

# %%
def end_to_end_test(query):
    # Step 1: Hybrid Search
    start_time = time.time()
    results = hybrid_search(query)
    retrieval_time = time.time() - start_time

    # Step 2: LLM Analysis
    analysis = analyze_financials(query)

    # Step 3: Logging
    monitor.log_query(query, results, retrieval_time)

    # Step 4: Versioning
    if "critical" in query.lower():
        version_id = version_system.commit(index, f"Critical query: {query}")

    return {
        "analysis": analysis,
        "retrieval_time": f"{retrieval_time:.2f}s",
        "documents_used": len(results)
    }

# Test Complex Query
response = end_to_end_test(
    "Identify companies with decreasing gross margins but increasing R&D spend. "
    "What does this suggest about their strategic priorities?"
)
print(json.dumps(response, indent=2))

# %%
version_system = VersionedIndex()
version_id = version_system.commit(index, "Final Production Version")

with open("versions.json") as f:
    print(json.dumps(json.load(f), indent=2))

# %%
# Test empty results handling
response = end_to_end_test("Find companies in Mars colony sector")
print(response["analysis"])  # Should handle gracefully


