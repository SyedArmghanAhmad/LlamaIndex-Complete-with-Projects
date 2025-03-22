import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure settings
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512  # Smaller chunks for faster processing
Settings.chunk_overlap = 50

# Load and index documents with persistent storage
persist_dir = "./storage"
if not os.path.exists(persist_dir):
    documents = SimpleDirectoryReader("medical_papers").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = VectorStoreIndex([], storage_context=storage_context)

# Create hybrid retriever
vector_retriever = index.as_retriever(similarity_top_k=10)  # Increased from 7
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)

class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query):
        vector_nodes = self.vector_retriever.retrieve(query)
        bm25_nodes = self.bm25_retriever.retrieve(query)
        all_nodes = vector_nodes + bm25_nodes
        return sorted(all_nodes, key=lambda n: n.score, reverse=True)[:15]  # Top 15

# Create query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=HybridRetriever(vector_retriever, bm25_retriever),
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.75),  # Increased from 0.7
        KeywordNodePostprocessor(
            required_keywords=["treatment", "diabetes", "hypertension", "AI"],  # More specific
            exclude_keywords=["animal", "vitro", "study", "unrelated"]  # Exclude noisy terms
        )
    ],
    response_mode="compact"  # Faster than tree_summarize
)

# Warm-up phase
def warmup():
    _ = query_engine.retriever.retrieve("warmup")
    _ = Settings.llm.complete("warmup")

warmup()

# Cache to store responses
query_cache = {}

# =========================================
# Custom CSS for Medical Theme + Animations
# =========================================
st.markdown("""
    <style>
        :root {
            --primary-color: #2A5C82;
            --secondary-color: #5BA4E6;
            --accent-color: #FF6B6B;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .main {
            background: linear-gradient(135deg, #f7f9fb 0%, #e3f2fd 100%);
            border-radius: 15px;
            padding: 2rem;
            animation: fadeIn 0.5s ease-out;
        }

        .stButton>button {
            background: var(--primary-color) !important;
            color: white !important;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(42, 92, 130, 0.3);
        }

        .response-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 4px solid var(--secondary-color);
        }

        .source-item {
            padding: 1rem;
            margin: 0.5rem 0;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            transition: transform 0.2s ease;
        }

        .source-item:hover {
            transform: translateX(10px);
            background: rgba(235, 245, 255, 0.9);
        }

        .stethoscope-animation {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# ======================
# Enhanced UI Components
# ======================
def medical_header():
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--primary-color); 
                      font-family: 'Arial Rounded MT Bold', sans-serif;
                      margin-bottom: 0.5rem;">
                🩺 MedAI Assistant
            </h1>
            <p style="color: #666; font-size: 1.1rem;">
                Advanced Medical Intelligence powered by AI
            </p>
        </div>
    """, unsafe_allow_html=True)

def animated_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style="background: var(--primary-color); 
                       padding: 1.5rem; 
                       border-radius: 12px;
                       color: white;
                       text-align: center;">
                <h2 style="color: white;">⚕️ About</h2>
                <p>Trusted by medical professionals worldwide</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="margin-top: 2rem;">
                <h3>📚 Knowledge Base</h3>
                <p>• Diabetes Research<br>
                   • Hypertension Studies<br>
                   • AI in Radiology</p>
            </div>
        """, unsafe_allow_html=True)

# ======================
# State-of-the-Art Prompt Engineering
# ======================
def generate_prompt(query):
    """
    Generates a detailed, context-rich prompt for the medical assistant.
    """
    prompt = f"""
    You are a board-certified medical expert with access to the latest clinical guidelines and peer-reviewed research. 
    Provide a detailed, evidence-based response to the following question. 
    Include step-by-step explanations, mechanisms of action, and references to recent studies where applicable.

    Question: {query}

    Instructions:
    1. Be concise but thorough.
    2. Use bullet points or numbered lists for clarity.
    3. Cite sources if available.
    4. Avoid unproven or harmful recommendations.
    """
    return prompt

# ======================
# Main App Interface
# ======================
medical_header()
animated_sidebar()

# Search Section
with st.container():
    col1, col2 = st.columns([4, 1])  # Adjust the ratio for better alignment
    with col1:
        query = st.text_input("", 
                            placeholder="Enter your medical question...", 
                            key="search_input",
                            help="Ask about treatments, research findings, or clinical guidelines")
    with col2:
        st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)  # Adjust margin for alignment
        if st.button("Search", key="main_search"):
            st.session_state.search_triggered = True
        st.markdown("</div>", unsafe_allow_html=True)

# Response Handling
if 'search_triggered' in st.session_state:
    if not query or len(query.split()) < 2:
        st.warning("🔍 Please enter a valid medical question (at least 2 words)")
    else:
        # Check cache
        if query in query_cache:
            with st.container():
                st.markdown(f"""
                    <div class="response-card" style="border-color: var(--accent-color);">
                        <h3 style="color: var(--primary-color); margin-bottom: 1rem;">⚡ Cached Answer</h3>
                        <div style="font-size: 1.1rem; line-height: 1.6;">
                            {query_cache[query]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            with st.spinner("🔬 Analyzing medical literature..."):
                # Use the enhanced prompt
                response = query_engine.query(generate_prompt(query))
                
                if not response.source_nodes:
                    st.markdown("""
                        <div class="response-card">
                            <h3 style="color: var(--accent-color);">⚠️ No Results Found</h3>
                            <p>Try these topics instead:</p>
                            <ul>
                                <li>Recent advancements in diabetes treatment</li>
                                <li>Hypertension management guidelines</li>
                                <li>AI applications in medical imaging</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Main Answer Card
                    st.markdown(f"""
                        <div class="response-card">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                                <div style="background: var(--primary-color); 
                                         width: 50px; 
                                         height: 50px; 
                                         border-radius: 50%;
                                         display: flex;
                                         align-items: center;
                                         justify-content: center;">
                                    <span style="color: white; font-size: 1.5rem;">⚕️</span>
                                </div>
                                <h3 style="color: var(--primary-color); margin: 0;">Clinical Summary</h3>
                            </div>
                            <div style="font-size: 1.1rem; line-height: 1.6; color: #333;">
                                {response}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# Floating Status
st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px;
              background: white; padding: 1rem; border-radius: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; align-items: center; gap: 0.5rem;">
        <div class="stethoscope-animation">🩺</div>
        <span style="color: var(--primary-color);">MedAI Assistant v1.0</span>
    </div>
""", unsafe_allow_html=True)