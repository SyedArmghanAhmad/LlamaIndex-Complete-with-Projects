import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from PyPDF2 import PdfReader
from pathlib import Path
import os
from dotenv import load_dotenv
import re
import plotly.express as px
import pandas as pd

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .fadeIn {
        animation: fadeIn 1.5s ease-in-out;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with animations
st.markdown("""
    <div class="fadeIn">
        <h1 style="text-align: center; color: #4CAF50;">FinSight AI</h1>
        <h3 style="text-align: center; color: #333;">Your AI-Powered Financial Analyst</h3>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for PDF upload and settings
with st.sidebar:
    st.markdown("## Upload Financial PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk Size", min_value=256, max_value=1024, value=512, step=128)
    st.markdown("---")
    st.markdown("### Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=50)

# Function to process PDFs
def get_pdf_docs(uploaded_files, chunk_size):
    docs = []
    splitter = SentenceSplitter(chunk_size=chunk_size)
    
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        full_text = "\n".join([page.extract_text() for page in reader.pages])
        
        base_doc = Document(text=full_text, metadata={"source": uploaded_file.name, "type": "10-K"})
        nodes = splitter.get_nodes_from_documents([base_doc])
        for node in nodes:
            docs.append(Document(text=node.text, metadata=node.metadata))
    
    return docs

# Initialize Groq
def initialize_groq():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env file. Please add your Groq API key to the .env file.")
        st.stop()  # Stop the app if the API key is missing
    return Groq(model="llama3-70b-8192", api_key=groq_api_key)

# Extract numerical data from text
def extract_numerical_data(text):
    # Improved regex to capture financial metrics (e.g., $1,000, 10%, 1.5M)
    numerical_data = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?', text)
    # Clean and convert to float
    cleaned_data = []
    for num in numerical_data:
        num = num.replace('$', '').replace('%', '').replace(',', '')
        try:
            cleaned_data.append(float(num))
        except ValueError:
            continue
    return cleaned_data

# Decide which type of graph to generate
def decide_graph_type(query):
    if "trend" in query.lower():
        return "line"
    elif "comparison" in query.lower() or "compare" in query.lower():
        return "bar"
    elif "distribution" in query.lower():
        return "pie"
    elif "relationship" in query.lower() or "correlation" in query.lower():
        return "scatter"
    else:
        return "bar"  # Default to bar chart

# Generate a Plotly graph
def generate_graph(data, graph_type, query):
    if isinstance(data, list):
        df = pd.DataFrame(data, columns=["Value"])
    elif isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=["Category", "Value"])
    else:
        return None  # Unsupported data format

    if graph_type == "line":
        fig = px.line(df, y="Value", x=df.index if "Category" not in df.columns else "Category", title=query)
    elif graph_type == "bar":
        fig = px.bar(df, y="Value", x=df.index if "Category" not in df.columns else "Category", title=query)
    elif graph_type == "pie":
        fig = px.pie(df, names="Category" if "Category" in df.columns else df.index, values="Value", title=query)
    elif graph_type == "scatter":
        fig = px.scatter(df, y="Value", x=df.index if "Category" not in df.columns else "Category", title=query)
    else:
        fig = px.bar(df, y="Value", x=df.index if "Category" not in df.columns else "Category", title=query)  # Default to bar chart
    return fig

# Generate an answer using the LLM
def generate_answer(llm, query, contexts, temperature, max_tokens):
    combined_context = "\n".join(contexts)[:4000]
    prompt = f"""You are a financial analyst. Answer the following question based on the provided context.
    
    Question: {query}
    
    Relevant Context: {combined_context}
    
    Instructions:
    - If the question involves comparisons, rankings, or multiple data points, format the answer as a complete table with all requested data.
    - If the question requires step-by-step explanations or lists, use numbered or bullet points.
    - If data is missing, explicitly state "Data not available in the provided context."
    - Do not use placeholders like X, Y, or Z. If data is missing, leave the cell blank and add a note.
    - Suggest a type of graph (line, bar, pie, scatter) that would best represent the data in the answer.
    - Keep the answer concise, accurate, and well-formatted.
    
    Answer:"""
    
    try:
        response = llm.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        answer = response.text.strip()
        
        # Extract numerical data from the answer
        numerical_data = extract_numerical_data(answer)
        
        if numerical_data:
            # Decide on the type of graph to display
            graph_type = decide_graph_type(query)
            
            # Generate the graph
            fig = generate_graph(numerical_data, graph_type, query)
            
            return answer, fig
        else:
            return answer, "No numerical data found to generate a graph."
    except Exception as e:
        return f"Error generating answer: {str(e)}", None

# Main function
def main():
    if uploaded_files:
        financial_docs = get_pdf_docs(uploaded_files, chunk_size)
        
        embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
        vector_index = VectorStoreIndex.from_documents(financial_docs, embed_model=embed_model)
        
        vector_retriever = vector_index.as_retriever(similarity_top_k=3)
        bm25_retriever = BM25Retriever.from_defaults(nodes=financial_docs, similarity_top_k=3)
        
        llm = initialize_groq()
        
        query = st.text_input("Enter your financial question:")
        
        if query:
            with st.spinner("Analyzing your question..."):
                vector_results = vector_retriever.retrieve(query)
                bm25_results = bm25_retriever.retrieve(query)
                
                combined_results = {r.node.node_id: r for r in vector_results + bm25_results}
                contexts = [result.node.text for result in combined_results.values()]
                
                answer, fig_or_message = generate_answer(llm, query, contexts, temperature, max_tokens)
                
                st.markdown("### AI Answer:")
                st.markdown(f"<div class='fadeIn'>{answer}</div>", unsafe_allow_html=True)
                
                if isinstance(fig_or_message, str):
                    st.info(fig_or_message)
                else:
                    st.plotly_chart(fig_or_message, use_container_width=True)
    else:
        st.info("Please upload financial PDFs to get started.")

if __name__ == "__main__":
    main()