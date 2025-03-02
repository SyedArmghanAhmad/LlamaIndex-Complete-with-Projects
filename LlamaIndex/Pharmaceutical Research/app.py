import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize models
groq_api_key = os.getenv("GROQ_API_KEY")
Settings.llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ====================== [UI Configuration] ======================
st.set_page_config(page_title="Clinical Trial Explorer", page_icon="🩺", layout="wide")

def set_custom_style():
    st.markdown("""
    <style>
        .header {color: #2e86c1; font-size: 24px !important;}
        .subheader {color: #5dade2; font-size: 18px !important;}
        .st-expander {border: 1px solid #d6eaf8 !important; border-radius: 8px !important;}
        .stTextInput>div>div>input {border-radius: 20px !important;}
        .stButton>button {border-radius: 20px !important; background: #2e86c1 !important;}
    </style>
    """, unsafe_allow_html=True)

set_custom_style()


### Expanded Clinical Trial Dataset
expanded_data = {
    "NCTId": [
        "NCT04512345", "NCT05678901", "NCT03456789", "NCT07890123",
        "NCT09876543", "NCT11223344", "NCT22334455", "NCT33445566",
        "NCT44556677", "NCT55667788"
    ],
    "Title": [
        "Study of Drug X in Alzheimer's Patients",
        "Trial of Compound Y for Early Alzheimer's",
        "Investigation of Drug Z for Parkinson's",
        "Study of Therapy A for Multiple Sclerosis",
        "Study of Drug B for Alzheimer's",
        "Long-term Cognitive Outcomes in Alzheimer's",
        "Gene Therapy Trial for Parkinson's Disease",
        "Stem Cell Treatment for Multiple Sclerosis",
        "Preventive Vaccine for Alzheimer's",
        "Digital Therapy for Parkinson's Symptom Management"
    ],
    "Condition": [
        "Alzheimer's Disease", "Alzheimer's Disease",
        "Parkinson's Disease", "Multiple Sclerosis",
        "Alzheimer's Disease", "Alzheimer's Disease",
        "Parkinson's Disease", "Multiple Sclerosis",
        "Alzheimer's Disease", "Parkinson's Disease"
    ],
    "Phase": [
        "Phase 2", "Phase 3", "Phase 1", "Phase 2",
        "Phase 2", "Phase 4", "Phase 1/2", "Phase 3",
        "Phase 2", "Phase 3"
    ],
    "Status": [
        "Recruiting", "Completed", "Active", "Recruiting",
        "Recruiting", "Completed", "Not yet recruiting", "Active",
        "Terminated", "Recruiting"
    ],
    "StartDate": [
        "2023-01-01", "2022-06-15", "2023-03-01", "2023-05-01",
        "2023-07-01", "2021-09-01", "2024-01-01", "2023-10-01",
        "2023-04-01", "2023-12-01"
    ],
    "CompletionDate": [
        "2024-12-31", "2023-12-31", "2024-06-30", "2025-01-31",
        "2024-12-31", "2023-06-30", "2026-12-31", "2025-06-30",
        "2023-12-31", "2025-06-30"
    ],
    "Interventions": [
        ["Drug X", "Placebo"],
        ["Compound Y", "Cognitive Therapy"],
        ["Drug Z"],
        ["Therapy A", "Placebo"],
        ["Drug B", "Placebo"],
        ["Cognitive Assessment", "MRI Scans"],
        ["Gene Therapy Vector"],
        ["Stem Cell Injection"],
        ["Vaccine X-101"],
        ["Digital Therapy App"]
    ],
    "Description": [
        "A study to evaluate the efficacy of Drug X in Alzheimer's patients.",
        "A trial to assess the safety and efficacy of Compound Y in early Alzheimer's.",
        "An investigation into the pharmacokinetics of Drug Z in Parkinson's patients.",
        "A study to evaluate the effectiveness of Therapy A in Multiple Sclerosis patients.",
        "A study to evaluate the efficacy of Drug B in Alzheimer's patients.",
        "Long-term follow-up study of cognitive outcomes in Alzheimer's patients.",
        "First-in-human trial of gene therapy for Parkinson's disease.",
        "Phase 3 trial of stem cell therapy for progressive Multiple Sclerosis.",
        "Phase 2 trial of preventive vaccine for high-risk Alzheimer's patients (terminated due to safety concerns).",
        "Digital therapy app for managing Parkinson's motor symptoms."
    ]
}

trials_df = pd.DataFrame(expanded_data)

### Document Conversion
# %%
def trials_to_documents(trials_df):
    """Convert DataFrame to LlamaIndex Documents with complete metadata"""
    documents = []
    for _, row in trials_df.iterrows():
        metadata = {
            "NCTId": row["NCTId"],
            "Title": row["Title"],
            "Condition": row["Condition"],
            "Phase": row["Phase"],
            "Status": row["Status"],
            "Interventions": ", ".join(row["Interventions"]),
            "StartDate": row["StartDate"],
            "CompletionDate": row["CompletionDate"],
            "Description": row["Description"]
        }
        text = f"{row['Description']}"  # Simplified text content
        documents.append(Document(text=text, metadata=metadata))
    return documents

def _explain_phase(phase):
    return {
        "Phase 1": "Initial safety testing in small groups (20-80 participants)",
        "Phase 2": "Efficacy and side effect testing in larger groups (100-300)",
        "Phase 3": "Large-scale testing for regulatory approval (1,000-3,000)",
        "Phase 4": "Post-marketing surveillance after approval",
        "Phase 1/2": "Combined safety and preliminary efficacy testing"
    }.get(phase, "Clinical trial stage")

def _explain_status(status):
    return {
        "Recruiting": "Currently accepting participants",
        "Completed": "Finished data collection, results analysis ongoing",
        "Active": "Ongoing but not currently recruiting",
        "Terminated": "Permanently stopped before completion",
        "Not yet recruiting": "Approved but not yet started"
    }.get(status, "Trial status")


### Query Engine
def get_query_engine(index, filters=None):
    """Create configured query engine"""
    return index.as_query_engine(
        similarity_top_k=5,
        filters=filters,
        verbose=False
    )
### Enhanced Response Formatting
def format_response(response, question):
    """Generate human-friendly explanations with context"""
    if not response or not response.source_nodes:
        return "No matching trials found based on your criteria."
    
    # Generate natural language summary
    base_prompt = f"""Context: {response}
    Task: Explain these clinical trial results to a healthcare worker.
    Requirements:
    1. Answer the question: {question}
    2. Highlight key trial aspects
    3. Explain technical terms in parentheses
    4. Use bullet points
    5. Keep under 300 words"""
    
    summary = Settings.llm.complete(base_prompt).text
    
    # Build detailed metadata section
    details = []
    for node in response.source_nodes[:3]:  # Show top 3 most relevant
        meta = node.metadata
        details.append(f"""
        Trial ID: {meta['NCTId']}
        • Title: {meta['Title']}
        • Condition: {meta['Condition']}
        • Phase: {meta['Phase']} ({_explain_phase(meta['Phase'])})
        • Status: {meta['Status']} ({_explain_status(meta['Status'])})
        • Interventions: {meta['Interventions']}
        • Timeline: {meta['StartDate']} to {meta['CompletionDate']}
        """)
    
    return f"""
    {summary.strip()}
    
    Detailed Information:
    {"".join(details)}
    """
    
def query_trials(index, question, phase=None, status=None):
    """Query the index with optional metadata filters"""
    filters = []
    if phase:
        filters.append(MetadataFilter(key="Phase", value=phase))
    if status:
        filters.append(MetadataFilter(key="Status", value=status))
    
    # Correct initialization
    metadata_filters = MetadataFilters(filters=filters) if filters else None
    
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=metadata_filters
    )
    return query_engine.query(question)


# ====================== [Streamlit UI] ======================
# ====================== [Streamlit UI] ======================
def main():
    st.title("🔍 Clinical Trial Intelligence System")
    st.markdown("Explore clinical trial data using natural language queries")
    
    # Initialize index once
    if "index" not in st.session_state:
        with st.spinner("⚙️ Loading trial data..."):
            trials_df = pd.DataFrame(expanded_data)
            documents = trials_to_documents(trials_df)
            st.session_state.index = VectorStoreIndex.from_documents(
                documents, 
                embed_model=Settings.embed_model
            )

    # Input Section
    st.markdown("### 📝 Enter Your Query")
    question = st.text_input("Ask about clinical trials:", placeholder="e.g., 'What Alzheimer's trials are in Phase 2?'")
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2 = st.columns(2)
    with col1:
        phase = st.selectbox("Phase", [None, "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 1/2"])
    with col2:
        status = st.selectbox("Status", [None, "Recruiting", "Completed", "Active", "Terminated"])

    # Query Execution
    if st.button("Search Trials", use_container_width=True):
        if question:
            with st.spinner("🔍 Analyzing trials..."):
                try:
                    response = query_trials(
                        index=st.session_state.index,
                        question=question,
                        phase=phase,
                        status=status
                    )
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("📄 Results Summary")
                    formatted_response = format_response(response, question)
                    st.markdown(formatted_response)
                    
                except Exception as e:
                    st.error(f"🚨 Error processing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question to search")

if __name__ == "__main__":
    main()