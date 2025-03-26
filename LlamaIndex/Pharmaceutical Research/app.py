import streamlit as st
from llama_index.core import VectorStoreIndex, Document, Settings,PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize models
groq_api_key = os.getenv("GROQ_API_KEY")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
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
        .trial-card {border-left: 4px solid #2e86c1; padding: 1rem; margin: 1rem 0; background: #f8f9fa;}
        .warning {color: #d35400; background-color: #fdebd0; padding: 10px; border-radius: 5px;}
        .success {color: #28b463; background-color: #d5f5e3; padding: 10px; border-radius: 5px;}
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
    ],
    "Eligibility": [
        "Ages 50-85, mild to moderate Alzheimer's, MMSE 18-26",
        "Ages 55-80, early Alzheimer's, CDR 0.5-1",
        "Ages 40-75, Parkinson's diagnosed <5 years, H&Y 1-3",
        "Ages 18-60, RRMS diagnosis, EDSS 2.0-6.5",
        "Ages 50-90, moderate Alzheimer's, MMSE 10-20",
        "Ages 60+, Alzheimer's diagnosis, previous trial participation",
        "Ages 30-70, Parkinson's diagnosed <10 years, no dementia",
        "Ages 18-65, progressive MS, EDSS 3.0-6.5",
        "Ages 60-80, high risk for Alzheimer's, no current diagnosis",
        "Ages 40+, Parkinson's diagnosis, smartphone access"
    ]
}

trials_df = pd.DataFrame(expanded_data)

### Document Conversion
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
            "Description": row["Description"],
            "Eligibility": row["Eligibility"]
        }
        text = f"""
        Title: {row['Title']}
        Condition: {row['Condition']}
        Phase: {row['Phase']}
        Status: {row['Status']}
        Interventions: {', '.join(row['Interventions'])}
        Description: {row['Description']}
        Eligibility: {row['Eligibility']}
        """
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

def _is_recent_trial(start_date, months=6):
    """Check if trial started within last N months"""
    trial_date = pd.to_datetime(start_date)
    cutoff_date = datetime.now() - timedelta(days=30*months)
    return trial_date >= cutoff_date

### Enhanced Query Engine
def get_query_engine(index, filters=None):
    """Create configured query engine with fixed response handling"""
    text_qa_template = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, answer the question: {query_str}\n"
        "If the context doesn't contain the answer, say 'No matching trials found.'"
    )
    
    return index.as_query_engine(
        similarity_top_k=5,
        filters=filters,
        verbose=False,
        response_mode="compact",
        text_qa_template=text_qa_template
    )


### Improved Response Formatting
def format_response(response, question):
    """Generate enhanced response with better context handling"""
    # Handle case where response is None or empty
    if response is None or not hasattr(response, 'source_nodes'):
        return fallback_response(question)
    
    # Handle case where no results found
    if not response.source_nodes or str(response).strip() == "No matching trials found.":
        return fallback_response(question)
    
    # Process successful responses
    try:
        # Generate natural language summary
        context = "\n".join([node.text for node in response.source_nodes])
        
        # Special handling for different query types
        if "recruiting" in question.lower():
            prompt = create_recruiting_prompt(context, question)
        elif "recent" in question.lower():
            prompt = create_recent_trials_prompt(context, question)
        elif "suggest" in question.lower():
            prompt = create_suggestion_prompt(context, question)
        else:
            prompt = create_general_prompt(context, question)
            
        response_text = Settings.llm.complete(prompt).text
        
        # Build detailed metadata section
        details = []
        for node in response.source_nodes[:3]:  # Show top 3 most relevant
            meta = node.metadata
            details.append(f"""
            <div class='trial-card'>
            <b>Trial ID:</b> {meta['NCTId']}<br>
            • <b>Title:</b> {meta['Title']}<br>
            • <b>Condition:</b> {meta['Condition']}<br>
            • <b>Phase:</b> {meta['Phase']} ({_explain_phase(meta['Phase'])})<br>
            • <b>Status:</b> {meta['Status']} ({_explain_status(meta['Status'])})<br>
            • <b>Interventions:</b> {meta['Interventions']}<br>
            • <b>Eligibility:</b> {meta['Eligibility']}<br>
            • <b>Timeline:</b> {meta['StartDate']} to {meta['CompletionDate']}<br>
            </div>
            """)
        
        return f"""
        {response_text.strip()}
        
        <h4>Relevant Clinical Trials:</h4>
        {"".join(details)}
        """
    except Exception as e:
        return fallback_response(question, error=str(e))

def create_recruiting_prompt(context, question):
    return f"""
    Context: {context}
    Question: {question}
    Task: Create a concise summary of recruiting trials for healthcare professionals.
    Requirements:
    1. List only currently recruiting trials
    2. Highlight key features of each trial
    3. Mention patient eligibility highlights
    4. Include practical next steps
    5. Keep under 200 words
    """

def create_recent_trials_prompt(context, question):
    return f"""
    Context: {context}
    Question: {question}
    Task: Summarize recently started clinical trials.
    Requirements:
    1. Only include trials started in last 6 months
    2. Highlight novel aspects of each trial
    3. Explain clinical significance
    4. Keep under 200 words
    """

def create_suggestion_prompt(context, question):
    return f"""
    Context: {context}
    Question: {question}
    Task: Suggest appropriate clinical trials based on patient characteristics.
    Requirements:
    1. Match trials to patient profile mentioned in question
    2. Rank by suitability (phase, status, interventions)
    3. Include risks/benefits analysis
    4. Provide clear next steps
    5. Keep under 250 words
    """

def create_general_prompt(context, question):
    return f"""
    Context: {context}
    Question: {question}
    Task: Summarize clinical trial information for healthcare professionals.
    Requirements:
    1. Directly answer the question first
    2. Highlight key trial aspects
    3. Explain technical terms in parentheses
    4. Use bullet points for clarity
    5. Keep under 250 words
    """

def fallback_response(question, error=None):
    """Generate a helpful response when no trials are found"""
    prompt = f"""
    Question: {question}
    Context: No matching clinical trials found in the database.
    Error: {error or 'No results found'}
    Task: Provide a helpful response that:
    1. Acknowledges the lack of specific results
    2. Provides general guidance
    3. Maintains professional tone
    4. Suggests alternative resources
    """
    general_response = Settings.llm.complete(prompt).text
    return f"""
    <div class='warning'>
    ⚠️ No matching trials found in our database. Here's some general information:
    </div>
    {general_response}
    <div class='success'>
    Tip: Try broadening your search criteria or check ClinicalTrials.gov for more options.
    </div>
    """
    
def query_trials(index, question, phase=None, status=None):
    """Enhanced query function with better error handling"""
    filters = []
    
    # Automatic status filtering based on question
    if "recruiting" in question.lower():
        filters.append(MetadataFilter(key="Status", value="Recruiting"))
    elif "completed" in question.lower():
        filters.append(MetadataFilter(key="Status", value="Completed"))
    elif "active" in question.lower():
        filters.append(MetadataFilter(key="Status", value="Active"))
    
    # Manual filters from UI
    if phase:
        filters.append(MetadataFilter(key="Phase", value=phase))
    if status:
        filters.append(MetadataFilter(key="Status", value=status))
    
    # Handle condition-specific queries
    if "alzheimer" in question.lower():
        filters.append(MetadataFilter(key="Condition", value="Alzheimer's Disease"))
    elif "parkinson" in question.lower():
        filters.append(MetadataFilter(key="Condition", value="Parkinson's Disease"))
    elif "multiple sclerosis" in question.lower():
        filters.append(MetadataFilter(key="Condition", value="Multiple Sclerosis"))
    
    # Handle recent trials filter
    if "recent" in question.lower():
        recent_trials = trials_df[trials_df.apply(
            lambda x: _is_recent_trial(x['StartDate']), axis=1)]
        if not recent_trials.empty:
            filters.append(MetadataFilter(
                key="NCTId", 
                value=recent_trials["NCTId"].tolist(),
                operator="in"
            ))
    
    metadata_filters = MetadataFilters(filters=filters) if filters else None
    
    try:
        query_engine = get_query_engine(index, metadata_filters)
        response = query_engine.query(question)
        
        # Manual verification for phase-specific queries
        if "phase" in question.lower():
            target_phase = None
            if "phase 1" in question.lower():
                target_phase = "Phase 1"
            elif "phase 2" in question.lower():
                target_phase = "Phase 2"
            elif "phase 3" in question.lower():
                target_phase = "Phase 3"
            elif "phase 4" in question.lower():
                target_phase = "Phase 4"
            
            if target_phase:
                response.source_nodes = [
                    node for node in response.source_nodes 
                    if node.metadata.get("Phase") == target_phase
                ]
        
        return response
    except Exception as e:
        st.error(f"Query processing error: {str(e)}")
        return None

# ====================== [Streamlit UI] ======================def format_response(response, question):
def format_response(response, question):
    """Generate enhanced response with both structured data and LLM explanation"""
    if response is None or not hasattr(response, 'source_nodes') or not response.source_nodes:
        return fallback_response(question)
    
    # Filter for specific queries
    if "phase 3" in question.lower():
        response.source_nodes = [
            node for node in response.source_nodes 
            if node.metadata.get("Phase") == "Phase 3"
        ]
        if not response.source_nodes:
            return fallback_response(question, phase_specific=True)

    st.subheader("📄 Results Summary")
    
    # Generate LLM explanation first
    context = "\n".join([node.text for node in response.source_nodes[:3]])
    explanation_prompt = f"""
    You are a medical research assistant analyzing clinical trials. Provide:
    1. A 2-3 sentence summary of these trials' significance
    2. Key clinical implications
    3. Important considerations for healthcare providers
    4. Write in professional but accessible language
    
    Question: {question}
    Trial Data: {context}
    """
    
    with st.spinner("Generating expert analysis..."):
        explanation = Settings.llm.complete(explanation_prompt).text
    
    # Display LLM explanation in a styled container
    with st.container():
        st.markdown("""
        <div style='
            background-color: #transparent;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #2e86c1;
            margin-bottom: 2rem;
            color: white;
        '>
            <h4 style='color: #2e86c1; margin-top: 0;'>🧑‍⚕️ Clinical Insight</h4>
            <p style='margin-bottom: 0;'>{}</p>
        </div>
        """.format(explanation), unsafe_allow_html=True)
    
    # Display each trial in expandable cards
    for i, node in enumerate(response.source_nodes[:3]):  # Show top 3
        meta = node.metadata
        with st.expander(f"🔬 Trial #{i+1}: {meta['Title']}", expanded=(i==0)):
            # Create two columns for better layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                **🆔 Trial ID:** `{meta['NCTId']}`  
                **🏥 Condition:** {meta['Condition']}  
                **🔬 Phase:** {meta['Phase']} ({_explain_phase(meta['Phase'])})  
                **🔄 Status:** {meta['Status']} ({_explain_status(meta['Status'])})  
                **📅 Timeline:** {meta['StartDate']} to {meta['CompletionDate']}
                """)
                
            with col2:
                st.markdown(f"""
                **💊 Interventions:**  
                {meta['Interventions']}  
                
                **🎯 Eligibility Criteria:**  
                {meta['Eligibility']}  
                
                **📝 Study Description:**  
                {meta['Description']}
                """)
            
            # Add visual separator between trials
            if i < len(response.source_nodes[:3]) - 1:
                st.markdown("---")
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
    question = st.text_input("Ask about clinical trials:", 
                           placeholder="e.g., 'What Alzheimer's trials are in Phase 2?'")
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2 = st.columns(2)
    with col1:
        phase = st.selectbox("Phase", [None, "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 1/2"])
    with col2:
        status = st.selectbox("Status", [None, "Recruiting", "Completed", "Active", "Terminated", "Not yet recruiting"])

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
                    format_response(response, question)
                    
                    # Add disclaimer
                    st.markdown("---")
                    st.caption(f"""
                    Note: Always verify trial details through official sources before making referrals. 
                    Eligibility criteria may change. Last updated: {datetime.now().strftime('%Y-%m-%d')}
                    """)
                    
                except Exception as e:
                    st.error(f"🚨 Error processing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question to search")

if __name__ == "__main__":
    main()