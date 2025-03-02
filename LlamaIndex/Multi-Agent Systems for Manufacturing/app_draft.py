import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Manufacturing Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import time
import datetime
import threading
import random
import plotly.graph_objects as go
import plotly.express as px
from streamlit_chat import message
import os
import requests
from dotenv import load_dotenv
import logging
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manufacturing_assistant")

# ThingSpeak configuration
CHANNEL_ID = os.getenv("CHANNEL_ID_IOT")
READ_API_KEY = os.getenv("READ_KEY_IOT")

# Initialize LLM components if not already done
@st.cache_resource
def initialize_llm_components():
    Settings.llm = Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return True

initialize_llm_components()

# Data fetcher with retry mechanism
def get_machine_data(machine_id: str, retries: int = 3, delay: int = 5) -> dict:
    """Fetch machine data with retry mechanism and error handling"""
    for attempt in range(retries):
        try:
            url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
            params = {
                "api_key": READ_API_KEY,
                "results": 1,
                "field4": machine_id  # Using field4 for machine ID tracking
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('feeds'):
                return {"error": "No data available"}
                
            latest = data['feeds'][0]
            return {
                "machine_id": latest.get('field4'),
                "vibration": float(latest.get('field1', 0)),
                "temperature": float(latest.get('field2', 0)),
                "hours": int(latest.get('field3', 0)),
                "timestamp": latest.get('created_at')
            }
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"All attempts failed: {str(e)}")
                return {"error": "Failed to fetch data after retries"}
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {"error": "Unexpected error occurred"}

# Get historical data for visualization
def get_historical_machine_data(machine_id: str, results: int = 10):
    """Fetch historical data for visualization"""
    try:
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        params = {
            "api_key": READ_API_KEY,
            "results": results,
            "field4": machine_id
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data.get('feeds'):
            return {"error": "No historical data available"}
            
        return data['feeds']
        
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        return {"error": str(e)}

# Document creation with actionable insights
def create_machine_docs():
    docs = []
    for machine_id in ["X", "Y", "Z"]:
        data = get_machine_data(machine_id)
        
        if data.get('error'):
            text = f"Machine {machine_id} - No recent data available"
        else:
            # Add actionable insights to the document
            vibration_status = "Normal" if data['vibration'] < 5.0 else "High"
            temp_status = "Normal" if data['temperature'] < 80 else "High"
            maintenance_due = "Yes" if data['hours'] >= 200 else "No"
            
            text = f"""Machine {data['machine_id']} Status:
            - Vibration: {data['vibration']} mm/s ({vibration_status})
            - Temperature: {data['temperature']}°C ({temp_status})
            - Operating Hours: {data['hours']}h
            - Maintenance Due: {maintenance_due}
            - Last Update: {data['timestamp']}"""
            
        docs.append(Document(
            text=text,
            metadata={
                "machine_id": machine_id,
                "type": "sensor_readings",
                "status": "active" if not data.get('error') else "inactive"
            }
        ))
    return docs

# Create query engine with fresh data
def get_query_engine():
    return VectorStoreIndex.from_documents(create_machine_docs()).as_query_engine()

# Create agent with fresh data
@st.cache_resource(ttl=300)  # Cache for 5 minutes
def create_agent():
    # Tool with automatic refresh
    machine_tool = QueryEngineTool(
        query_engine=get_query_engine(),
        metadata=ToolMetadata(
            name="machine_monitor",
            description="Real-time operational data with maintenance recommendations",
        ),
    )
    
    # Knowledge-enhanced agent
    agent = ReActAgent.from_tools(
        tools=[machine_tool],
        verbose=True,
        system_prompt="""You are a manufacturing engineer assistant. Use the following rules:
        - Normal vibration <5.0 mm/s
        - Max temperature 80°C
        - Lubrication needed every 200h
        Provide clear recommendations based on sensor data.""",
        max_iterations=5
    )
    return agent

# Custom CSS for cards
st.markdown("""
<style>
    .machine-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .card-header {
        padding: 10px;
        border-radius: 10px 10px 0 0;
        margin-bottom: 10px;
    }
    .normal-header {
        background-color: rgba(144, 238, 144, 0.2);
    }
    .warning-header {
        background-color: rgba(255, 99, 71, 0.2);
    }
    .sensor-value {
        font-size: 18px;
        font-weight: bold;
    }
    .gauge-container {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏭 Manufacturing Dashboard")
    st.divider()
    st.subheader("Machine Information")
    st.markdown("""
    - **Machine X**: Conveyor Belt
    - **Machine Y**: Assembly Robot
    - **Machine Z**: Packaging Unit
    """)
    st.divider()
    st.subheader("Operational Thresholds")
    st.markdown("""
    - Normal vibration: < 5.0 mm/s
    - Max temperature: 80°C
    - Maintenance due: Every 200 hours
    """)
    st.divider()
    
    refresh_button = st.button("🔄 Refresh Data", use_container_width=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Manufacturing Assistant. Ask me anything about your machines."}
    ]

# Tabs for Dashboard and Chat
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat Assistant"])

with tab1:
    st.header("Manufacturing IoT Dashboard")
    
    # Function to create a machine card
    def create_machine_card(machine_id, machine_name, col):
        data = get_machine_data(machine_id)
        
        if data.get('error'):
            col.error(f"⚠️ {machine_name} (Machine {machine_id}): Offline")
            return
        
        # Determine status based on readings
        has_issues = (
            data['vibration'] >= 5.0 or 
            data['temperature'] >= 80 or 
            data['hours'] >= 200
        )
        
        status = "⚠️ Needs Attention" if has_issues else "✅ Normal"
        header_class = "warning-header" if has_issues else "normal-header"
        
        # Vibration gauge
        vibration_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=data['vibration'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Vibration (mm/s)"},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgreen"},
                    {'range': [5, 10], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ))
        vibration_fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
        
        # Temperature gauge
        temp_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=data['temperature'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Temperature (°C)"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        temp_fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
        
        # Create card content
        with col:
            # Use Streamlit expander as a card
            with st.expander(f"{machine_name} (Machine {machine_id}): {status}", expanded=True):
                st.markdown(f"""
                <div class="machine-card">
                  <div class="{header_class} card-header">
                    <h3>Machine {machine_id} Status</h3>
                  </div>
                  <div class="card-body">
                    <p><b>Operating Hours:</b> <span class="sensor-value">{data['hours']}h</span> {'⚠️' if data['hours'] >= 200 else ''}</p>
                    <p><b>Last Update:</b> {data['timestamp']}</p>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauges
                st.plotly_chart(vibration_fig, use_container_width=True, key=f"vibration_{machine_id}")
                st.plotly_chart(temp_fig, use_container_width=True, key=f"temp_{machine_id}")

    # Create row of cards for all machines
    cols = st.columns(3)
    
    machine_info = [
        ("X", "Conveyor Belt"),
        ("Y", "Assembly Robot"),
        ("Z", "Packaging Unit")
    ]
    
    for i, (machine_id, machine_name) in enumerate(machine_info):
        col = cols[i]
        create_machine_card(machine_id, machine_name, col)
    
    # Historical data section
    st.subheader("Historical Data Trends")
    
    selected_machine = st.selectbox(
        "Select Machine",
        options=["X", "Y", "Z"],
        format_func=lambda x: f"Machine {x} - {dict(machine_info)[x]}"
    )
    
    metric_tab1, metric_tab2, metric_tab3 = st.tabs(["Vibration", "Temperature", "Operating Hours"])
    
    def plot_historical_data(machine_id, metric_field, metric_name, y_min, y_max, threshold=None):
        historical_data = get_historical_machine_data(machine_id, results=50)
        
        if isinstance(historical_data, dict) and historical_data.get('error'):
            st.error(f"Error: {historical_data.get('error')}")
            return
            
        # Convert to dataframe
        df = pd.DataFrame(historical_data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Convert to correct data type
        df[metric_field] = pd.to_numeric(df[metric_field])
        
        # Create figure
        fig = px.line(
            df, 
            x='created_at', 
            y=metric_field,
            title=f"{metric_name} History - Machine {machine_id}",
            labels={'created_at': 'Time', metric_field: metric_name}
        )
        
        # Add threshold line if specified
        if threshold is not None:
            fig.add_shape(
                type="line",
                x0=df['created_at'].min(),
                y0=threshold,
                x1=df['created_at'].max(),
                y1=threshold,
                line=dict(color="Red", width=2, dash="dash"),
                name="Threshold"
            )
            
            # Add annotation for threshold
            fig.add_annotation(
                x=df['created_at'].max(),
                y=threshold,
                text=f"Threshold: {threshold}",
                showarrow=False,
                yshift=10
            )
        
        # Set y-axis range
        fig.update_yaxes(range=[y_min, y_max])
        
        # Format
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=metric_name,
            hovermode="x unified"
        )
        
        return fig
    
    with metric_tab1:
        vibration_fig = plot_historical_data(
            selected_machine, 
            'field1', 
            'Vibration (mm/s)', 
            0, 10, 
            threshold=5.0
        )
        if vibration_fig:
            st.plotly_chart(vibration_fig, use_container_width=True, key=f"hist_vib_{selected_machine}")
    
    with metric_tab2:
        temp_fig = plot_historical_data(
            selected_machine, 
            'field2', 
            'Temperature (°C)', 
            0, 100, 
            threshold=80
        )
        if temp_fig:
            st.plotly_chart(temp_fig, use_container_width=True, key=f"hist_temp_{selected_machine}")
    
    with metric_tab3:
        hours_fig = plot_historical_data(
            selected_machine, 
            'field3', 
            'Operating Hours', 
            0, 300, 
            threshold=200
        )
        if hours_fig:
            st.plotly_chart(hours_fig, use_container_width=True, key=f"hist_hours_{selected_machine}")
    
    # Last updated time
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Chat interface
with tab2:
    st.header("Manufacturing Assistant")
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # User input
    if prompt := st.chat_input("Ask something about your machines..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant thinking message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Create/get agent
                agent = create_agent()
                
                # Get response from agent
                response = agent.chat(prompt)
                
                # Update thinking message
                message_placeholder.markdown(response.response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Handle refresh button
if refresh_button:
    st.cache_resource.clear()
    st.experimental_rerun()