import streamlit as st
from streamlit_autorefresh import st_autorefresh  # Ensure to install streamlit-autorefresh

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
import logging
import requests
import os
import random
from dotenv import load_dotenv

########################################
# If your LlamaIndex is more up to date
########################################
from llama_index.core.schema import Document  # <-- Usually the new location
# If that fails, try:
# from llama_index import Document

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import plotly.graph_objects as go
import plotly.express as px

# Auto-refresh every 2 minutes (120000 milliseconds)
st_autorefresh(interval=120000, key="datarefresh")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manufacturing_assistant")

# ThingSpeak configuration
CHANNEL_ID = os.getenv("CHANNEL_ID_IOT")
READ_API_KEY = os.getenv("READ_KEY_IOT")
WRITE_API_KEY = os.getenv("WRITE_KEY_IOT")  # For simulation

# Initialize LLM components if not already done
@st.cache_resource
def initialize_llm_components():
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return True

initialize_llm_components()

##############################
# No caching on data fetches #
##############################
def simulate_sensor_data(machine_id: str, num_samples: int = 5):
    """Simulate sending sensor data to ThingSpeak (requires WRITE_API_KEY)."""
    if not WRITE_API_KEY:
        logging.warning("WRITE_KEY_IOT not provided; simulation skipped")
        return
    update_url = "https://api.thingspeak.com/update.json"
    for _ in range(num_samples):
        vibration = round(random.uniform(0, 10), 2)
        temperature = round(random.uniform(20, 100), 2)
        hours = random.randint(0, 300)
        params = {
            "api_key": WRITE_API_KEY,
            "field1": vibration,
            "field2": temperature,
            "field3": hours,
            "field4": machine_id,
        }
        try:
            response = requests.post(update_url, params=params, timeout=30)
            if response.ok:
                logging.info(f"Simulated data for Machine {machine_id}: vib={vibration}, temp={temperature}, hours={hours}")
            else:
                logging.error(f"Failed to simulate data for Machine {machine_id}: {response.text}")
        except Exception as e:
            logging.error(f"Error simulating data for Machine {machine_id}: {str(e)}")
        time.sleep(0.2)

def get_machine_data(machine_id: str, retries: int = 3, delay: int = 5) -> dict:
    """Fetch machine data with retry mechanism and local filtering by machine_id."""
    for attempt in range(retries):
        try:
            url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
            params = {
                "api_key": READ_API_KEY,
                "results": 50
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            feeds = data.get('feeds', [])
            matching_feeds = [feed for feed in feeds if feed.get('field4') == machine_id]
            if not matching_feeds:
                return {"error": "No data available"}
            matching_feeds.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            latest = matching_feeds[0]
            logging.info(f"Fetched data for Machine {machine_id}: {latest}")
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

def get_historical_machine_data(machine_id: str, results: int = 10):
    """Fetch historical data for visualization by local filtering on machine_id."""
    try:
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
        params = {
            "api_key": READ_API_KEY,
            "results": results * 2,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        feeds = data.get('feeds', [])
        matching_feeds = [feed for feed in feeds if feed.get('field4') == machine_id]
        if not matching_feeds:
            return {"error": "No historical data available"}
        matching_feeds.sort(key=lambda x: x.get('created_at', ''))
        return matching_feeds[-results:]
    except Exception as e:
        logging.error(f"Error fetching historical data: {str(e)}")
        return {"error": str(e)}

def create_machine_docs():
    """Generate documents (no caching) so each run fetches fresh data."""
    docs = []
    for machine_id in ["X", "Y", "Z"]:
        data = get_machine_data(machine_id)
        if data.get('error'):
            text = f"Machine {machine_id} - No recent data available"
        else:
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

def get_query_engine():
    return VectorStoreIndex.from_documents(create_machine_docs()).as_query_engine()

########################################
# Increase max_iterations to avoid early "Reached max iterations" error
########################################
@st.cache_resource(ttl=120)
def create_agent():
    machine_tool = QueryEngineTool(
        query_engine=get_query_engine(),
        metadata=ToolMetadata(
            name="machine_monitor",
            description="Real-time operational data with maintenance recommendations",
        ),
    )
    agent = ReActAgent.from_tools(
        tools=[machine_tool],
        verbose=True,
        system_prompt="""You are a manufacturing engineer assistant. Use the following rules:
        - Normal vibration <5.0 mm/s
        - Max temperature 80°C
        - Lubrication needed every 200h
        Provide clear recommendations based on sensor data.""",
        max_iterations=10  # <-- Increased from 5 to 10
    )
    return agent

# Dark mode CSS
st.markdown("""
<style>
/* Dark overall background */
body {
  background-color: #121212 !important; 
  color: #EEEEEE !important;
}

/* Main Streamlit container */
.block-container {
  background-color: #1E1E1E !important; 
  color: #FFFFFF !important;
  padding: 1rem !important;
  border-radius: 0.5rem !important;
}

/* Machine card styling */
.machine-card {
  border: 1px solid #2C2C2C !important;
  border-radius: 10px !important;
  padding: 15px !important;
  margin-bottom: 20px !important;
  background-color: #2C2C2C !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.4) !important;
}

/* Header styling for the cards */
.card-header {
  padding: 10px !important;
  border-radius: 10px 10px 0 0 !important;
  margin-bottom: 10px !important;
  font-weight: bold !important;
  font-size: 20px !important;
  color: #FFFFFF !important;
  text-align: center !important;
}

/* Normal header color (green) */
.normal-header {
  background-color: #43A047 !important; 
}

/* Warning header color (red) */
.warning-header {
  background-color: #E53935 !important; 
}

/* Larger, bold sensor text */
.sensor-value {
  font-size: 18px !important;
  font-weight: bold !important;
  color: #FFFFFF !important;
}

/* Give Plotly charts a dark background to match */
.js-plotly-plot {
  background-color: #2C2C2C !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏭 Manufacturing Dashboard")
    st.markdown("### Real-Time Machine Monitoring & Analysis")
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
    - **Vibration:** Normal < 5.0 mm/s  
    - **Temperature:** Max 80°C  
    - **Maintenance:** Due every 200 hours  
    """)
    st.divider()
    simulate_button = st.button("💡 Simulate Sensor Data", key="simulate")
    if simulate_button:
        for machine_id in ["X", "Y", "Z"]:
            simulate_sensor_data(machine_id, num_samples=5)
        st.rerun()

    refresh_button = st.button("🔄 Refresh Data", use_container_width=True)

# Chat History
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Manufacturing Assistant. Ask me anything about your machines."}
    ]

tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat Assistant"])

with tab1:
    st.header("Manufacturing IoT Dashboard")
    
    def create_machine_card(machine_id, machine_name, col):
        data = get_machine_data(machine_id)  # Always fetch fresh
        if data.get('error'):
            col.error(f"⚠️ {machine_name} (Machine {machine_id}): Offline")
            return
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
                'bar': {'color': "#37474F"},
                'steps': [
                    {'range': [0, 5], 'color': "#81C784"},
                    {'range': [5, 10], 'color': "#E57373"}
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
                'bar': {'color': "#37474F"},
                'steps': [
                    {'range': [0, 80], 'color': "#81C784"},
                    {'range': [80, 100], 'color': "#E57373"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        temp_fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
        
        with col:
            with st.expander(f"{machine_name} (Machine {machine_id}): {status}", expanded=True):
                st.markdown(f"""
                <div class="machine-card">
                  <div class="{header_class} card-header">
                    Machine {machine_id} Status
                  </div>
                  <div class="card-body">
                    <p><b>Operating Hours:</b> <span class="sensor-value">{data['hours']}h</span> {'⚠️' if data['hours'] >= 200 else ''}</p>
                    <p><b>Last Update:</b> {data['timestamp']}</p>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(vibration_fig, use_container_width=True, key=f"vibration_{machine_id}")
                st.plotly_chart(temp_fig, use_container_width=True, key=f"temp_{machine_id}")

    # Machine cards
    cols = st.columns(3)
    machine_info = [
        ("X", "Conveyor Belt"),
        ("Y", "Assembly Robot"),
        ("Z", "Packaging Unit")
    ]
    for i, (machine_id, machine_name) in enumerate(machine_info):
        create_machine_card(machine_id, machine_name, cols[i])
    
    # Historical data
    st.subheader("Historical Data Trends")
    selected_machine = st.selectbox(
        "Select Machine",
        options=["X", "Y", "Z"],
        format_func=lambda x: f"Machine {x} - {dict(machine_info)[x]}"
    )
    
    metric_tab1, metric_tab2, metric_tab3 = st.tabs(["Vibration", "Temperature", "Operating Hours"])
    
    def plot_historical_data(machine_id, metric_field, metric_name, y_min=None, y_max=None, threshold=None):
        historical_data = get_historical_machine_data(machine_id, results=50)
        if isinstance(historical_data, dict) and historical_data.get('error'):
            st.error(f"Error: {historical_data.get('error')}")
            return
        df = pd.DataFrame(historical_data)
        if df.empty:
            st.info("No historical data to display.")
            return
        df['created_at'] = pd.to_datetime(df['created_at'])
        df[metric_field] = pd.to_numeric(df[metric_field])
        df.sort_values(by='created_at', inplace=True)
        
        fig = px.line(
            df, 
            x='created_at', 
            y=metric_field,
            title=f"{metric_name} History - Machine {machine_id}",
            labels={'created_at': 'Time', metric_field: metric_name},
            markers=True,         # show markers on points
            line_shape='spline'   # smooth out the line
        )
        
        if threshold is not None:
            fig.add_shape(
                type="line",
                x0=df['created_at'].min(),
                y0=threshold,
                x1=df['created_at'].max(),
                y1=threshold,
                line=dict(color="red", width=2, dash="dash"),
                name="Threshold"
            )
            fig.add_annotation(
                x=df['created_at'].max(),
                y=threshold,
                text=f"Threshold: {threshold}",
                showarrow=False,
                yshift=10
            )
        
        if y_min is not None and y_max is not None:
            fig.update_yaxes(range=[y_min, y_max])
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=metric_name,
            hovermode="x unified"
        )
        return fig
    
    with metric_tab1:
        vibration_fig = plot_historical_data(
            selected_machine, 'field1', 'Vibration (mm/s)',
            y_min=0, y_max=10, threshold=5.0
        )
        if vibration_fig:
            st.plotly_chart(vibration_fig, use_container_width=True, key=f"hist_vib_{selected_machine}")
    
    with metric_tab2:
        temp_fig = plot_historical_data(
            selected_machine, 'field2', 'Temperature (°C)',
            y_min=0, y_max=100, threshold=80
        )
        if temp_fig:
            st.plotly_chart(temp_fig, use_container_width=True, key=f"hist_temp_{selected_machine}")
    
    with metric_tab3:
        hours_fig = plot_historical_data(
            selected_machine, 'field3', 'Operating Hours',
            y_min=0, y_max=300, threshold=200
        )
        if hours_fig:
            st.plotly_chart(hours_fig, use_container_width=True, key=f"hist_hours_{selected_machine}")
    
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with tab2:
    st.header("Manufacturing Assistant")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input("Ask something about your machines..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                agent = create_agent()
                response = agent.chat(prompt)
                message_placeholder.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Pressing "Refresh Data" triggers a full re-run
if refresh_button:
    st.cache_resource.clear()
    st.rerun()
