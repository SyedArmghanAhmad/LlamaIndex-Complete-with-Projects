
import random
import time
from thingspeak import Channel
from dotenv import load_dotenv
import os
import logging
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import requests
import logging
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()

CHANNEL_ID = os.getenv("CHANNEL_ID_IOT")
WRITE_API_KEY =os.getenv("WRITE_KEY_IOT")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iot_simulator")

def simulate_machine_data():
    channel = Channel(CHANNEL_ID, WRITE_API_KEY)
    while True:
        try:
            # Machine X (Conveyor Belt)
            channel.update({
                'field1': random.uniform(4.5, 6.5),  # Vibration (mm/s)
                'field2': random.uniform(65, 85),    # Temperature (°C)
                'field3': random.randint(180, 220),  # Operational Hours
                'field4': 'X'                        # Machine ID
            })
            logger.info("Data sent for Machine X")
            
            # Machine Y (Assembly Robot)
            channel.update({
                'field1': random.uniform(2.0, 4.0),
                'field2': random.uniform(55, 75),
                'field3': random.randint(150, 190),
                'field4': 'Y'
            })
            logger.info("Data sent for Machine Y")
            
            # Machine Z (Packaging Unit)
            channel.update({
                'field1': random.uniform(1.5, 3.5),
                'field2': random.uniform(60, 80),
                'field3': random.randint(200, 240),
                'field4': 'Z'
            })
            logger.info("Data sent for Machine Z")
            
        except Exception as e:
            logger.error(f"Failed to send data: {str(e)}")
        
        time.sleep(300)  # Update every 5 minutes

# Run in background thread
import threading
data_thread = threading.Thread(target=simulate_machine_data)
data_thread.daemon = True
data_thread.start()

# Configure ThingSpeak
THINGSPEAK_CHANNEL_ID = os.getenv("CHANNEL_ID_IOT")
THINGSPEAK_READ_KEY = os.getenv("READ_KEY_IOT")

# Enhanced data fetcher with error handling
# Enhanced data fetcher with retry mechanism
def get_machine_data(machine_id: str, retries: int = 3, delay: int = 5) -> dict:
    """Fetch machine data with retry mechanism and error handling"""
    for attempt in range(retries):
        try:
            url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
            params = {
                "api_key": THINGSPEAK_READ_KEY,
                "results": 1,
                "field4": machine_id  # Using field4 for machine ID tracking
            }
            
            response = requests.get(url, params=params, timeout=30)  # Increased timeout
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
                time.sleep(delay)  # Wait before retrying
            else:
                logging.error(f"All attempts failed: {str(e)}")
                return {"error": "Failed to fetch data after retries"}
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return {"error": "Unexpected error occurred"}

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

# Initialize components
Settings.llm = Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create query engine with fresh data
def get_query_engine():
    return VectorStoreIndex.from_documents(create_machine_docs()).as_query_engine()

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
    max_iterations=5  # Increased to allow more reasoning steps
)
