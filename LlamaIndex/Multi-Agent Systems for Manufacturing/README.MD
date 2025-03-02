
# Multi-Agent Systems for Manufacturing 🏭

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/multi-agent-manufacturing)
![License](https://img.shields.io/badge/license-MIT-blue)

A **multi-agent system** built with **LlamaIndex** to optimize manufacturing processes. This project demonstrates my ability to handle **real-time IoT data**, **dynamic document creation**, and **multi-agent collaboration** for quality control, maintenance scheduling, and operational monitoring.

---

## Features ✨

- **Real-Time IoT Integration**: Fetches live sensor data from **ThingSpeak** (vibration, temperature, operational hours).
- **Dynamic Document Creation**: Generates documents from real-time data for context-aware querying.
- **Multi-Agent Collaboration**: Uses **ReAct agents** to handle quality control, maintenance, and operational monitoring.
- **Actionable Insights**: Provides maintenance recommendations based on sensor data.
- **Error Handling & Retry Mechanisms**: Ensures robust data fetching and processing.

---

## Technologies Used 🛠️

- **LlamaIndex**: For document indexing, retrieval, and multi-agent systems.
- **Groq**: For LLM-powered answer generation.
- **Hugging Face Embeddings**: For vector embeddings (`BAAI/bge-small-en-v1.5`).
- **ThingSpeak IoT API**: For real-time sensor data.
- **Python**: Core programming language.

---

## Project Evolution 🚀

### **Phase 1: Small Sample Data**

- **Goal**: Demonstrate the **multi-agent system** with static data.
- **Implementation**:
  - Uses **quality control** and **maintenance** data as sample documents.
  - Implements **ReAct agents** for query routing and decision-making.
  - Adds **metadata filtering** for high-severity defects.
- **Files**:
  - `Multi-Agent Systems for Manufacturing.ipynb`

### **Phase 2: Real-Time IoT Integration**

- **Goal**: Build a **production-grade system** with real-time data.
- **Implementation**:
  - Fetches live sensor data from **ThingSpeak** (vibration, temperature, operational hours).
  - Creates **dynamic documents** from real-time data.
  - Provides **actionable insights** (e.g., maintenance recommendations).
  - Adds **error handling** and **retry mechanisms** for robust data fetching.
- **Files**:
  - `BetterVer.ipynb`

---

## Installation 💻

### Prerequisites

- Python 3.8+
- Groq API key (set in `.env` file)
- ThingSpeak API keys (set in `.env` file)

### Steps

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys to a `.env` file:

   ```ini
   GROQ_API_KEY=your_api_key_here
   CHANNEL_ID_IOT=your_channel_id_here
   READ_KEY_IOT=your_read_key_here
   WRITE_KEY_IOT=your_write_key_here
   ```

3. Run the project:

   ```bash
   BetterVer.ipynb
   ```

---

## Usage 🚀

### Running the Project

1. Start the interactive query system:

   ```bash
   python main.py
   ```

2. Enter your manufacturing questions. For example:

   ```bash
   Enter your question: Machine X is showing increased vibration. Should we schedule maintenance?
   ```

3. The system will:
   - Fetch real-time sensor data from **ThingSpeak**.
   - Generate **dynamic documents** from the data.
   - Provide **actionable insights** using **ReAct agents**.

---

## How It Works 🧠

### 1. Real-Time IoT Integration

- Fetches live sensor data (vibration, temperature, operational hours) from **ThingSpeak**.
- Uses **retry mechanisms** and **error handling** for robust data fetching.

### 2. Dynamic Document Creation

- Generates documents from real-time sensor data.
- Adds **actionable insights** (e.g., maintenance recommendations).

### 3. Multi-Agent Collaboration

- Uses **ReAct agents** to handle:
  - **Quality Control**: Identifies defects and recommends actions.
  - **Maintenance Scheduling**: Suggests maintenance based on operational hours.
  - **Operational Monitoring**: Monitors vibration and temperature levels.

### 4. Query Processing

- Combines **vector search** and **BM25** for accurate document retrieval.
- Uses **metadata filtering** to prioritize high-severity issues.

### 5. Actionable Insights

- Provides **maintenance recommendations** based on sensor data.
- Highlights **critical issues** (e.g., high vibration, overheating).

---

## Key Concepts Implemented 🔍

### 1. **Multi-Agent Systems**

- Uses **ReAct agents** for collaborative decision-making.
- Demonstrates my ability to design **scalable AI systems**.

### 2. **Real-Time IoT Integration**

- Fetches and processes live sensor data from **ThingSpeak**.
- Highlights my skills in **IoT integration** and **real-time data processing**.

### 3. **Dynamic Document Creation**

- Generates documents from real-time data for context-aware querying.
- Showcases my ability to handle **dynamic data** and **actionable insights**.

### 4. **Error Handling & Retry Mechanisms**

- Ensures robust data fetching and processing.
- Demonstrates my focus on **production-grade reliability**.

### 5. **Metadata Filtering**

- Prioritizes high-severity issues using metadata filters.
- Highlights my ability to implement **advanced retrieval techniques**.

---

## Example Queries 💡

1. **Quality Control**:

   ```bash
   Machine X has misaligned bearings. What should I do?
   ```

2. **Maintenance Scheduling**:

   ```bash
   Machine Y has been running for 210 hours. Is maintenance due?
   ```

3. **Operational Monitoring**:

   ```bash
   What is the current temperature of Machine Z?
   ```

4. **Actionable Insights**:

   ```bash
   Machine X is showing increased vibration. Should we schedule maintenance?
   ```

5. **Comparative Analysis**:

   ```bash
   Compare the operational hours of all machines.
   ```

---

## Configuration ⚙️

- **Chunk Size**: Adjust the `chunk_size` parameter for optimal performance.
- **LLM Model**: Switch to a different Groq model by updating `initialize_groq()`.
- **Embedding Model**: Replace `BAAI/bge-small-en-v1.5` with another Hugging Face model if needed.

---

## Contributing 🤝

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Submit a pull request.

---

## License 📄

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments 🙏

- **LlamaIndex**: For the powerful document indexing and retrieval framework.
- **Groq**: For the high-performance LLM API.
- **ThingSpeak**: For the IoT data platform.

---
