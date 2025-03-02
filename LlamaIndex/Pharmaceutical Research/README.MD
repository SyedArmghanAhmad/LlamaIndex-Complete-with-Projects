# Clinical Trial Intelligence System

## Overview

The Clinical Trial Intelligence System is a sophisticated application designed to help healthcare professionals, researchers, and stakeholders explore and analyze clinical trial data using natural language queries. The system leverages advanced AI models, including Groq's Mixtral-8x7b-32768 and Hugging Face's BAAI/bge-small-en-v1.5 embeddings, to provide insightful and context-aware responses to user queries. The application is built using LlamaIndex for document indexing and querying, and Streamlit for the user interface.

## Key Concepts and Functionalities

### 1. **Document Indexing and Querying**

- **LlamaIndex**: The core of the system uses LlamaIndex to index and query clinical trial documents. This allows for efficient retrieval of relevant information based on user queries.
- **VectorStoreIndex**: Clinical trial data is indexed using VectorStoreIndex, which enables semantic search capabilities.
- **Metadata Filtering**: Users can filter results based on metadata such as trial phase, status, and condition, ensuring more precise and relevant responses.

### 2. **Natural Language Processing (NLP)**

- **Groq LLM**: The system utilizes Groq's Mixtral-8x7b-32768 model for generating natural language responses. This model is capable of understanding and generating human-like text, making it ideal for answering complex queries.
- **Hugging Face Embeddings**: The BAAI/bge-small-en-v1.5 embedding model is used to convert text into vector representations, enabling semantic search and similarity comparisons.

### 3. **Data Management**

- **Document Conversion**: Clinical trial data is converted into LlamaIndex documents with rich metadata, allowing for detailed and context-aware queries.
- **Versioning and Change Tracking**: The system supports versioning of indexed data, enabling users to track changes over time and roll back to previous versions if necessary.
- **Incremental Indexing**: New data can be added to the existing index without needing to rebuild it from scratch, ensuring that the system remains up-to-date with the latest information.

### 4. **User Interface**

- **Streamlit**: The user interface is built using Streamlit, providing an interactive and user-friendly experience. The UI includes features such as:
  - **Natural Language Query Input**: Users can input queries in natural language.
  - **Filter Options**: Users can filter results by trial phase, status, and other metadata.
  - **Response Formatting**: Responses are formatted for easy reading, with summaries and detailed metadata presented in a structured manner.

### 5. **Enhanced Response Formatting**

- **Contextual Summaries**: The system generates contextual summaries of query results, providing users with a high-level overview of the most relevant trials.
- **Technical Term Explanations**: Technical terms related to clinical trials (e.g., trial phases, statuses) are explained in simple language, making the information accessible to non-experts.
- **Bullet Points and Structured Data**: Responses are presented in bullet points and structured formats, making it easy to scan and understand the information.

### 6. **Interactive Query System**

- **Interactive Querying**: Users can interactively query the system, refining their searches with filters and receiving instant feedback.
- **Error Handling**: The system includes robust error handling to ensure that users are informed of any issues and can retry their queries if necessary.

### 7. **Custom Styling**

- **UI Customization**: The Streamlit UI is customized with a clean and professional design, including custom styles for headers, input fields, and buttons, enhancing the user experience.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Groq API key
- Hugging Face account (for embedding model)

### Installation

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your Groq API key:

     ```plaintext
     GROQ_API_KEY=your_groq_api_key_here
     ```

### Running the Application

1. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Usage

1. **Enter Your Query**: Type your question about clinical trials in the input box.
2. **Apply Filters**: Optionally, apply filters such as trial phase or status to refine your search.
3. **View Results**: The system will display a summary of the most relevant trials, along with detailed metadata.

## Example Queries

- "What Alzheimer's trials are in Phase 2?"
- "Show me all recruiting trials for Parkinson's disease."
- "What are the latest trials for Multiple Sclerosis?"

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- **Groq** for providing the powerful Mixtral-8x7b-32768 model.
- **Hugging Face** for the BAAI/bge-small-en-v1.5 embedding model.
- **LlamaIndex** for the robust document indexing and querying framework.
- **Streamlit** for the intuitive and interactive UI framework.

---
