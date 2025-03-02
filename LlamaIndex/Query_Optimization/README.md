# SEC Filing Analysis System

## Overview

The SEC Filing Analysis System is a sophisticated tool designed to help financial analysts, investors, and stakeholders analyze and interpret SEC filings using natural language queries. The system leverages advanced AI models, including Groq's Mixtral-8x7b-32768 and Hugging Face's BAAI/bge-small-en-v1.5 embeddings, to provide insightful and context-aware responses to user queries. The application is built using LlamaIndex for document indexing and querying, and includes custom post-processing for optimized query performance.

## Key Concepts and Functionalities

### 1. **Document Indexing and Querying**

- **LlamaIndex**: The core of the system uses LlamaIndex to index and query SEC filing documents. This allows for efficient retrieval of relevant information based on user queries.
- **VectorStoreIndex**: SEC filing data is indexed using VectorStoreIndex, which enables semantic search capabilities.
- **Metadata Filtering**: Users can filter results based on metadata such as form type, company, ticker, and filing date, ensuring more precise and relevant responses.

### 2. **Natural Language Processing (NLP)**

- **Groq LLM**: The system utilizes Groq's Mixtral-8x7b-32768 model for generating natural language responses. This model is capable of understanding and generating human-like text, making it ideal for answering complex financial queries.
- **Hugging Face Embeddings**: The BAAI/bge-small-en-v1.5 embedding model is used to convert text into vector representations, enabling semantic search and similarity comparisons.

### 3. **Custom Query Optimization**

- **Prompt Engineering**: The system uses a custom prompt template designed for financial analysis, ensuring that responses are precise and relevant to the user's query.
- **Reranking**: The SentenceTransformerRerank model is used to rerank the top results, ensuring that the most relevant documents are prioritized.
- **Cost Control**: A custom postprocessor, SimpleCostControl, is implemented to limit the number of nodes processed, optimizing query performance and reducing computational costs.

### 4. **Data Management**

- **Document Conversion**: SEC filing data is converted into LlamaIndex documents with rich metadata, allowing for detailed and context-aware queries.
- **Structured Data**: The system processes structured data from SEC filings, including financial metrics, risk factors, and strategic updates, providing comprehensive insights.

### 5. **Enhanced Response Formatting**

- **Contextual Summaries**: The system generates contextual summaries of query results, providing users with a high-level overview of the most relevant filings.
- **Financial Terminology**: Responses use precise financial terminology and include comparisons of year-over-year (YoY) and quarter-over-quarter (QoQ) changes when relevant.
- **Structured Outputs**: Responses are formatted in a structured manner, including sections such as Company Overview, Key Metrics, Risk Analysis, and Investment Considerations.

### 6. **Interactive Query System**

- **Interactive Querying**: Users can interactively query the system, refining their searches with filters and receiving instant feedback.
- **Error Handling**: The system includes robust error handling to ensure that users are informed of any issues and can retry their queries if necessary.

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

1. Start the Jupyter notebook or Python script:

   ```bash
   jupyter notebook
   ```

2. Execute the cells in the notebook to initialize the system and run queries.

## Usage

1. **Enter Your Query**: Type your question about SEC filings in the input box.
2. **View Results**: The system will display a summary of the most relevant filings, along with detailed metadata and financial analysis.

## Example Queries

- "What are the risk factors for Tesla in 2024?"
- "Compare Apple's and Microsoft's revenue growth over the past 3 years."
- "What are the details of OpenAI's IPO as mentioned in their S-1 filing?"
- "Compare the revenue growth rates of the tech sector (Apple, Microsoft, Alphabet) vs. the automotive sector (Tesla)."
- "What is the future outlook for Amazon and Alphabet based on their 10-K filings?"
- "What regulatory challenges are mentioned in Microsoft's 8-K filings?"
- "What are the key strategic updates from Tesla's recent 10-Q filing?"
- "Compare the cash reserves of Apple, Microsoft, and Alphabet."
- "What is the EPS for Apple, Microsoft, and Tesla in their latest filings?"
- "Provide details of Microsoft's acquisition of Activision Blizzard."

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- **Groq** for providing the powerful Mixtral-8x7b-32768 model.
- **Hugging Face** for the BAAI/bge-small-en-v1.5 embedding model.
- **LlamaIndex** for the robust document indexing and querying framework.

---
