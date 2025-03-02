
# Financial Document Analysis with LlamaIndex 🚀

![License](https://img.shields.io/badge/license-MIT-blue)

A **hybrid search system** built with **LlamaIndex** to analyze financial documents (e.g., SEC 10-K filings) using **vector search** and **BM25 keyword search**. This project demonstrates how to efficiently retrieve and analyze financial data using **Large Language Models (LLMs)** and **LlamaIndex**.

---

## Features ✨

- **Hybrid Search**: Combines **vector embeddings** (semantic search) and **BM25** (keyword search) for accurate document retrieval.
- **Chunking**: Splits large PDFs into manageable chunks for efficient processing.
- **LLM Integration**: Uses **Groq's Mixtral-8x7b-32768** model for generating concise, well-formatted answers.
- **Interactive Querying**: Allows users to ask financial questions and get detailed, context-aware responses.
- **Rate Limiting**: Protects against API overuse with built-in rate limiting.

---

## Technologies Used 🛠️

- **LlamaIndex**: For document indexing, retrieval, and hybrid search.
- **Groq**: For LLM-powered answer generation.
- **Hugging Face Embeddings**: For vector embeddings (`BAAI/bge-small-en-v1.5`).
- **PyPDF2**: For extracting text from PDFs.
- **Python**: Core programming language.

---

## Installation 💻

### Prerequisites

- Python 3.8+
- Groq API key (set in `.env` file)

### Steps

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your Groq API key to a `.env` file:

   ```ini
   GROQ_API_KEY=your_api_key_here
   ```

3. Place your financial PDFs (e.g., SEC 10-K filings) in the `financial_pdfs` folder.

---

Let me know if you'd like further adjustments! 🚀

---

## Usage 🚀

### Running the Project

1. Start the interactive query system:

   ```bash
   python main.py
   ```

2. Enter your financial questions. For example:

   ```bash
   Enter your financial question (type 'exit' to quit): What are the key risks mentioned in the 10-K filings?
   ```

3. The system will:
   - Retrieve relevant document chunks using **hybrid search**.
   - Generate a concise, well-formatted answer using **Groq's Mixtral-8x7b-32768** model.

---

## How It Works 🧠

### 1. Document Processing

- PDFs are loaded and split into chunks using **LlamaIndex's `SentenceSplitter`**.
- Each chunk is stored as a `Document` object with metadata (e.g., source file name).

### 2. Hybrid Search

- **Vector Search**: Uses **Hugging Face embeddings** to find semantically similar chunks.
- **BM25 Search**: Finds chunks with matching keywords.
- Results are combined and deduplicated for accuracy.

### 3. Answer Generation

- Relevant chunks are passed to **Groq's Mixtral-8x7b-32768** model.
- The LLM generates a concise, well-formatted answer based on the provided context.

---

## Key LlamaIndex Concepts Implemented 🔍

### 1. **Document Indexing**

- Documents are indexed using **LlamaIndex's `VectorStoreIndex`**.
- Supports efficient retrieval of relevant chunks.

### 2. **Hybrid Retrieval**

- Combines **vector search** (semantic understanding) and **BM25** (keyword matching).
- Ensures both relevance and precision in search results.

### 3. **Chunking**

- Large documents are split into smaller chunks for efficient processing.
- Improves retrieval accuracy and reduces computational overhead.

### 4. **Metadata Handling**

- Each chunk is tagged with metadata (e.g., source file name).
- Enables traceability and context-aware retrieval.

---

## Example Queries 💡

1. **Risk Analysis**:

   ```bash
   What are the key risks mentioned in the 10-K filings?
   ```

2. **Financial Metrics**:

   ```bash
   What is the revenue growth trend over the past 3 years?
   ```

3. **Comparative Analysis**:

   ```bash
   Compare the risk factors in the 2022 and 2023 10-K filings.
   ```

---

## Configuration ⚙️

- **Chunk Size**: Adjust the `chunk_size` parameter in `get_pdf_docs()` for optimal performance.
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
- **Hugging Face**: For the embedding models.

---

This `README.md` provides a **professional overview** of your project, highlighting the **LlamaIndex implementation** and its key features. Let me know if you’d like to add or modify anything! 🚀
