# Medical Research Assistant with LlamaIndex 🚀

![License](https://img.shields.io/badge/license-MIT-blue)  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9.0-orange)  
![Groq](https://img.shields.io/badge/Groq-API-green)  
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow)

A **hybrid search system** built with **LlamaIndex** to analyze medical research papers. This project demonstrates my ability to handle **low-level data preprocessing** (manual text extraction and cleaning) and **high-level AI systems** (RAG pipelines with hybrid search). It showcases my skills in **data engineering**, **AI integration**, and **production-grade system design**.

---

## Features ✨

- **Data Preprocessing**: Extracts text from PDFs, cleans it, and stores it in JSON files for further processing.
- **Hybrid Search**: Combines **vector embeddings** (semantic search) and **BM25** (keyword search) for accurate document retrieval.
- **RAG Pipeline**: Implements a **Retrieval-Augmented Generation (RAG)** system for context-aware question answering.
- **Interactive Querying**: Allows users to ask medical questions and get detailed, well-formatted answers.
- **Persistent Storage**: Saves indexed documents for faster reloads.
- **Query Caching**: Improves response times by caching frequently asked questions.

---

## Technologies Used 🛠️

- **LlamaIndex**: For document indexing, retrieval, and hybrid search.
- **Groq**: For LLM-powered answer generation.
- **Hugging Face Embeddings**: For vector embeddings (`BAAI/bge-small-en-v1.5`).
- **PyPDF2**: For extracting text from PDFs.
- **Python**: Core programming language.

---

## Project Evolution 🚀

### **Phase 1: Data Preprocessing**

- **Goal**: Demonstrate my ability to handle **low-level data processing**.
- **Implementation**:
  - Extracts text from PDFs word-by-word using **PyPDF2**.
  - Cleans the text by removing headers, footers, and special characters.
  - Saves the raw and cleaned text in JSON files for further processing.
- **Files**:
  - `llamaindex-basics_medical_document_q&a.ipynb`

### **Phase 2: RAG System**

- **Goal**: Build a **production-grade RAG system** using LlamaIndex.
- **Implementation**:
  - Uses **LlamaIndex** for document indexing and retrieval.
  - Implements **hybrid search** (vector + keyword) for accurate results.
  - Integrates **Groq's Llama-3.1-8b-instant** model for answer generation.
  - Adds **persistent storage** and **query caching** for efficiency.
- **Files**:
  - `better.ipynb`

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

3. Place your medical research PDFs in the `medical_papers` folder.

---

## Usage 🚀

### Running the Project

1. Start the interactive query system:

   ```bash
   python main.py
   ```

2. Enter your medical questions. For example:

   ```bash
   Enter your medical question (or 'exit' to quit): What are the latest treatments for diabetes insipidus?
   ```

3. The system will:
   - Retrieve relevant document chunks using **hybrid search**.
   - Generate a concise, well-formatted answer using **Groq's Llama-3.1-8b-instant** model.

---

## How It Works 🧠

### 1. Data Preprocessing

- **Text Extraction**: PDFs are processed word-by-word using **PyPDF2**.
- **Text Cleaning**: Removes headers, footers, and special characters.
- **JSON Storage**: Saves raw and cleaned text in JSON files for further processing.

### 2. Document Indexing

- PDFs are loaded and split into chunks using **LlamaIndex's `SentenceSplitter`**.
- Each chunk is stored as a `Document` object with metadata (e.g., source file name).

### 3. Hybrid Search

- **Vector Search**: Uses **Hugging Face embeddings** to find semantically similar chunks.
- **BM25 Search**: Finds chunks with matching keywords.
- Results are combined and deduplicated for accuracy.

### 4. Answer Generation

- Relevant chunks are passed to **Groq's Llama-3.1-8b-instant** model.
- The LLM generates a concise, well-formatted answer based on the provided context.

### 5. Persistent Storage

- Indexed documents are saved to disk for faster reloads.
- New documents can be added without rebuilding the entire index.

### 6. Query Caching

- Frequently asked questions are cached to improve response times.

---

## Key Concepts Implemented 🔍

### 1. **Data Preprocessing**

- Demonstrates my ability to handle **raw data** and prepare it for AI systems.
- Highlights my attention to detail and understanding of **data engineering**.

### 2. **Hybrid Retrieval**

- Combines **vector search** (semantic understanding) and **BM25** (keyword matching).
- Ensures both relevance and precision in search results.

### 3. **RAG Pipeline**

- Implements a **Retrieval-Augmented Generation (RAG)** system for context-aware question answering.
- Showcases my ability to integrate **LLMs** with **document retrieval systems**.

### 4. **Persistent Storage**

- Indexed documents are saved to disk for faster reloads.
- Supports dynamic updates without rebuilding the entire index.

### 5. **Query Caching**

- Frequently asked questions are cached to improve response times.
- Demonstrates my focus on **performance optimization**.

---

## Example Queries 💡

1. **Diabetes Treatment**:

   ```bash
   What are the latest treatments for diabetes insipidus?
   ```

2. **Hypertension Management**:

   ```bash
   What are the recommended treatment algorithms for pulmonary arterial hypertension?
   ```

3. **COVID-19 Vaccines**:

   ```bash
   What are the latest updates on COVID-19 vaccine development?
   ```

4. **Cancer Immunotherapy**:

   ```bash
   How are nanoparticles being used in cancer immunotherapy?
   ```

5. **AI in Radiology**:

   ```bash
   What are the key applications of AI in radiology?
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
