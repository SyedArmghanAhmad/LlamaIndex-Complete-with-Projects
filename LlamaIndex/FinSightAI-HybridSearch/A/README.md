
# 📊 FinSight AI: Financial Document Analysis with LlamaIndex 🚀

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9.0-green)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)

Welcome to **FinSight AI**, your AI-powered financial analyst! This project leverages **LlamaIndex** and **Groq's Mixtral-8x7b-32768** model to analyze financial documents (e.g., SEC 10-K filings) with **hybrid search** and **interactive querying**. Whether you're a financial analyst, investor, or researcher, FinSight AI provides **accurate, context-aware insights** from complex financial data.

---

## 🌟 Features

- **Hybrid Search**: Combines **vector embeddings** (semantic search) and **BM25** (keyword search) for precise document retrieval.
- **Chunking**: Splits large PDFs into manageable chunks for efficient processing.
- **LLM Integration**: Uses **Groq's Mixtral-8x7b-32768** model for generating concise, well-formatted answers.
- **Interactive Querying**: Ask financial questions and get detailed, context-aware responses.
- **Dynamic Graphs**: Visualize financial trends and comparisons with **Plotly** graphs.
- **Rate Limiting**: Protects against API overuse with built-in rate limiting.

---

## 🛠️ Technologies Used

- **LlamaIndex**: For document indexing, retrieval, and hybrid search.
- **Groq**: For LLM-powered answer generation.
- **Hugging Face Embeddings**: For vector embeddings (`BAAI/bge-small-en-v1.5`).
- **PyPDF2**: For extracting text from PDFs.
- **Streamlit**: For the interactive web interface.
- **Plotly**: For dynamic and interactive graph visualizations.

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Groq API key (set in `.env` file)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/finsight-ai.git
   cd finsight-ai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq API key to a `.env` file:

   ```ini
   GROQ_API_KEY=your_api_key_here
   ```

4. Place your financial PDFs (e.g., SEC 10-K filings) in the `financial_pdfs` folder.

---

## 🖥️ Usage

### Running the Project

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`.

3. Upload financial PDFs and ask questions. For example:

   - **Trend Analysis**: "What is the revenue growth trend over the past 3 years?"
   - **Risk Analysis**: "What are the key risks mentioned in the 10-K filings?"
   - **Comparative Analysis**: "Compare the risk factors in the 2022 and 2023 10-K filings."

4. The system will:
   - Retrieve relevant document chunks using **hybrid search**.
   - Generate a concise, well-formatted answer using **Groq's Mixtral-8x7b-32768** model.
   - Display dynamic graphs (e.g., line charts, bar charts, pie charts) for visual insights.

---

## 🧠 How It Works

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

### 4. Graph Generation

- Numerical data is extracted from the LLM's response.
- Dynamic graphs (e.g., line charts, bar charts, pie charts) are generated using **Plotly**.

---

## 💡 Example Queries

1. **Income Variability**:
   - *Question*: "What percentage of adults experienced income that varied month-to-month in 2023, and which industries had the highest rates of income variability?"
   - *Graph*: Bar chart comparing income variability across industries.

2. **Childcare Costs**:
   - *Question*: "What was the median monthly childcare cost for parents using paid care, and how does this compare to housing expenses?"
   - *Graph*: Pie chart showing the proportion of childcare costs relative to housing expenses.

3. **Retirement Savings**:
   - *Question*: "What percentage of non-retirees felt their retirement savings were 'on track' in 2023, and how did this vary by education level?"
   - *Graph*: Bar chart comparing retirement savings confidence across education levels.

4. **Inflation Impact**:
   - *Question*: "Compare how inflation affected U.S. households and how it might influence Berkshire Hathaway’s insurance underwriting profits."
   - *Graph*: Line chart showing trends in inflation impact on households and Berkshire’s profits.

---

## ⚙️ Configuration

- **Chunk Size**: Adjust the `chunk_size` parameter in `get_pdf_docs()` for optimal performance.
- **LLM Model**: Switch to a different Groq model by updating `initialize_groq()`.
- **Embedding Model**: Replace `BAAI/bge-small-en-v1.5` with another Hugging Face model if needed.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **LlamaIndex**: For the powerful document indexing and retrieval framework.
- **Groq**: For the high-performance LLM API.
- **Hugging Face**: For the embedding models.
- **Streamlit**: For the interactive web interface.
- **Plotly**: For dynamic and interactive graph visualizations.
