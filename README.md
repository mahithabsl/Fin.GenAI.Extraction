# FinGen.AI - EDGAR Filing Analysis Tool

A powerful tool for analyzing and extracting information from EDGAR filings using advanced AI techniques. This project provides an interactive interface to analyze 10-K filings, extract key financial information, and visualize the results.

## Features

- 🔍 Interactive EDGAR filing analysis through a Streamlit interface
- 📊 Extraction of key financial metrics from 10-K filings
- 🤖 Advanced AI-powered text analysis using sentence transformers
- 🔄 Efficient document chunking and indexing
- 📈 Vector database integration with Pinecone
- 🎯 Precise information retrieval and question answering
- 📱 Modern and responsive web interface

## Prerequisites

- Python 3.8+
- Pinecone API key
- Groq API key
- LangSmith API key (optional, for tracing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sec_Filings_Project.git
cd Sec_Filings_Project
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
LANGSMITH_API_KEY=your_langsmith_key
```

## Project Structure

```
FinGen.AI.Extraction/
├── src/                    # Source code
│   ├── main.py            # Main application entry point
│   ├── chunking/          # Text chunking modules
│   ├── embeddings/        # Embedding generation
│   ├── retrieval/         # Document retrieval
│   ├── indexing/          # Vector database indexing
│   └── rag/               # RAG implementation
├── data/                  # Data storage
├── tests/                 # Test files
├── static/               # Static assets
└── requirements.txt      # Project dependencies
```

## Usage

1. Start the Streamlit application:
```bash
cd FinGen.AI.Extraction
streamlit run src/main.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter the following information:
   - CIK (Company Identifier)
   - Year of the filing
   - Split (train/test/validate)

4. Click "Analyze" to process the filing

## Features in Detail

### Financial Information Extraction
The tool extracts key financial metrics including:
- Total stockholders
- Employee headcount
- Net sales
- Total cash and cash equivalents
- Quarterly cash dividend

### Document Processing
- Automatic document chunking
- Vector embeddings generation
- Efficient indexing in Pinecone
- Smart retrieval of relevant context

### Interactive Interface
- Real-time analysis results
- Tabbed interface for results and filing content
- Clean and organized data presentation
- Source chunk visualization

