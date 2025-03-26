import os
import json
import streamlit as st
from typing import List, Dict, Optional
from dataclasses import dataclass
from pinecone import Pinecone
import warnings
from pathlib import Path
from chunking.chunker import TextChunker
from embeddings.SentenceTransformer import SentenceTransformersEmbedder
from retrieval.retriever import PineconeRetriever
from indexing.index import Indexer
from rag.answer import QueryAnswerer
from utilities import download_edgar_entry_for_cik
from html_renderer import EdgarHTMLRenderer
# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class Config:
    """Configuration class for EDGAR analysis settings"""
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")

class EdgarAnalyzer:
    """Main class for EDGAR filing analysis"""
    
    def __init__(self):
        self.config = Config()
        self._setup_environment()
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
        self.index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
        self.embedder = SentenceTransformersEmbedder()
        self.indexer = Indexer(self.embedder, self.index)
        self.answerer = QueryAnswerer()
        self.retriever = PineconeRetriever(self.config.PINECONE_INDEX_NAME)
        self.chunker = TextChunker(model_name="sentence-transformers/all-mpnet-base-v2")
        
    def _setup_environment(self):
        """Set up environment variables and configurations"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["GROQ_API_KEY"] = self.config.GROQ_API_KEY
        os.environ["PINECONE_API_KEY"] = self.config.PINECONE_API_KEY
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = self.config.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = self.config.LANGSMITH_PROJECT

    def _get_file_paths(self, cik: str, year: int, split: str) -> Dict[str, Path]:
        """Get file paths for data and chunks"""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data' / f'edgar_corpus_{year}' / split
        
        return {
            'data_file': data_dir / f'{cik}_{year}.json',
            'chunks_file': data_dir / f'{cik}_{year}_chunks.json'
        }

    def _process_document(self, cik: str, year: int, split: str) -> Optional[List[Dict]]:
        """Process and chunk the document"""
        paths = self._get_file_paths(cik, year, split)
        
        if not paths['data_file'].exists():
            results = download_edgar_entry_for_cik(cik, years=[year], splits=[split])
            if not results:
                return None

        with open(paths['data_file'], 'r') as f:
            data = json.load(f)

        if paths['chunks_file'].exists():
            with open(paths['chunks_file'], 'r', encoding='utf-8') as f:
                return json.load(f)

        chunks = self.chunker.chunk_data(data, cik, year, split)
        with open(paths['chunks_file'], 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        st.success("✅ Chunking completed!")
        return chunks

    def analyze_filing(self, cik: str, year: int, split: str) -> Dict[str, Dict[str, str]]:
        """Analyze EDGAR filing and return results"""
        results = {}
        
        if not self.retriever.is_file_indexed_in_pinecone(cik, year, split):
            st.info("File is not indexed in Pinecone")
            chunks = self._process_document(cik, year, split)
            self.indexer.index_chunks(chunks, cik, year, split)
            st.success("✅ File indexed in Pinecone")
                
        else:
            st.success("✅ File already indexed in Pinecone")
            
        shorter_questions = [
            'Total stockholders',
            'Employee headcount',
            'Net sales',
            'Total cash and cash equivalents',
            'Quarterly cash dividend'
        ]

        financial_questions = [
            "How many record holders of the common stock were reported for the latest year?",
            "What is the employee headcount for the latest year?",
            "What was the net sales for the latest year?",
            "What was the total cash and cash equivalents for the latest year?",
            "What is the quarterly cash dividend declared for the latest year?"
        ]

        for _id in range(len(financial_questions)):
            st.info(f"Analyzing questions")
            context, chunk_ids = self.retriever.retrieve_documents(financial_questions[_id], cik, year, split)
            if context:
                answer, used_chunk_ids = self.answerer.answer_query(financial_questions[_id], context, chunk_ids)
                results[shorter_questions[_id]] = {
                    'answer': answer,
                    'chunk_ids': used_chunk_ids
                }

        return results
    
    def analyze_filing_graph(self, cik: str, year: int, split: str) -> Dict[str, str]:
        """Analyze EDGAR filing and return results"""
        results = {}
        return results  



def main():
    """Main Streamlit application"""
    st.title("📊 EDGAR Filing Analysis")
    st.markdown("""
    Enter the CIK, year, and split to begin analysis.
    """)
    # User inputs
    cik = st.text_input("Enter CIK:", "29669")
    year = st.number_input("Enter Year:", min_value=1994, max_value=2023, value=2018)
    split = st.selectbox("Select Split:", ["train", "test", "validate"])
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Analysis Results", "Filing Content"])


    if st.button("🚀 Analyze", type="secondary"):
        with st.spinner("⚙️ Processing..."):
            analyzer = EdgarAnalyzer()
            results = analyzer.analyze_filing(cik, year, split)
            
            with tab1:
                st.subheader("Analysis Results")
                
                # Create a list to store all results
                table_data = []
                
                for question, result in results.items():
                    # Format source chunks for better readability
                    source_chunks = "\n\n".join([f"• {chunk}" for chunk in result['chunk_ids']])
                    
                    table_data.append({
                        'Data Point': question,
                        'Answer': result['answer'],
                        'Source Chunks': source_chunks
                    })
                
                # Display results in a clean table with full width
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Data Point": st.column_config.TextColumn(
                            "Data Point",
                            width="medium",
                        ),
                        "Answer": st.column_config.TextColumn(
                            "Answer",
                            width="small",
                        ),
                        "Source Chunks": st.column_config.TextColumn(
                            "Source Chunks",
                            width="large",
                        ),
                    }
                )
                
                # Add a success message
                st.success('✅ Results displayed successfully')
            
            with tab2:
                st.subheader("Filing Content")
                renderer = EdgarHTMLRenderer()
                # Construct the file path for the HTML file
                project_root = Path(__file__).parent.parent
                file_path = project_root / 'data' / f'edgar_corpus_{year}' / split / f'{cik}.html'
                
                if not file_path.exists():
                    st.error(f"No HTML file found for CIK {cik} in year {year}")
                    return
                    
                renderer.render_filing(file_path)

if __name__ == "__main__":
    # Create data directory
    project_root = Path(__file__).parent.parent
    (project_root / 'data').mkdir(exist_ok=True)
    
    # Run the Streamlit app
    main()