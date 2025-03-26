from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from pinecone import Pinecone
from wasabi import msg
import streamlit as st
import re
import os
from typing import List, Dict, Tuple, Optional, Union

class PineconeRetriever:
    """
    A class for retrieving and reranking documents from a Pinecone vector database.
    Implements semantic search with cross-encoder reranking.
    """

    def __init__(self, index_name: str, k: int = 10, text_field: str = "text"):
        """
        Initialize the PineconeRetriever with specified parameters.
        
        Args:
            index_name (str): Name of the Pinecone index to use
            k (int, optional): Number of documents to retrieve. Defaults to 5
            text_field (str, optional): Field name containing the text. Defaults to "text"
        """
        self.index_name = index_name
        self.pinecone_api_key = 'pcsk_2LZnji_5FkeVeGnpuS6ENKR2SWXEKHzsThAMPfXPe1tANF63vnYyZRSQqm7m6aPUi62ovm'
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        self.k = k
        self.text_field = text_field
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

        msg.info(f"Initialized PineconeRetriever with index: {self.index_name}")

    def is_file_indexed_in_pinecone(self, cik: str, year: int, split: str) -> bool:
        """
        Check if a file with given parameters exists in the Pinecone index.
        
        Args:
            cik (str): Company CIK number
            year (int): Filing year
            split (str): Dataset split
            
        Returns:
            bool: True if file is indexed, False otherwise
        """
        try:
            st.info("Checking if file is indexed in Pinecone")
            query_response = self.index.query(
                vector=[0] * 768,
                filter={"cik": cik, "year": year, "split": split},
                top_k=1,
                include_metadata=True,
                namespace='ns1'
            )
            return len(query_response.matches) > 0
        except Exception as e:
            print(f"Error checking Pinecone index: {str(e)}")
            return False

    def retrieve_documents(self, query: str, cik: str, year: int, split: str, k: int = 10) -> Tuple[Optional[str], List[str]]:
        """
        Retrieve and rerank documents based on the query and metadata filters.
        
        Args:
            query (str): Search query
            cik (str): Company CIK number
            year (int): Filing year
            split (str): Dataset split
            k (int, optional): Number of documents to retrieve. Defaults to 10
            
        Returns:
            Tuple[Optional[str], List[str]]: Tuple containing:
                - Concatenated reranked documents or None if retrieval fails
                - List of chunk IDs used
        """
        try:
            retriever = PineconeRetriever(index_name=self.index_name, k=k)
            reranked_documents, reranked_documents_text, chunk_ids = retriever.get_relevant_documents(query, cik, year, split)
            return reranked_documents_text, chunk_ids
        except Exception as e:
            return None, []

    def get_relevant_documents(self, query: str, cik: Optional[str] = None, 
                             year: Optional[int] = None, split: Optional[str] = None) -> Tuple[List[str], str, List[str]]:
        """
        Get relevant documents for a query with optional metadata filters.
        
        Args:
            query (str): Search query
            cik (Optional[str]): Company CIK number
            year (Optional[int]): Filing year
            split (Optional[str]): Dataset split
            
        Returns:
            Tuple[List[str], str, List[str]]: Tuple containing:
                - List of reranked documents
                - Concatenated reranked documents
                - List of chunk IDs
        """
        try:
            msg.info(f"Retrieving top {self.k} documents for query: '{query}'")
            result = self.query_index(query, cik, year, split)
            st.success("✅ Retrieving documents done")
            formatted_documents, chunk_ids = self.format_docs(result['matches'])
            reranked_documents, reranked_documents_text = self.get_reranked_contexts(query, formatted_documents, self.k)
            return reranked_documents, reranked_documents_text, chunk_ids
        except Exception as e:
            msg.error(f"Error retrieving documents: {str(e)}")
            return [], "", []

    def format_docs(self, documents: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Format retrieved documents into a clean text format.
        
        Args:
            documents (List[Dict]): List of document dictionaries from Pinecone
            
        Returns:
            Tuple[List[str], List[str]]: Tuple containing:
                - List of formatted document strings
                - List of chunk IDs
        """
        docs = []
        chunk_ids = []
        for doc in documents:
            if 'metadata' in doc:
                content = doc['metadata'].get('content', "No content Available")
                chunk_id = doc['id']
            else:
                content = "No content Available"
                chunk_id = "Untitled"

            cleaned_content = self.clean_text(content)
            docs.append(f"{cleaned_content}\n\n(Source: {chunk_id})")
            chunk_ids.append(chunk_id)

        return docs, chunk_ids

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        text = str(text)
        cleaned_text = re.sub(r"(\\)+n*|[\t\n\r]", '', text)
        cleaned_text = re.sub(r' {2,}', " ", cleaned_text)
        return cleaned_text

    def get_query_vector(self, query: str) -> List[float]:
        """
        Generate embedding vector for a query.
        
        Args:
            query (str): Query text
            
        Returns:
            List[float]: Embedding vector
        """
        return self.embedding_model.encode(query).tolist()

    def query_index(self, query: str, cik: Optional[str] = None, 
                   year: Optional[int] = None, split: Optional[str] = None) -> Dict:
        """
        Query the Pinecone index with filters.
        
        Args:
            query (str): Search query
            cik (Optional[str]): Company CIK number
            year (Optional[int]): Filing year
            split (Optional[str]): Dataset split
            
        Returns:
            Dict: Pinecone query response
        """
        vector = self.get_query_vector(query)
        filter_dict = {}
        if cik:
            filter_dict['cik'] = cik
        if year:
            filter_dict['year'] = year
        if split:
            filter_dict['split'] = split
            
        response = self.index.query(
            namespace="ns1",
            vector=vector,
            top_k=self.k,
            include_values=True,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return response

    def generate_cross_encoder_score(self, query: str, contexts: List[str]) -> List[float]:
        """
        Generate relevance scores using cross-encoder.
        
        Args:
            query (str): Search query
            contexts (List[str]): List of context passages
            
        Returns:
            List[float]: List of relevance scores
        """
        return self.RERANKER.predict([(query, text) for text in contexts])

    def get_reranked_contexts(self, query: str, contexts: List[str], num_docs: int) -> Tuple[List[str], str]:
        """
        Rerank contexts based on cross-encoder scores.
        
        Args:
            query (str): Search query
            contexts (List[str]): List of context passages
            num_docs (int): Number of documents to return
            
        Returns:
            Tuple[List[str], str]: Tuple containing:
                - List of reranked documents
                - Concatenated reranked documents
        """
        scores = self.generate_cross_encoder_score(query, contexts)
        sorted_scores = scores.argsort()[::-1]
        reranked_paragraphs = []
        for item in sorted_scores:
            reranked_paragraphs.append(contexts[item])
        reranked_paragraphs = reranked_paragraphs[:num_docs]
        st.success("✅ Reranking documents done")
        return reranked_paragraphs, "\n\n".join(reranked_paragraphs)
