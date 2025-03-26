from embeddings.SentenceTransformer import SentenceTransformersEmbedder
from typing import Dict, List, Optional
from pinecone import Pinecone

import sys
import os
import streamlit as st
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Indexer:
    def __init__(self, embedder: SentenceTransformersEmbedder, index: Pinecone):
        """
        Initialize the Indexer with required parameters.
        
        Args:
            embedder (SentenceTransformersEmbedder): The embedding model to use
            index_name (str): Name of the Pinecone index
            cik (str): Company CIK number
            year (int): Year of the filing
            split (str): Dataset split (train/test/validate)
        """
        self.embedder = embedder
        self.index = index



    def get_index_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dict: Statistics about the index
        """
        try:
            stats = self.index.describe_index(self.index_name)
            return stats
        except Exception as e:
            print(f"Error getting index stats: {str(e)}")
            raise

    def index_chunks(self, chunks: Dict, cik: str, year: int, split: str) -> None:
        """
        Index chunks into Pinecone.
        
        Args:
            chunks (Dict): Dictionary containing text chunks to index
        """
        # Index chunks into Pinecone
        st.info("\nIndexing chunks into Pinecone...")
        for section in chunks:
            if section not in ['cik', 'year', 'split']:     # Skip metadata fields
                try:
                    section_chunks = chunks[section]['chunks']
                    item_number = chunks[section]['item_number']
                    item_name = chunks[section]['item_name']

                    # Get embeddings for all chunks in this section
                    embeddings = self.embedder.vectorize(section_chunks)
                    # Convert embeddings to list if they're numpy arrays
                    embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

            
                    # Prepare vectors for Pinecone
                    vectors = []
                    for i, (chunk, embedding) in enumerate(zip(section_chunks, embeddings)):
                        vector = {
                            "id": f"{cik}_{year}_{section}_{i}",
                            "values": embedding,
                            "metadata": {
                                "cik": cik,
                                "year": year,
                                "split": split,
                                "section": section,
                                "item_number": item_number,
                                "item_name": item_name,
                                "chunk_index": i,
                                "content": chunk
                            }
                        }
                        vectors.append(vector)

                    # Batch upsert to Pinecone (in smaller batches)
                    batch_size = 10
                    for i in range(0, len(vectors), batch_size):
                        batch = vectors[i:i + batch_size]
                        try:
                            self.index.upsert(vectors=batch, namespace="ns1")

                        except Exception as e:
                            print(f"Error upserting batch: {str(e)}")
                   
                except Exception as e:
                    print(f"Error indexing section {section}: {str(e)}")
        st.success('✅  Completed indexing')
