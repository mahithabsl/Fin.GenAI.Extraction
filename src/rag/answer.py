import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from prompts.query_prompt import QUERY_TEMPLATE
from typing import Tuple, List

class QueryAnswerer:
    def __init__(self):
        """
        Initialize the QueryAnswerer with Groq API key and LLM chain.
        """
        # Get API key from environment variable
        st.info("Initializing LLM")
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0,
            max_tokens=None
        )
        
        # Create the prompt template
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=QUERY_TEMPLATE
        )
        
        # Create the chain with the prompt template
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )
    
    def answer_query(self, query: str, context: str, chunk_ids: List[str]) -> Tuple[str, List[str]]:
        """
        Generate an answer to the query using the provided context.
        
        Args:
            query (str): The question to answer
            context (str): The relevant context from retrieved documents
            chunk_ids (List[str]): List of chunk IDs used in the context
            
        Returns:
            Tuple[str, List[str]]: The generated answer and list of chunk IDs used
        """
        try:
            # Run the chain to generate the answer
            response = self.chain.run(
                query=query,
                context=context
            )
            st.success(" ✅ Generated answer !!")
            return response.strip(), chunk_ids
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
