from typing import List, Dict, Optional
from transformers import GPT2TokenizerFast
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    CharacterTextSplitter,
)
from nltk.tokenize import TextTilingTokenizer
import nltk
import ssl
import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class TextChunker:
    """
    A class for chunking text documents into smaller segments using various methods.
    Supports GPT-2 tokenization, NLTK TextTiling, and character/token-based splitting.
    """

    def __init__(self, model_name: str):
        """
        Initialize the TextChunker with specified model and default parameters.
        
        Args:
            model_name (str): Name of the model to use for tokenization
        """
        self.model_name = model_name
        self.character_separators = ["\n\n", "\n", ". ", " ", ""]
        self.character_chunk_size = 1000
        self.character_chunk_overlap = 0
        self.token_chunk_overlap = 0
        self.tokens_per_chunk = 200
        self.nltk_w = 20
        self.nltk_k = 5
        
        # Download NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                # Create SSL context that ignores certificate verification
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                nltk.download('punkt')
            except Exception as e:
                print(f"Warning: Could not download NLTK data: {str(e)}")
                print("Please download punkt tokenizer manually or check your SSL certificates.")

    
    def chunk_data(self, data: Dict, cik_to_search: str, year: int, split: str) -> Dict:
        """
        Process and chunk EDGAR filing data by sections.
        
        Args:
            data (Dict): The EDGAR filing data containing sections
            cik_to_search (str): The CIK number of the company
            year (int): The year of the filing
            split (str): The dataset split (train/test/validate)
            
        Returns:
            Dict: Processed chunks with metadata for each section
        """
        sections = [k for k in data.keys() if k.startswith('section')]
        chunker = TextChunker(model_name=config['chunking']['model'])
        all_chunks = {'cik': cik_to_search, 'year': year, 'split': split}
        st.info("Processing sections with NLTK TextTiling...")

        count_of_chunks = 0
        
        for section in sections:
            all_chunks[section] = {}
    
            text = data[section]
            all_chunks[section]['item_number'] = text.split('.')[0].strip()
            all_chunks[section]['item_name'] = self.extract_item_name(data[section])
            if not text or not isinstance(text, str):
                # st.write(f"Skipping section {section} - invalid content")
                continue
                
            try:
                method = config['chunking']['method']
                text_processor = TextChunker(model_name='nltk')
                chunks = text_processor.chunk_text(text, method='nltk')
                chunks = chunker.chunk_text(text, method=method, config=config['chunking'])
                
                all_chunks[section]['chunks'] = chunks
                count_of_chunks += len(chunks)
            except Exception as e:
                st.error(f"Error processing section {section}: {str(e)}")

        
        return all_chunks
    

    def extract_item_name(self, text: str) -> str:
        """
        Extract section name from text like 'Item 1. Business'.
        
        Args:
            text (str): The text containing the item name
            
        Returns:
            str: Extracted item name or "Unknown Section" if not found
        """
        first_line = text.split('\n')[0].strip()
        try:
            if '.' in first_line:
                section_name = first_line.split('.', 1)[1].strip()
                return section_name
            else:
                return "Unknown Section"
        except:
            return "Unknown Section"


    def split_text_by_character_and_token(self, text: str) -> List[str]:
        """
        Split text using both character and token-based splitting.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks split by both characters and tokens
        """
        character_splitter = RecursiveCharacterTextSplitter(
            separators=self.character_separators,
            chunk_size=self.character_chunk_size,
            chunk_overlap=self.character_chunk_overlap,
        )
        character_split_texts = character_splitter.split_text(text)

        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=self.token_chunk_overlap,
            tokens_per_chunk=self.tokens_per_chunk,
        )
        token_split_texts = []
        for t in character_split_texts:
            token_split_texts += token_splitter.split_text(t)
        return token_split_texts

    def preprocess_text_for_texttiling(self, text: str) -> str:
        """
        Prepare text for TextTiling by ensuring proper paragraph breaks.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text with proper paragraph breaks
        """
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paragraphs) == 1:
            mid_point = len(text) // 2
            paragraphs = [text[:mid_point], text[mid_point:]]
        return '\n\n'.join(paragraphs)

    def tokenize_text_with_texttiling(self, text: str) -> List[str]:
        """
        Split text using NLTK's TextTiling algorithm.
        
        Args:
            text (str): Text to split using TextTiling
            
        Returns:
            List[str]: List of text tiles/chunks
        """
        try:
            processed_text = self.preprocess_text_for_texttiling(text)
            tt = TextTilingTokenizer(w=self.nltk_w, k=self.nltk_k)
            tiles = tt.tokenize(processed_text)
            return [tile.strip() for tile in tiles if tile.strip()]
        except Exception as e:
            return [text]

    def split_text_with_gpt2_tokenizer(self, text: str) -> List[str]:
        """
        Split text using GPT-2 tokenizer.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of text chunks based on GPT-2 tokens
        """
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer, chunk_size=self.tokens_per_chunk,
        )
        return text_splitter.split_text(text)

    def chunk_text(self, text: str, method: str, config: Optional[Dict] = None) -> List[str]:
        """
        Chunk text using the specified method.
        
        Args:
            text (str): Text to chunk
            method (str): Chunking method to use ('gpt2', 'nltk', or 'character_and_token')
            config (Dict, optional): Configuration parameters for chunking
                Possible keys:
                - 'nltk': {'w': int, 'k': int}
                - 'chunk_size': int
                - 'tokens_per_chunk': int
            
        Returns:
            List[str]: List of text chunks
            
        Raises:
            ValueError: If unsupported method is specified
        """
        if config:
            if 'nltk' in config:
                self.nltk_w = config['nltk'].get('w', self.nltk_w)
                self.nltk_k = config['nltk'].get('k', self.nltk_k)
            self.character_chunk_size = config.get('chunk_size', self.character_chunk_size)
            self.tokens_per_chunk = config.get('tokens_per_chunk', self.tokens_per_chunk)
            
        if method.lower() == "gpt2":
            return self.split_text_with_gpt2_tokenizer(text)
        elif method.lower() == "nltk":
            return self.tokenize_text_with_texttiling(text)
        elif method.lower() == "character_and_token":
            return self.split_text_by_character_and_token(text)
        else:
            raise ValueError(f"Unsupported method: {method}")


