import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
import time
import random
from pathlib import Path

class EdgarHTMLRenderer:
    """Class to handle HTML rendering of SEC EDGAR filings"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    def read_local_html(self, file_path: Path) -> Optional[str]:
        """Read HTML content from a local file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading local HTML file: {str(e)}")
            return None
    
    def render_filing(self, file_path: Path):
        """Render an SEC EDGAR filing from local file"""
        with st.spinner(f"Loading filing from {file_path}..."):
            html_content = self.read_local_html(file_path)
            if not html_content:
                st.error("Failed to read the filing content. Please check if the file exists.")
                return
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract and display the content
            st.markdown("### SEC EDGAR Filing Content")
            
            # Find the main content area (adjust selectors based on SEC's HTML structure)
            content = soup.find('div', {'class': 'document'})
            if content:
                # Convert HTML to string and render
                st.markdown(content.prettify(), unsafe_allow_html=True)
            else:
                # Try alternative content selectors
                content = soup.find('div', {'class': 'ix-content'})
                if content:
                    st.markdown(content.prettify(), unsafe_allow_html=True)
                else:
                    # If no specific content div is found, render the entire body
                    body = soup.find('body')
                    if body:
                        st.markdown(body.prettify(), unsafe_allow_html=True)
                    else:
                        st.warning("Could not find main content area in the filing")
    
    def render_multiple_filings(self, file_paths: List[Path]):
        """Render multiple SEC EDGAR filings"""
        for file_path in file_paths:
            st.markdown(f"### Filing: {file_path.name}")
            self.render_filing(file_path)
            st.markdown("---")

def main():
    st.set_page_config(page_title="SEC EDGAR Filing Viewer", layout="wide")
    st.title("SEC EDGAR Filing Viewer")
    
    # Example file paths
    project_root = Path(__file__).parent.parent
    file_paths = [
        project_root / 'data' / 'edgar_corpus_2018' / 'train' / '29669.html',
        project_root / 'data' / 'edgar_corpus_2019' / 'train' / '29669.html',
        project_root / 'data' / 'edgar_corpus_2020' / 'train' / '29669.html'
    ]
    
    renderer = EdgarHTMLRenderer()
    renderer.render_multiple_filings(file_paths)

if __name__ == "__main__":
    main() 