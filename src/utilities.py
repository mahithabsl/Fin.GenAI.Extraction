import os
import json
import streamlit as st
from datasets import load_dataset

def download_edgar_entry_for_cik(cik, years, splits):
    """
    Downloads EDGAR filings for a specific CIK from the specified years and splits.
    
    Parameters:
      cik (str): The CIK to search for
      years (list): List of years to search in
      splits (list): List of splits to search in ('train', 'test', 'validate')
    
    Returns:
      dict: Dictionary containing the filings found for each year and split
    """
    # Get the absolute path to the project root (one level up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = {}
    
    for year in years:
        results[year] = {}
        st.info(f"\nSearching year {year} for CIK {cik}...")
        
        for split in splits:
            st.info(f"Checking {split} split...")
            
            # Check if file already exists
            file_path = os.path.join(project_root, 'data', f'edgar_corpus_{year}', split, f'{cik}_{year}.json')
            if os.path.exists(file_path):
                st.info(f"File already exists at {file_path}, skipping download...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    results[year][split] = json.load(f)
                continue
                
            try:
                # Load the dataset for the specific year and split
                dataset = load_dataset(
                    "eloukas/edgar-corpus",
                    name=f"year_{year}",
                    split=split
                )
                
                # Filter for the specific CIK
                matches = dataset.filter(lambda x: x['cik'] == cik)
                
                if len(matches) > 0:
                    st.info(f"Found {len(matches)} record(s) for year {year} in {split} split")
                    results[year][split] = matches[0]
                    
                    # Use absolute path for saving
                    save_dir = os.path.join(project_root, 'data', f'edgar_corpus_{year}', split)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(matches[0], f, indent=2, ensure_ascii=False)
                    # st.info(f"Saved to {file_path}")
                else:
                    st.info(f"No records found for year {year} in {split} split")
                    return None
                    
            except Exception as e:
                st.error(f"Error processing year {year}, split {split}: {str(e)}")
    
    return results

