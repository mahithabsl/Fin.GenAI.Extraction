import subprocess
import sys

def setup_spacy():
    print("Installing spaCy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    
    print("\nDownloading English language model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("\nSpaCy setup complete!")

if __name__ == "__main__":
    setup_spacy() 