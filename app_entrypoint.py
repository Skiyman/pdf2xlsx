import os

from src.scripts.clear_cahce import clear_cache
from src.scripts.create_cache_path import create_cache_path

if __name__ == "__main__":
    create_cache_path()
    clear_cache()
    os.system("streamlit run main.py")


