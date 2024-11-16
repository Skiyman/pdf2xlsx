import os
from src.consts.path_consts import PDF_PATH, TABLES_PATH, IMG_PATH, CACHE_PATH, JSON_PATH


def create_cache_path():
    if not os.path.isdir(CACHE_PATH):
        os.mkdir(CACHE_PATH)
        os.mkdir(PDF_PATH)
        os.mkdir(TABLES_PATH)
        os.mkdir(IMG_PATH)
        os.mkdir(JSON_PATH)
