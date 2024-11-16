import os
import shutil
from src.consts.path_consts import TABLES_PATH, IMG_PATH, PDF_PATH, JSON_PATH


def clear_cache():
    cache_path = [
        PDF_PATH,
        IMG_PATH,
        TABLES_PATH,
    ]

    for i in cache_path:
        files = os.listdir(i)
        if files:
            for file in files:
                try:
                    os.remove(i + file)
                    print(i + file + " was removed")
                except IsADirectoryError:
                    shutil.rmtree(i + file)
                    print(f"Folder: {i + file} was removed")


if __name__ == "__main__":
    clear_cache()
