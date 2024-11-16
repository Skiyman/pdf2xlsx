from tempfile import NamedTemporaryFile

import streamlit as st
import json

from openpyxl.workbook import Workbook

from api.models.models import ImagePredict
from src.errors.no_table_error import NoTableError
from src.errors.requered_headers_error import RequiredHeadersError
from src.models.table_data_extractor import TableDataExtractor
from src.consts.path_consts import JSON_PATH


class DocumentDataExtractor:
    def __init__(
            self,
            images_data: list[ImagePredict],
            session_id: str,
            progress_bar: st.progress = None,
    ):
        self.images_data = images_data
        self.progress_bar = progress_bar
        self.session_id = session_id

    # def _save_data(self, data: list) -> str:
    #     json_path = f"{JSON_PATH}{self.session_id}.json"
    #     with open(json_path, "w") as file:
    #         json.dump(data, file, ensure_ascii=False, indent=4)
    #         return json_path

    def parse_data(self):
        wb = Workbook()
        wb
        progress_counter = 0
        images_amount = len(self.images_data)

        progress_increment = 40 // images_amount
        progress = 60

        if self.progress_bar:
            self.progress_bar.progress(progress, text=f"Parsing Information {progress_counter}/{images_amount}")
        else:
            print(f"Parsing Information {progress_counter}/{images_amount}")
        print("Start")
        for image in self.images_data:
            # Skip images without tables
            if not image.data:
                progress += progress_increment
                progress_counter += 1
                print(progress_counter)
                if self.progress_bar:
                    self.progress_bar.progress(progress, text=f"Parsing Information {progress_counter}/{images_amount}")
                else:
                    print(f"Parsing Information {progress_counter}/{images_amount}")
                continue

            for table in image.data:
                try:
                    extractor = TableDataExtractor(table.path, wb=wb)
                    extractor.get_data()
                except NoTableError:
                    continue

            progress += progress_increment
            progress_counter += 1
            if self.progress_bar:
                self.progress_bar.progress(progress, text=f"Parsing Information {progress_counter}/{images_amount}")
            else:
                print(f"Parsing Information {progress_counter}/{images_amount}")
        del wb['Sheet']
        with NamedTemporaryFile() as f:
            wb.save(f.name)
            f.seek(0)
            return f.read()

