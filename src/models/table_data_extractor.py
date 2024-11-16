import re

import pandas as pd
from img2table.document import Image
from img2table.ocr import EasyOCR
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.workbook import Workbook

from src.errors.no_table_error import NoTableError
from src.errors.requered_headers_error import RequiredHeadersError


class TableDataExtractor:
    def __init__(
            self,
            table_path: str,
            wb: Workbook = None,
    ):
        self.ocr = EasyOCR(lang=["ru"])
        self.table = Image(table_path, detect_rotation=True)

        self.table_df = self._get_table_df()
        self.wb = wb

    def get_data(self) -> Workbook:
        return self._extract_data()

    def _get_table_df(self) -> pd.DataFrame:
        extracted_tables = self.table.extract_tables(ocr=self.ocr, implicit_rows=False)

        if extracted_tables:
            table_data = extracted_tables.pop().df
            return table_data
        else:
            raise NoTableError(self.table.src)

    def _extract_data(self) -> Workbook:
        page_number = str(len(self.wb.get_sheet_names()))
        ws = self.wb.create_sheet(page_number)

        rows = dataframe_to_rows(self.table_df, index=False, header=False)
        for r_idx, row in enumerate(rows, 1):
            for c_idx, value in enumerate(row, 1):
                ws.cell(row=r_idx, column=c_idx, value=value)

        return self.wb
