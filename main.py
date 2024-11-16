import asyncio
import uuid
from pathlib import Path

import streamlit as st
from pydantic import TypeAdapter

from api.models.models import ImagePredict
from src.models.table_detector import TableDetector
from src.preprocessing.image_preprocessor import Preprocessor
from src.shared.document_data_extractor import DocumentDataExtractor
from src.shared.pdf_converter import PdfConverter

st.header("Documet's OcR")
uploaded_file = st.file_uploader("**Upload PDF**")


if uploaded_file is not None:
    session_id = str(uuid.uuid4())

    progress_bar = st.progress(0, text="Extracting Images")

    # Crop Pdf to images
    converter = PdfConverter(uploaded_file, session_id)
    result = asyncio.run(converter.start_convert())

    progress_bar.progress(30, text="Detecting Tables")
    # Detecting tables on Images and crop them
    table_detector = TableDetector(result, session_id)
    table_detector.make_predict()
    files = table_detector.crop_images()
    print(files)
    progress_bar.progress(50, text="Preprocessing images")
    # Run preprocessing
    preprocessor = Preprocessor(session_id)
    preprocessor.fully_preprocess()

    ta = TypeAdapter(list[ImagePredict])
    files = ta.validate_python(files)
    progress_bar.progress(60, text="Parsing Information")

    # Extracting data in JSON format
    extractor = DocumentDataExtractor(
        files,
        session_id,
        progress_bar
    )
    result = extractor.parse_data()

    st.download_button(
        label="Download Excel",
        file_name="result.xlsx",
        mime="application/vnd.ms-excel",
        data=result,
    )
    # progress_bar.progress(100, text="Done")

# if __name__ == "__main__":
#     print("log")
#     clear_cache()
#     create_cache_path()
