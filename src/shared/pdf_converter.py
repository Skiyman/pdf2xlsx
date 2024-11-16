import os
import shutil

import aiofiles
import aiofiles.os
from starlette.datastructures import UploadFile

from streamlit.runtime.uploaded_file_manager import UploadedFile
from pdf2image import convert_from_path
from src.consts.path_consts import PDF_PATH, IMG_PATH


class PdfConverter:
    def __init__(self, file: UploadedFile | UploadFile, session_id: str):
        self.filename = session_id
        self.file = file
        self.pdf_path = ""
        self.img_path = f"{IMG_PATH}{self.filename}/"

    async def start_convert(self) -> list[str]:
        self.pdf_path = await self.save_pdf()
        images_path = self.convert_to_img()

        await aiofiles.os.remove(self.pdf_path)
        return images_path

    async def save_pdf(self) -> str:
        path = f"{PDF_PATH}{self.filename}.pdf"

        async with aiofiles.open(path, "wb") as destination_file:
            file_data = self.file.file.read() if isinstance(self.file, UploadFile) else self.file.read()
            await destination_file.write(file_data)

        return path

    # def start_convert(self) -> list[str]:
    #     self.pdf_path = self.save_pdf()
    #     images_path = self.convert_to_img()
    #
    #     os.remove(self.pdf_path)
    #     return images_path
    #
    # def save_pdf(self):
    #     self.pdf_path = f"{PDF_PATH}{self.filename}.pdf"
    #
    #     with open(self.pdf_path, "wb") as file:
    #         shutil.copyfileobj(self.file.file, file)
    #
    #     return self.pdf_path

    def convert_to_img(self) -> list[str]:
        os.makedirs(self.img_path, exist_ok=True)
        images_name = []
        images = convert_from_path(self.pdf_path)

        for i in range(len(images)):
            name = f"{str(i)}.jpg"
            path = self.img_path + name

            images[i].save(path, 'JPEG')
            images_name.append(path)

        return images_name
