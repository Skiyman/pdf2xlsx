import uuid

from fastapi import APIRouter, UploadFile

from api.models.models import SavePdfResponse
from src.shared.pdf_converter import PdfConverter

converter_route = APIRouter(
    prefix="/converter",
    tags=["Pdf converter"]
)


@converter_route.post("")
async def save_pdf(file: UploadFile) -> SavePdfResponse:
    session_id = str(uuid.uuid4())
    pdf_converter = PdfConverter(file, session_id)
    paths = await pdf_converter.start_convert()

    response = SavePdfResponse(
        session_id=session_id,
        paths=paths
    )

    return response
