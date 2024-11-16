import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models.models import ImagePredict
from src.consts.path_consts import JSON_PATH
from src.shared.document_data_extractor import DocumentDataExtractor

extract_data_route = APIRouter(
    prefix="/extract_data",
    tags=["Data Extraction"]
)


@extract_data_route.post("/{session_id}")
async def table_data_extract(session_id: str, table_data: list[ImagePredict]) -> str:
    extractor = DocumentDataExtractor(
        table_data,
        session_id,
    )
    json_path = extractor.parse_data()

    return json_path


@extract_data_route.get("/{session_id}")
async def get_json(session_id: str) -> FileResponse:
    json_path = JSON_PATH + session_id + ".json"
    if os.path.isfile(json_path):
        response = FileResponse(json_path, filename="result.json", media_type="application/json")
        return response

    raise HTTPException(status_code=404, detail="File not found")
