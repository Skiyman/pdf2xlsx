from fastapi import APIRouter
from pydantic import parse_obj_as

from api.models.models import ImagePredict
from src.models.table_detector import TableDetector
from src.preprocessing.image_preprocessor import Preprocessor

table_detector_route = APIRouter(
    prefix="/table_detector",
    tags=["Table Detector"]
)


@table_detector_route.post("/{session_id}")
async def detect_table(session_id: str, make_preprocessing: bool, image_paths: list[str]) -> list[ImagePredict]:
    detector = TableDetector(image_paths, session_id)
    detector.make_predict()
    files = detector.crop_images()

    if make_preprocessing:
        preprocessor = Preprocessor(session_id)
        preprocessor.fully_preprocess()

    response = parse_obj_as(list[ImagePredict], files)

    return response
