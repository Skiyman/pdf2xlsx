from pydantic import BaseModel


class SavePdfResponse(BaseModel):
    session_id: str
    paths: list[str]


class ImagePredictDataItem(BaseModel):
    path: str
    table_class: str
    confidence: float


class ImagePredict(BaseModel):
    path: str
    data: list[ImagePredictDataItem]


class DetectTableResponse(BaseModel):
    data: list[ImagePredict]
