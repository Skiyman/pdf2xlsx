import uuid

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, HTTPException
from fastapi.responses import Response
from pydantic import TypeAdapter
from redis import Redis
from rq import Queue
from rq.job import Job

from api.models.models import ImagePredict
from src.models.table_detector import TableDetector
from src.preprocessing.image_preprocessor import Preprocessor
from src.shared.document_data_extractor import DocumentDataExtractor
from src.shared.pdf_converter import PdfConverter

app = FastAPI()

redis_connection = Redis(host="127.0.0.1", port=6379)
task_queue = Queue("task_queue", connection=redis_connection)

tasks = {}


def get_document_data(images, session_id):
    # Crop Pdf to images
    # tasks[session_id] = "Saving document"

    # Detecting tables on Images and crop them
    # tasks[session_id] = "Detecting Tables"
    table_detector = TableDetector(images, session_id)
    table_detector.make_predict()
    tables = table_detector.crop_images()

    ta = TypeAdapter(list[ImagePredict])
    files = ta.validate_python(tables)

    # Run preprocessing
    # tasks[session_id] = "Preprocessing Images"
    preprocessor = Preprocessor(session_id)
    preprocessor.fully_preprocess()

    # Extracting data in JSON format
    # tasks[session_id] = "Parsing Information"
    extractor = DocumentDataExtractor(
        files,
        session_id,
    )
    result = extractor.parse_data()

    return result


@app.post("/")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks) -> str:
    session_id = str(uuid.uuid4())
    converter = PdfConverter(file, session_id)
    images = await converter.start_convert()

    job = Job.create(
        get_document_data,
        args=[images, session_id],
        connection=redis_connection,
        id=session_id,
        timeout=1000
    )
    # session = task_queue.enqueue(get_document_data, file, session_id, job_id=session_id, timeout=500)
    task_queue.enqueue_job(job)
    # get_document_data.apply_async(args=(result, session_id), task_id=session_id)
    return session_id


@app.get("/{session_id}")
async def get_session_data(session_id: str):
    job = Job.fetch(session_id, connection=redis_connection)
    print(job.get_status())
    if not job.is_finished:
        raise HTTPException(status_code=202, detail="File in working")

    # Получение результата задачи
    content = job.return_value()
    if not content or not isinstance(content, bytes):
        raise HTTPException(status_code=500, detail="File content is invalid")

    # Возврат файла
    return Response(
        content=content,
        media_type="application/vnd.ms-excel",
        headers={
            "Content-Disposition": 'attachment; filename="result.xlsx"',
        },
    )


# app.include_router(converter_route)
# app.include_router(table_detector_route)
# app.include_router(extract_data_route)

if __name__ == "__main__":
    uvicorn.run('api_entrypoint:app', reload=True)
