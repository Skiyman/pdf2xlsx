# pdf2xlsx
Данная библиотека предназначена для извлечения таблиц из пдф файлов в xlsx

## Getting started

---
```bash
git clone git@github.com:Skiyman/pdf2xlsx.git
cd pdf2xlsx
python3 -m venv venv
source venv bin activate
pip install -r requrements.txt
```

**Дополнительно надо установить tesseract**

## Запуск в качестве веб-сервиса

---
Достаточно просто запустить ``app_entrypoint.py``

## Запуск в качестве api

---
```bash
docker compose -f docker-compose.yml up
rq worker task_queue
python3 api_entrypoint.py
```
