# syntax=docker/dockerfile:1
#
# Streamlit dashboard image. Thin client that calls the API over HTTP, so it only
# needs streamlit + requests, not the full RAG stack.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN pip install --no-cache-dir "streamlit>=1.36" "requests>=2.31"

COPY frontend/ /app/frontend/

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
