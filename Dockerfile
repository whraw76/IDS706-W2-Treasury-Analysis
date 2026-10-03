FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY analysis.py compare_polars.py pytest.ini ./
COPY data/ ./data/
COPY tests/ ./tests/

CMD ["python", "analysis.py"]
