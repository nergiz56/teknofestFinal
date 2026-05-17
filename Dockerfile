FROM python:3.11-slim

WORKDIR /app

# Sorun çıkaran apt-get komutlarını temizledik.
# OpenCV'nin Linux'ta kütüphanesiz çalışabilen headless sürümünü requirements ile çözeceğiz.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models/heatmaps

ENV PYTHONPATH=/app

CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
