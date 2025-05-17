FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py /app/main.py
COPY invoke.py /app/invoke.py
COPY config.json /app/config.json
COPY outputs /app/outputs

RUN apt-get clean && rm -rf /root/.cache /var/cache/apt

ENV MODEL_PATH=/app/outputs/best_lightgbm_model.pkl
ENV SCALER_PATH=/app/outputs/scaler_lgb.pkl

CMD ["python", "main.py"]
