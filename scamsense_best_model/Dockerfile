# ScamSense Telegram Bot — Cloud Run Deployment
# Build: docker build -t scamsense-bot .
# Run locally: docker run -p 8080:8080 --env-file .env scamsense-bot

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run expects the app to listen on PORT (default 8080)
ENV PORT=8080

# Start the FastAPI webhook server
CMD ["uvicorn", "bot.webhook:app", "--host", "0.0.0.0", "--port", "8080"]