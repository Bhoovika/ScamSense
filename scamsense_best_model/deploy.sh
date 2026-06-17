#!/bin/bash
# deploy.sh — run from repo root after installing gcloud CLI
# Usage: bash deploy.sh

PROJECT_ID="your-gcp-project-id"      # replace
REGION="asia-southeast1"               # Singapore
SERVICE="scamsense-bot"

gcloud config set project $PROJECT_ID

# Build and deploy
gcloud run deploy $SERVICE \
  --source . \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets="TELEGRAM_TOKEN=telegram-token:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest"

echo "Deployed! Set Telegram webhook:"
echo "https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook?url=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')"