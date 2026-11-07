#!/usr/bin/env bash
# Deploy DocAgent to Google Cloud Run.
#
# Prerequisites: gcloud CLI logged in (`gcloud auth login`), a project with billing
# (`gcloud config set project <id>`), Docker running, the index built
# (python -m docagent.ingest.build), and ANTHROPIC_API_KEY + DOCAGENT_API_KEY in the
# environment. Secrets go to Secret Manager; they are never baked into the image.
set -euo pipefail

REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-docagent}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
REPO="${REGION}-docker.pkg.dev/${PROJECT}/docagent"
IMAGE="${REPO}/docagent:$(date +%Y%m%d-%H%M)"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
: "${DOCAGENT_API_KEY:?set DOCAGENT_API_KEY (the key clients send as X-API-Key)}"

echo "project=${PROJECT} region=${REGION} image=${IMAGE}"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

put_secret () {   # name value
  if gcloud secrets describe "$1" >/dev/null 2>&1; then
    printf '%s' "$2" | gcloud secrets versions add "$1" --data-file=-
  else
    printf '%s' "$2" | gcloud secrets create "$1" --replication-policy=automatic --data-file=-
  fi
}
put_secret anthropic-api-key "${ANTHROPIC_API_KEY}"
put_secret docagent-api-key "${DOCAGENT_API_KEY}"

SA="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for s in anthropic-api-key docagent-api-key; do
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:${SA}" \
    --role=roles/secretmanager.secretAccessor >/dev/null
done

gcloud artifacts repositories describe docagent --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create docagent --repository-format=docker --location="${REGION}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker build -t "${IMAGE}" .
docker push "${IMAGE}"

# Scale to zero when idle; cap instances so a public endpoint cannot run up costs.
gcloud run deploy "${SERVICE}" --image "${IMAGE}" --region "${REGION}" \
  --allow-unauthenticated --memory 4Gi --cpu 2 --timeout 300 --concurrency 4 \
  --min-instances 0 --max-instances 2 --cpu-boost \
  --set-env-vars "DOCAGENT_PUBLIC=1,DOCAGENT_RATE_PER_MIN=6" \
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,DOCAGENT_API_KEY=docagent-api-key:latest"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
echo "Deployed: ${URL}"
curl -fsS "${URL}/health" && echo
