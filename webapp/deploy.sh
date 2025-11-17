#!/bin/bash
# Deploy Stock Pattern Annotator to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================  ${NC}"
echo -e "${GREEN}Stock Pattern Annotator - Cloud Run Deploy${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: No GCP project set${NC}"
    echo "Set project with: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✓ Project:${NC} $PROJECT_ID"
echo ""

# Configuration
REGION=${CLOUD_RUN_REGION:-us-central1}
WEB_SERVICE_NAME="stockpattern-web"
WORKER_JOB_NAME="stockpattern-worker"
BUCKET_NAME="${PROJECT_ID}-stockpattern-results"

echo "Configuration:"
echo "  Region: $REGION"
echo "  Web Service: $WEB_SERVICE_NAME"
echo "  Worker Job: $WORKER_JOB_NAME"
echo "  GCS Bucket: $BUCKET_NAME"
echo ""

# Step 1: Enable required APIs
echo -e "${YELLOW}Step 1: Enabling required APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    iap.googleapis.com \
    --project=$PROJECT_ID

echo -e "${GREEN}✓ APIs enabled${NC}"
echo ""

# Step 2: Create GCS bucket if it doesn't exist
echo -e "${YELLOW}Step 2: Creating Cloud Storage bucket...${NC}"
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
    echo -e "${GREEN}✓ Bucket created: $BUCKET_NAME${NC}"
else
    echo -e "${GREEN}✓ Bucket already exists: $BUCKET_NAME${NC}"
fi
echo ""

# Step 3: Create Firestore database (if needed)
echo -e "${YELLOW}Step 3: Checking Firestore database...${NC}"
echo "Note: If Firestore is not initialized, create it manually at:"
echo "https://console.cloud.google.com/firestore/data?project=$PROJECT_ID"
echo ""

# Step 4: Build and deploy web service
echo -e "${YELLOW}Step 4: Building and deploying web service...${NC}"
cd "$(dirname "$0")/.."  # Go to project root

gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$WEB_SERVICE_NAME \
    --file webapp/Dockerfile.webapp \
    --project=$PROJECT_ID

gcloud run deploy $WEB_SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$WEB_SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET=$BUCKET_NAME,CLOUD_RUN_REGION=$REGION" \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --project=$PROJECT_ID

echo -e "${GREEN}✓ Web service deployed${NC}"
echo ""

# Step 5: Build and create worker job
echo -e "${YELLOW}Step 5: Building worker container...${NC}"

gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$WORKER_JOB_NAME \
    --file webapp/Dockerfile.worker \
    --project=$PROJECT_ID

echo -e "${GREEN}✓ Worker image built${NC}"
echo "Note: Worker jobs are created dynamically by the web service"
echo ""

# Step 6: Get service URL
SERVICE_URL=$(gcloud run services describe $WEB_SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)' \
    --project=$PROJECT_ID)

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Web Application URL:"
echo -e "${GREEN}$SERVICE_URL${NC}"
echo ""
echo "Next Steps:"
echo "1. Enable Identity-Aware Proxy (IAP) for authentication:"
echo "   https://console.cloud.google.com/security/iap?project=$PROJECT_ID"
echo ""
echo "2. Add authorized users in IAP settings"
echo ""
echo "3. Grant Cloud Run Invoker role to authorized users:"
echo "   gcloud run services add-iam-policy-binding $WEB_SERVICE_NAME \\"
echo "     --region=$REGION \\"
echo "     --member='user:YOUR_EMAIL@gmail.com' \\"
echo "     --role='roles/run.invoker'"
echo ""
echo "4. Visit the application:"
echo "   $SERVICE_URL"
echo ""
