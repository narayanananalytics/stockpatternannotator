#!/bin/bash
# Quick Start Deployment Guide for Stock Pattern Annotator on Cloud Run

echo "=================================================="
echo "Stock Pattern Annotator - Cloud Run Deployment"
echo "=================================================="
echo ""
echo "This guide will walk you through deploying to Google Cloud Run."
echo ""

# Prerequisites Check
echo "Step 1: Prerequisites Check"
echo "============================"
echo ""
echo "Required:"
echo "  ✓ Google Cloud account with billing enabled"
echo "  ✓ gcloud CLI installed"
echo "  ✓ Polygon.io API key (get free at polygon.io)"
echo ""
echo "Checking gcloud installation..."

if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo ""
    echo "Install gcloud CLI:"
    echo "  macOS:   brew install google-cloud-sdk"
    echo "  Linux:   https://cloud.google.com/sdk/docs/install"
    echo "  Windows: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✓ gcloud CLI found: $(gcloud version | head -n 1)"
echo ""

# Project Setup
echo "Step 2: Google Cloud Project Setup"
echo "==================================="
echo ""
echo "Do you have an existing GCP project? (y/n)"
read -p "> " has_project

if [ "$has_project" = "n" ] || [ "$has_project" = "N" ]; then
    echo ""
    echo "Enter a project ID (must be globally unique):"
    echo "Example: stockpattern-prod-12345"
    read -p "> " PROJECT_ID

    echo ""
    echo "Creating project: $PROJECT_ID"
    gcloud projects create $PROJECT_ID --name="Stock Pattern Annotator"

    echo ""
    echo "⚠️  IMPORTANT: Enable billing for this project"
    echo "Visit: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
    echo ""
    read -p "Press Enter when billing is enabled..."
else
    echo ""
    echo "Enter your existing project ID:"
    read -p "> " PROJECT_ID
fi

echo ""
echo "Setting active project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

echo ""
echo "✓ Project configured"
echo ""

# Enable APIs
echo "Step 3: Enable Required APIs"
echo "============================="
echo ""
echo "Enabling Cloud Run, Firestore, Storage, and other required APIs..."
echo "This may take 2-3 minutes..."
echo ""

gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    iap.googleapis.com \
    --project=$PROJECT_ID

echo ""
echo "✓ APIs enabled"
echo ""

# Firestore Setup
echo "Step 4: Firestore Database Setup"
echo "================================="
echo ""
echo "⚠️  Manual step required: Initialize Firestore"
echo ""
echo "1. Visit: https://console.cloud.google.com/firestore/data?project=$PROJECT_ID"
echo "2. Click 'Create Database'"
echo "3. Select 'Native mode'"
echo "4. Choose region: us-central1 (or your preferred region)"
echo "5. Click 'Create'"
echo ""
read -p "Press Enter when Firestore is initialized..."

echo ""
echo "✓ Firestore ready"
echo ""

# Storage Bucket
echo "Step 5: Create Storage Bucket"
echo "=============================="
echo ""

BUCKET_NAME="${PROJECT_ID}-stockpattern-results"
REGION="us-central1"

echo "Creating bucket: $BUCKET_NAME"

if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo "✓ Bucket already exists"
else
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
    echo "✓ Bucket created"
fi

echo ""

# Build and Deploy Web Service
echo "Step 6: Build and Deploy Web Service"
echo "====================================="
echo ""
echo "Building container image..."
echo "This may take 5-10 minutes..."
echo ""

cd "$(dirname "$0")/.."

gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/stockpattern-web \
    --file webapp/Dockerfile.webapp \
    --project=$PROJECT_ID

echo ""
echo "✓ Image built"
echo ""
echo "Deploying to Cloud Run..."

gcloud run deploy stockpattern-web \
    --image gcr.io/$PROJECT_ID/stockpattern-web \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCS_BUCKET=$BUCKET_NAME,CLOUD_RUN_REGION=$REGION" \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --project=$PROJECT_ID

echo ""
echo "✓ Web service deployed"
echo ""

# Build Worker Image
echo "Step 7: Build Worker Container"
echo "==============================="
echo ""
echo "Building worker container image..."
echo "This may take 5-10 minutes..."
echo ""

gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/stockpattern-worker \
    --file webapp/Dockerfile.worker \
    --project=$PROJECT_ID

echo ""
echo "✓ Worker image built"
echo ""

# Get Service URL
SERVICE_URL=$(gcloud run services describe stockpattern-web \
    --region $REGION \
    --format 'value(status.url)' \
    --project=$PROJECT_ID)

echo "Step 8: Configure Authentication"
echo "================================="
echo ""
echo "Adding you as authorized user..."
echo ""
echo "Enter your Google email address:"
read -p "> " USER_EMAIL

gcloud run services add-iam-policy-binding stockpattern-web \
    --region=$REGION \
    --member="user:$USER_EMAIL" \
    --role='roles/run.invoker' \
    --project=$PROJECT_ID

echo ""
echo "✓ User authorized"
echo ""

# Enable IAP
echo "Step 9: Enable Identity-Aware Proxy (IAP)"
echo "=========================================="
echo ""
echo "⚠️  Manual step required: Enable IAP for authentication"
echo ""
echo "1. Visit: https://console.cloud.google.com/security/iap?project=$PROJECT_ID"
echo "2. Find 'stockpattern-web' in the list"
echo "3. Toggle IAP to 'ON'"
echo "4. Add principals: $USER_EMAIL with role 'IAP-secured Web App User'"
echo ""
echo "Note: You can skip this for now and enable it later"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "=================================================="
echo "🎉 Deployment Complete!"
echo "=================================================="
echo ""
echo "Your application is live at:"
echo ""
echo "  🌐 $SERVICE_URL"
echo ""
echo "Next Steps:"
echo ""
echo "1. Visit the URL above"
echo "2. Log in with your Google Account ($USER_EMAIL)"
echo "3. Get a free Polygon.io API key at https://polygon.io"
echo "4. Create your first training job!"
echo ""
echo "Resources:"
echo "  - Web Service: https://console.cloud.google.com/run?project=$PROJECT_ID"
echo "  - Firestore: https://console.cloud.google.com/firestore?project=$PROJECT_ID"
echo "  - Storage: https://console.cloud.google.com/storage/browser?project=$PROJECT_ID"
echo "  - Logs: gcloud run services logs tail stockpattern-web --region=$REGION"
echo ""
echo "Cost Estimate: ~\$5-30/month (includes \$300 free credit for new users)"
echo ""
echo "Documentation: webapp/README.md"
echo ""
