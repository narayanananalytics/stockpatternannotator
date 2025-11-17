# 🚀 Deployment Guide: Push to Google Cloud Run

Complete step-by-step guide to deploy Stock Pattern Annotator to Google Cloud Run.

## ⚡ Quick Start (Interactive Script)

The easiest way to deploy:

```bash
cd webapp
./QUICKSTART.sh
```

This interactive script will guide you through:
1. ✅ Prerequisites check
2. ✅ Project setup
3. ✅ API enablement
4. ✅ Firestore initialization
5. ✅ Storage bucket creation
6. ✅ Container builds
7. ✅ Service deployment
8. ✅ Authentication setup

**Time: ~15-20 minutes**

---

## 📋 Manual Deployment (Step-by-Step)

If you prefer manual control, follow these detailed steps:

### Prerequisites

1. **Install gcloud CLI**

   ```bash
   # macOS
   brew install google-cloud-sdk

   # Linux
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL

   # Windows
   # Download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Verify Installation**

   ```bash
   gcloud --version
   # Should show: Google Cloud SDK 450.0.0 or higher
   ```

3. **Login to Google Cloud**

   ```bash
   gcloud auth login
   ```

### Step 1: Create GCP Project

```bash
# Set your project ID (must be globally unique)
export PROJECT_ID="stockpattern-prod-12345"

# Create project
gcloud projects create $PROJECT_ID --name="Stock Pattern Annotator"

# Set as active project
gcloud config set project $PROJECT_ID
```

### Step 2: Enable Billing

⚠️ **Required**: Cloud Run requires billing to be enabled.

```bash
# Open billing page
echo "Enable billing at:"
echo "https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"

# Or link existing billing account
gcloud beta billing accounts list
gcloud beta billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

### Step 3: Enable Required APIs

```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    iap.googleapis.com \
    --project=$PROJECT_ID
```

**Time: ~2-3 minutes**

### Step 4: Initialize Firestore

Firestore requires manual initialization:

1. **Visit Firestore Console**
   ```
   https://console.cloud.google.com/firestore/data?project=$PROJECT_ID
   ```

2. **Click "Create Database"**

3. **Select Settings:**
   - Mode: **Native mode**
   - Location: **us-central1** (or your region)
   - Security: **Production mode**

4. **Click "Create"**

**Time: ~1 minute**

### Step 5: Create Cloud Storage Bucket

```bash
export BUCKET_NAME="${PROJECT_ID}-stockpattern-results"
export REGION="us-central1"

# Create bucket
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Verify
gsutil ls -b gs://$BUCKET_NAME
```

### Step 6: Build Web Service Container

```bash
# Navigate to project root
cd /path/to/stockpatternannotator

# Build container image
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/stockpattern-web \
    --file webapp/Dockerfile.webapp \
    --project=$PROJECT_ID
```

**Time: ~5-10 minutes**

### Step 7: Deploy Web Service

```bash
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
```

**Output will include your service URL:**
```
Service [stockpattern-web] revision [stockpattern-web-00001-abc] has been deployed and is serving 100 percent of traffic.
Service URL: https://stockpattern-web-abc123-uc.a.run.app
```

**Time: ~2 minutes**

### Step 8: Build Worker Container

```bash
# Build worker image
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/stockpattern-worker \
    --file webapp/Dockerfile.worker \
    --project=$PROJECT_ID
```

**Time: ~5-10 minutes**

### Step 9: Configure Authentication

#### Option A: Add Specific Users (Recommended for Testing)

```bash
# Add yourself
gcloud run services add-iam-policy-binding stockpattern-web \
    --region=$REGION \
    --member='user:your.email@gmail.com' \
    --role='roles/run.invoker' \
    --project=$PROJECT_ID

# Add additional users
gcloud run services add-iam-policy-binding stockpattern-web \
    --region=$REGION \
    --member='user:colleague@company.com' \
    --role='roles/run.invoker' \
    --project=$PROJECT_ID
```

#### Option B: Enable Identity-Aware Proxy (Recommended for Production)

1. **Visit IAP Console:**
   ```
   https://console.cloud.google.com/security/iap?project=$PROJECT_ID
   ```

2. **Find your service** in the list

3. **Toggle IAP to ON**

4. **Add Principals:**
   - Click "Add Principal"
   - Enter email: `your.email@gmail.com`
   - Role: `IAP-secured Web App User`
   - Click "Save"

### Step 10: Test Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe stockpattern-web \
    --region $REGION \
    --format 'value(status.url)' \
    --project=$PROJECT_ID)

echo "Your app is live at: $SERVICE_URL"

# Open in browser
open $SERVICE_URL  # macOS
xdg-open $SERVICE_URL  # Linux
start $SERVICE_URL  # Windows
```

---

## 🔄 Update Deployment (After Code Changes)

When you make code changes and want to redeploy:

### Quick Update (Rebuild and Deploy)

```bash
# Set variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Rebuild and redeploy web service
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/stockpattern-web \
    --file webapp/Dockerfile.webapp \
    --project=$PROJECT_ID

# Deploy new version
gcloud run deploy stockpattern-web \
    --image gcr.io/$PROJECT_ID/stockpattern-web \
    --region $REGION \
    --project=$PROJECT_ID
```

### Update Worker Only

```bash
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/stockpattern-worker \
    --file webapp/Dockerfile.worker \
    --project=$PROJECT_ID
```

### Update Environment Variables

```bash
gcloud run services update stockpattern-web \
    --update-env-vars "NEW_VAR=value" \
    --region $REGION \
    --project=$PROJECT_ID
```

---

## 🔍 Monitoring and Debugging

### View Logs

```bash
# Real-time logs (web service)
gcloud run services logs tail stockpattern-web --region=$REGION

# Recent logs
gcloud run services logs read stockpattern-web \
    --region=$REGION \
    --limit=100

# Filter for errors
gcloud run services logs read stockpattern-web \
    --region=$REGION \
    --filter='severity>=ERROR' \
    --limit=50
```

### Check Service Status

```bash
# Service details
gcloud run services describe stockpattern-web \
    --region=$REGION \
    --project=$PROJECT_ID

# List all revisions
gcloud run revisions list \
    --service=stockpattern-web \
    --region=$REGION
```

### View in Cloud Console

```bash
echo "Cloud Run Console:"
echo "https://console.cloud.google.com/run?project=$PROJECT_ID"

echo "Firestore Console:"
echo "https://console.cloud.google.com/firestore?project=$PROJECT_ID"

echo "Storage Console:"
echo "https://console.cloud.google.com/storage?project=$PROJECT_ID"
```

---

## 🛠️ Troubleshooting

### Issue: "Permission denied" errors during deployment

**Solution:**
```bash
# Grant yourself owner role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:your.email@gmail.com" \
    --role="roles/owner"
```

### Issue: Build fails with "insufficient permissions"

**Solution:**
```bash
# Enable Cloud Build service account permissions
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin"
```

### Issue: "Firestore not initialized"

**Solution:** Firestore must be initialized manually (see Step 4 above)

### Issue: "Bucket already exists" (owned by another project)

**Solution:**
```bash
# Use a different bucket name
export BUCKET_NAME="${PROJECT_ID}-stockpattern-$(date +%s)"
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Update deployment
gcloud run services update stockpattern-web \
    --update-env-vars "GCS_BUCKET=$BUCKET_NAME" \
    --region $REGION
```

### Issue: Service is slow or timing out

**Solution:**
```bash
# Increase resources
gcloud run services update stockpattern-web \
    --memory 8Gi \
    --cpu 4 \
    --timeout 900 \
    --region $REGION
```

---

## 🗑️ Cleanup (Delete Everything)

To completely remove the deployment and avoid charges:

```bash
# Delete Cloud Run services
gcloud run services delete stockpattern-web --region=$REGION --quiet

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/stockpattern-web --quiet
gcloud container images delete gcr.io/$PROJECT_ID/stockpattern-worker --quiet

# Delete storage bucket
gsutil -m rm -r gs://$BUCKET_NAME

# Delete Firestore (manual)
echo "Delete Firestore at:"
echo "https://console.cloud.google.com/firestore?project=$PROJECT_ID"

# (Optional) Delete entire project
gcloud projects delete $PROJECT_ID
```

---

## 💰 Cost Management

### Monitor Costs

```bash
# View billing dashboard
echo "https://console.cloud.google.com/billing?project=$PROJECT_ID"
```

### Set Budget Alerts

```bash
# Create budget
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Stock Pattern Budget" \
    --budget-amount=50.00 \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90
```

### Minimize Costs

```bash
# Reduce max instances
gcloud run services update stockpattern-web \
    --max-instances=3 \
    --region=$REGION

# Reduce memory
gcloud run services update stockpattern-web \
    --memory=2Gi \
    --region=$REGION
```

---

## 📚 Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Firestore Documentation](https://cloud.google.com/firestore/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [IAP Documentation](https://cloud.google.com/iap/docs)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference)

---

## 🎯 Summary

**One-command deployment:**
```bash
cd webapp && ./QUICKSTART.sh
```

**Manual deployment checklist:**
- [ ] Create GCP project
- [ ] Enable billing
- [ ] Enable APIs
- [ ] Initialize Firestore
- [ ] Create storage bucket
- [ ] Build web container
- [ ] Deploy web service
- [ ] Build worker container
- [ ] Configure authentication
- [ ] Test deployment

**Need help?** Open an issue at: https://github.com/narayanananalytics/stockpatternannotator/issues
