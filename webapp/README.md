# Stock Pattern Annotator - Cloud Run Web Application

A professional web application for running RL trading simulations on Google Cloud Run, with Google Account authentication and a modern UI.

## 🌟 Features

- **Web UI**: Clean, responsive interface built with Tailwind CSS
- **Google Authentication**: Protected by Google Cloud Identity-Aware Proxy (IAP)
- **Async Job Processing**: Long-running training jobs via Cloud Run Jobs
- **Real-time Progress**: Live progress updates using HTMX and Firestore
- **Results Visualization**: Interactive charts and performance metrics
- **Cloud Storage**: Automatic storage of models and results in GCS
- **Serverless**: Auto-scaling, pay-per-use infrastructure

## 📋 Prerequisites

- **Google Cloud Platform account** with billing enabled
- **gcloud CLI** installed ([install guide](https://cloud.google.com/sdk/docs/install))
- **Polygon.io API key** ([free tier available](https://polygon.io))

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/narayanananalytics/stockpatternannotator.git
cd stockpatternannotator
```

### 2. Set Up GCP Project

```bash
# Create a new project (or use existing)
gcloud projects create my-stockpattern-project
gcloud config set project my-stockpattern-project

# Enable billing (required)
# Visit: https://console.cloud.google.com/billing
```

### 3. Deploy to Cloud Run

```bash
cd webapp
chmod +x deploy.sh
./deploy.sh
```

The script will:
- ✅ Enable required GCP APIs
- ✅ Create Cloud Storage bucket
- ✅ Build and deploy web service
- ✅ Build worker container image
- ✅ Output application URL

### 4. Enable Authentication

```bash
# Enable Identity-Aware Proxy
# Visit: https://console.cloud.google.com/security/iap?project=YOUR_PROJECT_ID

# Add authorized users
gcloud run services add-iam-policy-binding stockpattern-web \
  --region=us-central1 \
  --member='user:your.email@gmail.com' \
  --role='roles/run.invoker'
```

### 5. Initialize Firestore

First-time setup requires manual Firestore initialization:

1. Visit [Firestore Console](https://console.cloud.google.com/firestore)
2. Click "Create Database"
3. Select "Native mode"
4. Choose region (use same as Cloud Run: `us-central1`)
5. Start in production mode

### 6. Access Application

Visit the URL provided by the deployment script (e.g., `https://stockpattern-web-xxx.a.run.app`)

## 🏗️ Architecture

```
┌──────────────┐
│   Browser    │
│  (User UI)   │
└──────┬───────┘
       │ HTTPS (IAP Protected)
       ↓
┌──────────────────────┐
│  Cloud Run Service   │
│  (FastAPI Web App)   │
│  - Job submission    │
│  - Status monitoring │
│  - Results display   │
└──────┬───────────────┘
       │
       ├─────────→ ┌────────────────┐
       │           │   Firestore    │
       │           │  (Job state)   │
       │           └────────────────┘
       │
       ├─────────→ ┌────────────────┐
       │           │ Cloud Storage  │
       │           │(Models/Results)│
       │           └────────────────┘
       │
       ↓
┌────────────────────────┐
│  Cloud Run Jobs        │
│  (Worker Containers)   │
│  - Fetch data          │
│  - Train RL agent      │
│  - Save results        │
└────────────────────────┘
```

## 📝 Usage

### Creating a Training Job

1. **Log in** with your Google Account (via IAP)
2. **Fill out the form**:
   - Polygon.io API Key
   - Symbol (e.g., AAPL, TSLA)
   - Timeframe (1M, 3M, 5M, 1H, 1D)
   - Start and End dates
   - Training timesteps (50K - 500K)
3. **Click "Start Training"**
4. **Monitor progress** in real-time
5. **View results** when complete

### Viewing Results

Completed jobs show:
- **Performance metrics**: Total return, win rate, Sharpe ratio
- **Backtest details**: Trade-by-trade analysis
- **Download options**: Trained model (.zip) and results (.csv)

## 💰 Cost Estimation

Based on typical usage:

| Component | Usage | Monthly Cost |
|-----------|-------|--------------|
| Cloud Run (Web) | 100 requests/day, 2 vCPU, 4GB RAM | ~$5 |
| Cloud Run Jobs | 10 jobs/day @ 30 min each, 4 vCPU, 16GB RAM | ~$15 |
| Firestore | 10K reads, 1K writes/day | ~$0.50 |
| Cloud Storage | 10 GB storage, 100 GB egress | ~$2 |
| **Total** | **Light usage** | **~$22.50/month** |

**Free Tier**: New GCP users get $300 credit for 90 days.

**Cost Optimization Tips**:
- Use CPU instead of GPU (GPU: ~$2-5/hour)
- Set shorter training timesteps for testing
- Delete old models/results from Cloud Storage
- Use Cloud Run's "min instances: 0" for zero cost when idle

## ⚙️ Configuration

### Environment Variables

Set in Cloud Run service:

```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GCS_BUCKET=your-bucket-name
CLOUD_RUN_REGION=us-central1
DEV_USER_EMAIL=dev@example.com  # For local development
```

### Compute Resources

Edit in `app.yaml`:

```yaml
resources:
  cpu: 2              # 1-8 vCPUs
  memory: 4Gi         # 512Mi - 32Gi

automatic_scaling:
  min_instances: 0    # 0 for cost savings
  max_instances: 10   # Limit concurrent users
```

### Worker Resources

Modify in `jobs.py`:

```python
"resources": {
    "limits": {
        "cpu": "4",      # More CPU = faster training
        "memory": "16Gi" # More memory = larger datasets
    }
}
```

## 🔒 Security

### Authentication

**Identity-Aware Proxy (IAP)** protects the entire application:

- Only users with Google Accounts can access
- Add/remove users in Cloud Console
- No code changes needed
- Integrated with Google Workspace

### Authorization

Each user can only:
- View their own jobs
- Download their own results
- Submit new jobs (rate limited by Cloud Run)

### API Keys

Polygon.io API keys are:
- Entered per-job (not stored permanently)
- Passed securely to worker containers
- Not logged or persisted in Firestore

**Best Practice**: Use Secret Manager for production:

```bash
# Store API key in Secret Manager
gcloud secrets create polygon-api-key --data-file=- <<< "your-key-here"

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding polygon-api-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 🐛 Troubleshooting

### "Permission denied" when accessing app

**Solution**: Add your email to IAP authorized users

```bash
gcloud run services add-iam-policy-binding stockpattern-web \
  --region=us-central1 \
  --member='user:your.email@gmail.com' \
  --role='roles/run.invoker'
```

### "Job stuck in pending"

**Check**:
1. Cloud Run Jobs API is enabled
2. Worker image exists: `gcloud container images list --repository=gcr.io/YOUR_PROJECT_ID`
3. Service account has permissions: `run.jobs.create`

### "No data returned from Polygon.io"

**Verify**:
- API key is valid
- Symbol exists (try AAPL first)
- Date range is valid (not weekends/holidays for stocks)
- Polygon.io subscription supports requested timeframe

### Job fails immediately

**Check logs**:

```bash
# Web service logs
gcloud run services logs read stockpattern-web --region=us-central1 --limit=50

# Job execution logs
gcloud run jobs executions logs read EXECUTION_NAME --region=us-central1
```

## 🔧 Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt -r webapp/requirements.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT=your-project-id
export GCS_BUCKET=your-bucket-name
export DEV_USER_EMAIL=dev@example.com

# Run locally
cd webapp
uvicorn main:app --reload --port 8080

# Visit http://localhost:8080
```

### Testing Worker Locally

```bash
# Set job environment variables
export JOB_ID=test-job-123
export POLYGON_API_KEY=your-key
export SYMBOL=AAPL
export TIMEFRAME=1D
export START_DATE=2023-01-01
export END_DATE=2024-01-01
export TOTAL_TIMESTEPS=50000

# Run worker
python -m webapp.worker
```

### Manual Deployment

```bash
# Build web service
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/stockpattern-web -f webapp/Dockerfile.webapp

# Deploy
gcloud run deploy stockpattern-web \
  --image gcr.io/YOUR_PROJECT_ID/stockpattern-web \
  --region us-central1 \
  --platform managed

# Build worker
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/stockpattern-worker -f webapp/Dockerfile.worker
```

## 📊 Monitoring

### Metrics

View in Cloud Console:
- Request count and latency (web service)
- Job execution times (worker)
- Error rates
- Resource utilization

### Logging

```bash
# Web service logs
gcloud run services logs tail stockpattern-web --region=us-central1

# Filter for errors
gcloud run services logs read stockpattern-web \
  --region=us-central1 \
  --filter='severity>=ERROR' \
  --limit=100
```

### Alerts

Set up alerts in Cloud Monitoring:
- Job failures
- High error rates
- Cost anomalies

## 🚀 Roadmap

- [ ] Add GPU support via Vertex AI
- [ ] Hyperparameter tuning UI
- [ ] Model comparison charts
- [ ] Real-time paper trading mode
- [ ] Multi-symbol backtesting
- [ ] Export to TradingView
- [ ] Webhook notifications
- [ ] Team workspaces

## 📄 License

MIT License - see LICENSE file

## 🤝 Support

- GitHub Issues: https://github.com/narayanananalytics/stockpatternannotator/issues
- Cloud Run Documentation: https://cloud.google.com/run/docs
- Firestore Documentation: https://cloud.google.com/firestore/docs

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [HTMX](https://htmx.org/) - Dynamic HTML without JavaScript complexity
- [Google Cloud Run](https://cloud.google.com/run) - Serverless container platform
- [Firestore](https://cloud.google.com/firestore) - NoSQL document database
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) - RL library
