# Bat Monitoring Backend API

Backend API for the Bat Monitoring System deployed on Azure App Service.

## Features

- Flask REST API for bat species classification
- Google Drive integration for audio file management
- ML-powered bat species prediction using EfficientNet
- Spectrogram generation and analysis
- Audio call parameter extraction

## Tech Stack

- **Framework**: Flask 2.3.3
- **ML Models**: PyTorch + EfficientNet
- **Audio Processing**: Librosa, SciPy
- **Cloud Storage**: Google Drive API
- **Deployment**: Azure App Service (Linux)

## Azure Deployment

This application is configured for Azure App Service deployment with:
- Python 3.11 runtime (`runtime.txt`)
- Gunicorn WSGI server (`startup.sh`)
- Automated dependency installation
- Environment variable support for credentials

### Required Environment Variables

Configure these in Azure App Service Configuration:

```
CLIENT_SECRETS_JSON=<your-google-client-secrets-json>
CREDENTIALS_JSON=<your-google-credentials-json>
FLASK_ENV=production
PORT=8000
```

### Deployment Methods

#### Option 1: GitHub Actions (Recommended)
1. Set up Azure App Service
2. Configure deployment credentials
3. Add secrets to GitHub repository:
   - `AZURE_WEBAPP_PUBLISH_PROFILE`
4. Push to main branch - auto-deploys via `azure-app-service.yml`

#### Option 2: Azure CLI
```bash
az webapp up --name <your-app-name> --resource-group <your-rg> --runtime PYTHON:3.11
```

#### Option 3: Git Deployment
```bash
# Add Azure remote
git remote add azure <azure-git-url>

# Deploy
git push azure main
```

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/NikhilBakale/wifibackend.git
cd wifibackend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Google Drive credentials:
   - Create a project in Google Cloud Console
   - Enable Google Drive API
   - Create OAuth 2.0 credentials
   - Download and save as `client_secrets.json`

5. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Docker Deployment

### 1. Create Environment File

Copy `.env.example` to `.env` and set real values:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Important:
- Keep `CLIENT_SECRETS_JSON` and `CREDENTIALS_JSON` as one-line JSON strings.
- Do not commit `.env`.

### 2. Build Docker Image

```bash
docker build -t bat-backend:latest .
```

### 3. Run Container

```bash
docker run --env-file .env -p 8000:8000 --name bat-backend bat-backend:latest
```

The API will be available at `http://localhost:8000`.

### 4. Optional Runtime Tuning

You can tune Gunicorn workers/threads/timeouts in `.env`:

```env
GUNICORN_WORKERS=2
GUNICORN_THREADS=4
GUNICORN_TIMEOUT=600
```

### 5. Health Check

```bash
curl http://localhost:8000/api/health
```

## API Endpoints

### Health Check
- `GET /` - Returns API status

### Google Drive Operations
- `GET /folders` - List all folders
- `GET /folder/<folder_id>` - Get folder details and files
- `GET /download/<file_id>` - Download file from Google Drive
- `POST /predict-folder/<folder_id>` - Predict bat species for all audio files in folder

### Standalone Predictions
- `POST /predict-standalone` - Upload and predict single audio file

### Configuration
- `GET /config` - Get ML model availability status

## Project Structure

```
bat-admin-backend-main/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── startup.sh                      # Azure startup script
├── runtime.txt                     # Python version specification
├── Procfile                        # Process file for deployment
├── .deployment                     # Azure deployment config
├── settings.yaml                   # Google Drive settings
├── models/                         # ML models and prediction code
│   ├── predict.py
│   ├── bat_28.pth
│   └── classes_28.json
├── static/                         # Static files (generated)
│   ├── spectrograms/
│   └── audio/
├── spectrogram_analyzer.py         # Audio analysis utilities
└── guano_metadata_extractor.py     # Metadata extraction

```

## Model Information

The application uses EfficientNet-B0 trained on bat call spectrograms for species classification.

## CORS Configuration

Configured to accept requests from:
- Local development (localhost:5173, 5174, 5175, 3000)
- Vercel deployments (*.vercel.app)
- Azure Static Web Apps (*.azurestaticapps.net)

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.
