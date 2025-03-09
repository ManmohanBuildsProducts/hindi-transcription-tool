# Hindi Audio Transcription Tool

A web application for capturing and transcribing Hindi speech from both microphone and system audio sources, using Sarvam AI's transcription API.

## Features

- Record audio from microphone
- Capture system audio (from screen sharing)
- Combine microphone and system audio simultaneously
- Process long recordings (1-2 hours) by chunking into 8-minute segments
- Display Hindi transcriptions with proper font support
- Track progress for long recordings
- Copy and download transcriptions

## Technology Stack

- **Frontend**: React with Tailwind CSS
- **Backend**: FastAPI (Python)
- **Database**: MongoDB (optional)
- **API Integration**: Sarvam AI Batch API for Hindi transcription

## Setup Instructions

### Prerequisites

- Python 3.8+ 
- Node.js 14+
- MongoDB (optional)
- ffmpeg (required for audio processing)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with:
   ```
   SARVAM_API_KEY=ec7650e8-3560-48c7-8c69-649f1c659680
   SARVAM_API_URL=https://api.sarvam.ai/v1/transcribe/batch
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=hindi_transcription
   ```

5. Start the server:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8001
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file with:
   ```
   REACT_APP_BACKEND_URL=http://localhost:8001
   ```

4. Start the development server:
   ```bash
   npm start
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Deployment

### Deploying to Render.com

#### Backend Service

1. Create a new Web Service on Render.com
2. Connect to this GitHub repository
3. Configure as follows:
   - **Name**: hindi-transcription-api
   - **Root Directory**: backend
   - **Runtime**: Python 3
   - **Build Command**: `apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or as needed)
   - **Advanced** → **Environment Variables**:
     - `SARVAM_API_KEY`: Your Sarvam API key
     - `SARVAM_API_URL`: https://api.sarvam.ai/v1/transcribe/batch
     - `MONGO_URL`: Your MongoDB connection string (from MongoDB Atlas)
     - `DB_NAME`: hindi_transcription

#### Frontend Service

1. Create another Web Service on Render.com
2. Connect to the same GitHub repository
3. Configure as follows:
   - **Name**: hindi-transcription-frontend
   - **Root Directory**: frontend
   - **Runtime**: Node
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npx serve -s build`
   - **Plan**: Free (or as needed)
   - **Advanced** → **Environment Variables**:
     - `REACT_APP_BACKEND_URL`: Your backend API URL (from the previous step)

## Usage

1. Open the application in your browser
2. Choose your audio source:
   - **Microphone**: Captures your voice
   - **System Audio**: Captures audio playing on your computer (requires screen sharing)
   - **Combined**: Captures both your voice and system audio
3. Click "Start Recording" and speak in Hindi
4. Click "Stop Recording" when done
5. Wait for transcription to complete
6. View, copy, or download your Hindi transcription

## Browser Compatibility

- **Chrome/Edge**: Full support for all features including system audio capture
- **Firefox**: Supports microphone recording; system audio capture has limitations
- **Safari**: Basic microphone recording support

## Notes

- System audio capture requires HTTPS or localhost
- For long recordings (1-2 hours), processing may take several minutes
- The default in-memory storage will lose data on server restart; use MongoDB for persistence

## License

MIT
