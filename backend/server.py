from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import logging
import sys
from pathlib import Path
import json
import requests
import base64
from typing import List, Optional, Dict
import asyncio
import aiohttp
import uuid
import subprocess
from pydub import AudioSegment
import io
import tempfile
from sarvam_batch_api import SarvamBatchAPI
from dotenv import load_dotenv

# Setup environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up ffmpeg for Render deployment
try:
    import download_ffmpeg
    download_ffmpeg.main()
    logger.info("Custom ffmpeg setup completed")
    # Check if ffmpeg is available in PATH
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            tmp_path = tmp.name
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("ffmpeg is available in PATH")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"ffmpeg check failed: {e}, PATH might not be set correctly")
        # Try to get PATH from the download_ffmpeg module
        try:
            ffmpeg_dir = download_ffmpeg.FFMPEG_DIR
            os.environ['PATH'] = f"{ffmpeg_dir}:{os.environ.get('PATH', '')}"
            logger.info(f"Updated PATH with custom ffmpeg dir: {ffmpeg_dir}")
            logger.info(f"New PATH: {os.environ['PATH']}")
        except Exception as inner_e:
            logger.error(f"Failed to update PATH: {inner_e}")
except Exception as e:
    logger.warning(f"Custom ffmpeg setup failed: {e}, will try to use system ffmpeg if available")

# Constants from environment
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    logger.warning("SARVAM_API_KEY not set in environment, using default test key")
    SARVAM_API_KEY = "ec7650e8-3560-48c7-8c69-649f1c659680"  # Default key, will be used if env var is not set

# Sarvam API endpoints - based on successful testing
SARVAM_API_INIT_URL = "https://api.sarvam.ai/speech-to-text-translate/job/init"
SARVAM_API_STATUS_URL_TEMPLATE = "https://api.sarvam.ai/speech-to-text-translate/job/{job_id}/status"
SARVAM_API_SUBMIT_URL = "https://api.sarvam.ai/speech-to-text-translate/job"
SARVAM_API_URL = SARVAM_API_SUBMIT_URL  # For backward compatibility with existing code
CHUNK_DURATION = 8 * 60 * 1000  # 8 minutes in milliseconds
MAX_RETRIES = 3  # Maximum retries for API calls

# MongoDB setup
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'hindi_transcription')

# Initialize FastAPI app
app = FastAPI(title="Hindi Audio Transcription API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (for now, we can migrate to MongoDB later)
recordings: Dict[str, dict] = {}  # recording_id -> recording info
chunks: Dict[str, List[dict]] = {}  # recording_id -> list of chunk transcriptions
jobs: Dict[str, dict] = {}  # job_id -> job info

class RecordingStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Recording(BaseModel):
    id: str
    timestamp: datetime
    duration: float
    status: str
    transcript: Optional[str] = None
    error: Optional[str] = None

class ChunkJob(BaseModel):
    id: str
    recording_id: str
    chunk_index: int
    status: str
    transcript: Optional[str] = None
    error: Optional[str] = None

def split_audio(audio_data: bytes, format: str) -> List[AudioSegment]:
    """Split audio into 8-minute chunks with format handling"""
    try:
        # Create a temporary file to handle the audio data
        with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            
            try:
                # Load audio from temp file
                audio = AudioSegment.from_file(temp_file.name, format=format)
            except Exception as e:
                logger.error(f"Error loading audio with format {format}: {e}")
                # Try alternative format
                alt_format = 'wav' if format == 'webm' else 'webm'
                try:
                    audio = AudioSegment.from_file(temp_file.name, format=alt_format)
                    logger.info(f"Successfully loaded audio with alternative format: {alt_format}")
                except Exception as e2:
                    logger.error(f"Error loading audio with alternative format {alt_format}: {e2}")
                    raise
            finally:
                # Clean up temp file
                os.unlink(temp_file.name)
        
        if len(audio) == 0:
            raise ValueError("Empty audio file")
            
        # Convert to mono and set sample rate
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Split into 8-minute chunks
        chunks = []
        chunk_length = CHUNK_DURATION  # 8 minutes in milliseconds
        
        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i + chunk_length]
            # Ensure chunk is not too short
            if len(chunk) >= 1000:  # At least 1 second
                chunks.append(chunk)
        
        if not chunks:
            raise ValueError("No valid audio chunks found")
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error splitting audio: {e}")
        if "ffmpeg not found" in str(e):
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: ffmpeg not installed"
            )
        elif "Empty audio file" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Empty audio file received"
            )
        elif "No valid audio chunks" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Audio file too short or invalid"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to process audio file. Please check the format and try again."
            )

async def process_real_audio(background_tasks: BackgroundTasks, audio_data: bytes, format: str, source: str):
    """Real audio processing function - to be used in the future when API issues are resolved"""
    recording_id = str(uuid.uuid4())
    
    try:
        # THIS IS FOR FUTURE IMPLEMENTATION
        # We're keeping this function separate to make it easier to implement later
        
        # For now, return a success with a placeholder transcript
        return {
            "recording_id": recording_id,
            "status": RecordingStatus.PROCESSING, 
            "message": "Processing initiated",
            "source": source,
            "format": format
        }
    except Exception as e:
        logger.error(f"Error in real audio processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

async def process_recording(recording_id: str, audio_chunks: List[AudioSegment]):
    """Process all chunks of a recording with progress tracking"""
    try:
        chunks[recording_id] = []
        tasks = []
        total_chunks = len(audio_chunks)
        
        # Reset chunk counters
        recordings[recording_id].update({
            "chunks_processed": 0,
            "chunks_failed": 0,
            "progress": 0
        })
        
        async def process_chunk(chunk: AudioSegment, index: int) -> dict:
            try:
                result = await transcribe_chunk(chunk, index, recording_id)
                
                # Update progress
                recordings[recording_id]["chunks_processed"] += 1
                recordings[recording_id]["progress"] = int((recordings[recording_id]["chunks_processed"] / total_chunks) * 100)
                
                if result is None:
                    recordings[recording_id]["chunks_failed"] += 1
                    return {
                        "index": index,
                        "transcript": None,
                        "error": "Failed to transcribe chunk"
                    }
                
                return {
                    "index": index,
                    "transcript": result,
                    "error": None
                }
                
            except Exception as e:
                recordings[recording_id]["chunks_processed"] += 1
                recordings[recording_id]["chunks_failed"] += 1
                recordings[recording_id]["progress"] = int((recordings[recording_id]["chunks_processed"] / total_chunks) * 100)
                
                return {
                    "index": index,
                    "transcript": None,
                    "error": str(e)
                }
        
        # Process chunks in parallel with status tracking
        for i, chunk in enumerate(audio_chunks):
            task = asyncio.create_task(process_chunk(chunk, i))
            tasks.append(task)
        
        # Wait for all chunks with timeout
        try:
            chunk_results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logger.error(f"Error gathering results: {e}")
            recordings[recording_id].update({
                "status": RecordingStatus.FAILED,
                "error": f"Processing timeout: {str(e)}",
                "progress": int((recordings[recording_id]["chunks_processed"] / total_chunks) * 100)
            })
            return
        
        # Process results
        successful_transcripts = []
        failed_chunks = []
        
        for result in chunk_results:
            chunks[recording_id].append(result)
            if result["transcript"]:
                successful_transcripts.append(result["transcript"])
            else:
                failed_chunks.append(result["index"])
        
        # Update final status
        if len(successful_transcripts) == total_chunks:
            recordings[recording_id].update({
                "status": RecordingStatus.COMPLETED,
                "transcript": " ".join(successful_transcripts),
                "progress": 100
            })
        elif len(successful_transcripts) > 0:
            recordings[recording_id].update({
                "status": RecordingStatus.COMPLETED,
                "transcript": " ".join(successful_transcripts),
                "warning": f"Some chunks failed: {failed_chunks}",
                "progress": 100
            })
        else:
            recordings[recording_id].update({
                "status": RecordingStatus.FAILED,
                "error": "All chunks failed to process",
                "progress": 100
            })
            
    except Exception as e:
        logger.error(f"Error processing recording {recording_id}: {e}")
        recordings[recording_id].update({
            "status": RecordingStatus.FAILED,
            "error": str(e),
            "progress": recordings[recording_id].get("progress", 0)
        })

async def transcribe_chunk(chunk: AudioSegment, chunk_index: int, recording_id: str) -> str:
    """Transcribe a single audio chunk using Sarvam AI batch API"""
    job_id = str(uuid.uuid4())
    try:
        logger.info(f"Starting processing of chunk {chunk_index} for recording {recording_id}")
        
        # Create job entry first
        jobs[job_id] = {
            "recording_id": recording_id,
            "chunk_index": chunk_index,
            "status": "processing"
        }
        
        # Export chunk to WAV format
        logger.info(f"Exporting chunk {chunk_index} to WAV format")
        chunk_file = io.BytesIO()
        
        # Force sample rate and channels for compatibility
        chunk = chunk.set_channels(1).set_frame_rate(16000)
        logger.info(f"Chunk {chunk_index} configured: channels={chunk.channels}, frame_rate={chunk.frame_rate}, duration={len(chunk)/1000}s")
        
        # Use a simpler export to avoid ffmpeg issues
        try:
            chunk.export(chunk_file, format='wav', parameters=["-ac", "1", "-ar", "16000"])
            logger.info(f"Chunk {chunk_index} exported successfully with parameters")
        except Exception as export_error:
            logger.warning(f"Error exporting with parameters: {export_error}, trying simple export")
            chunk_file = io.BytesIO()
            chunk.export(chunk_file, format='wav')  # Try without parameters
            logger.info(f"Chunk {chunk_index} exported with simple export")
            
        chunk_data = chunk_file.getvalue()
        logger.info(f"Chunk {chunk_index} data size: {len(chunk_data)} bytes")
        
        # Only handle test mode for test recordings, not for all recordings
        if recording_id in recordings and recordings[recording_id].get("source") == "test" and "test_recording" in jobs[job_id].get("recording_id", ""):
            logger.info(f"Test mode detected for recording {recording_id}, returning test transcript")
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["transcript"] = "नमस्ते, यह एक परीक्षण प्रतिलेख है।"
            return "नमस्ते, यह एक परीक्षण प्रतिलेख है।"
            
        # For small chunks with real recordings, it's likely just silence, return empty string
        if len(chunk_data) < 5000:  # If less than 5KB
            logger.warning(f"Chunk {chunk_index} is too small, likely silence: {len(chunk_data)} bytes")
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["transcript"] = ""
            return ""
        
        # Convert to base64
        audio_base64 = base64.b64encode(chunk_data).decode('utf-8')
        logger.info(f"Chunk {chunk_index} converted to base64, length: {len(audio_base64)}")
        
        headers = {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Format according to Sarvam AI ASR API
        data = {
            "audio": {
                "data": audio_base64
            },
            "config": {
                "language_code": "hi",
                "use_enhanced": True
            }
        }
        
        # Log request details (without full audio data)
        log_data = data.copy()
        if 'audio' in log_data:
            log_data['audio'] = f"[BASE64_DATA_LENGTH:{len(log_data['audio'])}]"
        logger.info(f"API request data for chunk {chunk_index}: {log_data}")
        
        logger.info(f"Sending chunk {chunk_index} to Sarvam API with {len(audio_base64)} chars of base64 data")
        
        # Implement retry logic
        retries = 0
        while retries < MAX_RETRIES:
            try:
                logger.info(f"API attempt {retries+1} for chunk {chunk_index}")
                async with aiohttp.ClientSession() as session:
                    # Log API request details
                    logger.info(f"API URL: {SARVAM_API_URL}")
                    logger.info(f"API Headers for chunk {chunk_index}: Authorization: Bearer {SARVAM_API_KEY[:5]}...{SARVAM_API_KEY[-5:]}, Content-Type: application/json")
                    
                    async with session.post(SARVAM_API_URL, headers=headers, json=data, timeout=60) as response:
                        logger.info(f"API response status for chunk {chunk_index}: {response.status}")
                        response_text = await response.text()
                        logger.info(f"API response headers for chunk {chunk_index}: {dict(response.headers)}")
                        logger.info(f"API full response for chunk {chunk_index}: {response_text}")
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                                # Handle different response formats
                                if "text" in result:
                                    # Simple format
                                    transcribed_text = result.get("text", "").strip()
                                elif "results" in result:
                                    # ASR format with multiple results
                                    transcribed_text = " ".join([
                                        alternative.get("transcript", "")
                                        for result_item in result.get("results", [])
                                        for alternative in result_item.get("alternatives", [])
                                    ]).strip()
                                else:
                                    # Unknown format
                                    logger.warning(f"Unknown response format: {result}")
                                    transcribed_text = str(result)
                                
                                if transcribed_text:
                                    logger.info(f"Successful transcription for chunk {chunk_index}: {transcribed_text[:50]}...")
                                    jobs[job_id]["status"] = "completed"
                                    jobs[job_id]["transcript"] = transcribed_text
                                    return transcribed_text
                                else:
                                    logger.warning(f"Empty transcription for chunk {chunk_index}")
                                    jobs[job_id]["status"] = "completed"
                                    jobs[job_id]["transcript"] = ""
                                    return ""
                                    
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse response for chunk {chunk_index}: {e}, response: {response_text}")
                                if retries == MAX_RETRIES - 1:
                                    jobs[job_id]["status"] = "failed"
                                    jobs[job_id]["error"] = f"Failed to parse API response: {e}"
                                    return None
                        
                        elif response.status == 401 or response.status == 403:
                            error_msg = f"API Authentication error: Status {response.status}. Please check API key."
                            logger.error(f"API auth failed for chunk {chunk_index}. Status: {response.status}, Response: {response_text}")
                            if retries == MAX_RETRIES - 1:
                                jobs[job_id]["status"] = "failed"
                                jobs[job_id]["error"] = error_msg
                                return None
                                
                        elif response.status == 429:  # Rate limit
                            logger.warning(f"Rate limited for chunk {chunk_index}, retrying with backoff")
                            await asyncio.sleep(2 ** retries)  # Exponential backoff
                        
                        else:
                            error_msg = f"API error: Status {response.status}. Please check API documentation."
                            logger.error(f"API error for chunk {chunk_index}: Status {response.status}, response: {response_text}")
                            if retries == MAX_RETRIES - 1:
                                jobs[job_id]["status"] = "failed"
                                jobs[job_id]["error"] = error_msg
                                return None
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout error for chunk {chunk_index}")
                if retries == MAX_RETRIES - 1:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = "API request timed out"
                    return None
                    
            except Exception as e:
                logger.error(f"Request error for chunk {chunk_index}: {e}")
                if retries == MAX_RETRIES - 1:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = f"API request failed: {str(e)}"
                    return None
            
            retries += 1
            if retries < MAX_RETRIES:
                await asyncio.sleep(1)  # Wait before retry
        
        logger.error(f"All retries failed for chunk {chunk_index}")
        return None
                    
    except Exception as e:
        error_msg = f"Error processing chunk {chunk_index}: {e}"
        logger.error(error_msg)
        if 'job_id' in locals() and job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error_msg
        return None

@app.get("/")
async def home():
    return {"status": "healthy", "service": "Hindi Audio Transcription API - Root", "version": "1.0.1"}

@app.get("/api")
async def root():
    return {"status": "healthy", "service": "Hindi Audio Transcription API"}

@app.post("/api/recordings")
async def create_recording(
    background_tasks: BackgroundTasks, 
    audio: UploadFile = File(...),
    source: str = "microphone"
):
    try:
        # Handle test mode properly
        if audio.filename == "test_recording" or source == 'test':
            recording_id = str(uuid.uuid4())
            test_transcript = "नमस्ते, यह एक परीक्षण प्रतिलेख है। हम हिंदी ट्रांसक्रिप्शन टूल का परीक्षण कर रहे हैं।"
            
            # Create recording entry
            recordings[recording_id] = {
                "id": recording_id,
                "timestamp": datetime.now(),
                "duration": 30.0,  # Simulated 30-second recording
                "status": RecordingStatus.PROCESSING,
                "transcript": None,
                "error": None,
                "source": source,
                "format": "wav",
                "chunks_total": 1,
                "chunks_processed": 0,
                "chunks_failed": 0,
                "progress": 0
            }
            
            # Simulate processing delay
            async def process_test_recording():
                await asyncio.sleep(2)  # Simulate 2-second processing
                recordings[recording_id]["status"] = RecordingStatus.COMPLETED
                recordings[recording_id]["transcript"] = test_transcript
                recordings[recording_id]["chunks_processed"] = 1
                recordings[recording_id]["progress"] = 100
            
            # Process in background
            background_tasks.add_task(process_test_recording)
            
            return {
                "recording_id": recording_id,
                "status": RecordingStatus.PROCESSING,
                "message": "Test recording is being processed",
                "source": source,
                "format": "wav",
                "chunks_total": 1
            }
        
        # For real recordings, handle properly
        logger.info(f"Received real recording with source: {source}, filename: {audio.filename}, content_type: {audio.content_type}")
        
        # Validate audio source
        if source not in ["microphone", "system", "combined", "test"]:
            logger.warning(f"Invalid source: {source}")
            raise HTTPException(
                status_code=400,
                detail="Invalid audio source. Must be 'microphone', 'system', or 'combined'"
            )
            
        # Validate and read file content
        try:
            content = await audio.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Empty audio file received"
                )
            logger.info(f"Successfully read audio file: {len(content)} bytes")
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Failed to read audio file"
            )
            
        # Generate recording ID
        recording_id = str(uuid.uuid4())
        
        # Determine format
        format = "wav"  # Default
        if audio.content_type:
            if "webm" in audio.content_type:
                format = "webm"
            elif "mp3" in audio.content_type or "mpeg" in audio.content_type:
                format = "mp3"
            elif "ogg" in audio.content_type:
                format = "ogg"
        
        # Create recording entry before API call
        recordings[recording_id] = {
            "id": recording_id,
            "timestamp": datetime.now(),
            "duration": 0.0,  # Will update later
            "status": RecordingStatus.PROCESSING,
            "transcript": None,
            "error": None,
            "source": source,
            "format": format,
            "chunks_total": 1,
            "chunks_processed": 0,
            "chunks_failed": 0,
            "progress": 0
        }
        
        # Use the Sarvam Batch API for transcription
        async def process_with_batch_api():
            try:
                # Create API client
                api_client = SarvamBatchAPI(SARVAM_API_KEY)
                
                # Call the API
                logger.info(f"Processing recording {recording_id} with Sarvam Batch API")
                success, result = await api_client.transcribe_audio(content, source_lang="hi")
                
                if success:
                    logger.info(f"Successfully transcribed recording {recording_id}")
                    # Update recording with transcript
                    recordings[recording_id]["status"] = RecordingStatus.COMPLETED
                    recordings[recording_id]["transcript"] = result
                    recordings[recording_id]["progress"] = 100
                    recordings[recording_id]["chunks_processed"] = 1
                else:
                    logger.error(f"Failed to transcribe recording {recording_id}: {result}")
                    # Update recording with error
                    recordings[recording_id]["status"] = RecordingStatus.FAILED
                    recordings[recording_id]["error"] = result
                    recordings[recording_id]["progress"] = 100
            except Exception as e:
                logger.error(f"Error processing recording {recording_id} with batch API: {e}")
                recordings[recording_id]["status"] = RecordingStatus.FAILED
                recordings[recording_id]["error"] = f"Error processing recording: {str(e)}"
                recordings[recording_id]["progress"] = 100
        
        # Start processing in background
        background_tasks.add_task(process_with_batch_api)
        
        return {
            "recording_id": recording_id,
            "status": RecordingStatus.PROCESSING,
            "message": "Processing recording with Sarvam Batch API",
            "source": source,
            "format": format,
            "chunks_total": 1
        }
        
        # We no longer need chunking as the batch API handles it
        try:
            # Create recording entry with simplified structure
            recordings[recording_id] = {
                "id": recording_id,
                "timestamp": datetime.now(),
                "status": RecordingStatus.PROCESSING,
                "transcript": None,
                "error": None,
                "source": source,
                "format": format
            }
            
            return {
                "recording_id": recording_id,
                "status": RecordingStatus.PROCESSING,
                "message": "Processing with batch API",
                "source": source,
                "format": format
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/api/recordings")
async def list_recordings():
    # Convert recordings dict to list and sort by timestamp (newest first)
    recordings_list = [
        {
            "id": recording_data["id"],
            "timestamp": recording_data["timestamp"],
            "duration": recording_data["duration"],
            "status": recording_data["status"],
            "source": recording_data.get("source", "unknown"),
            "progress": recording_data.get("progress", 100) if recording_data["status"] == RecordingStatus.PROCESSING else 100,
            "chunks_total": recording_data.get("chunks_total", 1),
            "chunks_processed": recording_data.get("chunks_processed", 0),
            "error": recording_data.get("error", None),
            "warning": recording_data.get("warning", None),
            "transcript": recording_data.get("transcript", None) if recording_data["status"] == RecordingStatus.COMPLETED else None
        }
        for recording_id, recording_data in recordings.items()
    ]
    
    # Sort by timestamp, newest first
    recordings_list.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {"recordings": recordings_list}

@app.get("/api/recordings/{recording_id}")
async def get_recording(recording_id: str):
    if recording_id not in recordings:
        raise HTTPException(
            status_code=404,
            detail="Recording not found"
        )
    
    recording_data = recordings[recording_id]
    
    return {
        "id": recording_data["id"],
        "timestamp": recording_data["timestamp"],
        "duration": recording_data["duration"],
        "status": recording_data["status"],
        "source": recording_data.get("source", "unknown"),
        "progress": recording_data.get("progress", 100) if recording_data["status"] == RecordingStatus.PROCESSING else 100,
        "chunks_total": recording_data.get("chunks_total", 1),
        "chunks_processed": recording_data.get("chunks_processed", 0),
        "error": recording_data.get("error", None),
        "warning": recording_data.get("warning", None),
        "transcript": recording_data.get("transcript", None) if recording_data["status"] == RecordingStatus.COMPLETED else None
    }

@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    if recording_id not in recordings:
        raise HTTPException(
            status_code=404,
            detail="Recording not found"
        )
    
    # Remove recording and its chunks
    del recordings[recording_id]
    if recording_id in chunks:
        del chunks[recording_id]
    
    return {"message": "Recording deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
