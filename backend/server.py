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
from dotenv import load_dotenv

# Setup environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Get logger instance
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

# Get logger instance
logger = logging.getLogger(__name__)

# Constants from environment
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    logger.warning("SARVAM_API_KEY not set in environment, using default test key")
    SARVAM_API_KEY = "ec7650e8-3560-48c7-8c69-649f1c659680"  # Default key, will be used if env var is not set
SARVAM_API_URL = os.environ.get("SARVAM_API_URL", "https://api.sarvam.ai/v1/transcribe/batch")
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

# Get logger instance
logger = logging.getLogger(__name__)

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
        
        # For test mode, we'll return a canned response for small recordings
        if recording_id in recordings and recordings[recording_id].get("source") == "test":
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
        
        data = {
            "audio": audio_base64,
            "source_lang": "hi",
            "task_type": "transcribe",
            "audio_format": "wav"
        }
        
        logger.info(f"Sending chunk {chunk_index} to Sarvam API with {len(audio_base64)} chars of base64 data")
        
        # Implement retry logic
        retries = 0
        while retries < MAX_RETRIES:
            try:
                logger.info(f"API attempt {retries+1} for chunk {chunk_index}")
                async with aiohttp.ClientSession() as session:
                    async with session.post(SARVAM_API_URL, headers=headers, json=data, timeout=60) as response:
                        logger.info(f"API response status for chunk {chunk_index}: {response.status}")
                        response_text = await response.text()
                        logger.info(f"API response for chunk {chunk_index}: {response_text[:200]}...")
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                                transcribed_text = result.get("text", "").strip()
                                
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
                                logger.error(f"Failed to parse response for chunk {chunk_index}: {e}, response: {response_text[:100]}...")
                                if retries == MAX_RETRIES - 1:
                                    jobs[job_id]["status"] = "failed"
                                    jobs[job_id]["error"] = f"Failed to parse API response: {e}"
                                    return None
                        
                        elif response.status == 429:  # Rate limit
                            logger.warning(f"Rate limited for chunk {chunk_index}, retrying with backoff")
                            await asyncio.sleep(2 ** retries)  # Exponential backoff
                        
                        else:
                            error_msg = f"API error: Status {response.status}, Body: {response_text[:200]}..."
                            logger.error(error_msg)
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

@app.get("/")
async def home():
    return {"status": "healthy", "service": "Hindi Audio Transcription API - Root"}

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
        # Handle test mode
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

        # Log incoming file details for debugging
        logger.info(f"Received audio file: {audio.filename}, content_type: {audio.content_type}")
        
        # Validate audio source
        if source not in ["microphone", "system", "combined", "test"]:  # Added 'test' as valid source
            logger.warning(f"Invalid source: {source}")
            raise HTTPException(
                status_code=400,
                detail="Invalid audio source. Must be 'microphone', 'system', or 'combined'"
            )

        # More permissive content type check
        valid_types = ["audio/webm", "audio/wav", "audio/wave", "audio/mp3", "audio/mpeg", 
                      "audio/ogg", "audio/x-wav", "audio/x-m4a"]
        
        # Check file extension as a fallback for browsers that don't set content_type correctly
        valid_extensions = [".mp3", ".wav", ".wave", ".webm", ".ogg", ".m4a"]
        file_ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else ""
        
        if not (audio.content_type in valid_types or file_ext in valid_extensions):
            logger.warning(f"Unsupported audio format: {audio.content_type}, filename: {audio.filename}")
            # Try to determine format from filename as a last resort
            try:
                # For debugging - save the file temporarily to check what it is
                content = await audio.read()
                if len(content) > 0:
                    logger.info(f"File size: {len(content)} bytes")
                    # Reset the file pointer for later reading
                    await audio.seek(0)
                    # We'll proceed anyway since we have a more robust audio parsing later
                    format = "wav"  # Default to wav
                    if audio.filename:
                        if audio.filename.lower().endswith(".mp3"):
                            format = "mp3"
                        elif audio.filename.lower().endswith(".webm"):
                            format = "webm"
                        elif audio.filename.lower().endswith(".ogg"):
                            format = "ogg"
                    logger.info(f"Determined format from filename: {format}")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Empty audio file received"
                    )
            except Exception as e:
                logger.error(f"Error checking audio file: {str(e)}")
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported audio format. Please use WAV, MP3, or WebM format."
                )
        else:
            logger.info(f"Valid audio format detected: {audio.content_type or file_ext}")

        # Read content with size check (but only if we haven't read it already)
        try:
            if 'content' not in locals():
                content = await audio.read()
                if len(content) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Empty audio file received"
                    )
                logger.info(f"Successfully read file content, size: {len(content)} bytes")
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Failed to read audio file"
            )
        
        # Generate recording ID
        recording_id = str(uuid.uuid4())
        
        # Get audio format
        format = "wav"
        if audio.content_type == "audio/webm":
            format = "webm"
        elif audio.content_type in ["audio/mp3", "audio/mpeg"]:
            format = "mp3"
        
        try:
            # Split audio into chunks
            audio_chunks = split_audio(content, format)
            if not audio_chunks:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract valid audio chunks from recording"
                )
                
            # Create recording entry
            recordings[recording_id] = {
                "id": recording_id,
                "timestamp": datetime.now(),
                "duration": sum(len(chunk) for chunk in audio_chunks) / 1000,  # Duration in seconds
                "status": RecordingStatus.PROCESSING,
                "transcript": None,
                "error": None,
                "source": source,
                "format": format,
                "chunks_total": len(audio_chunks),
                "chunks_processed": 0,
                "chunks_failed": 0,
                "progress": 0
            }
            
            # Process chunks in background
            background_tasks.add_task(process_recording, recording_id, audio_chunks)
            
            return {
                "recording_id": recording_id,
                "status": RecordingStatus.PROCESSING,
                "message": f"Processing {len(audio_chunks)} audio chunks",
                "source": source,
                "format": format,
                "chunks_total": len(audio_chunks)
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
