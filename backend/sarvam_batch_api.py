import requests
import json
import base64
import logging
import time
import aiohttp
import asyncio
from typing import Optional, Dict, Any, Tuple, List
import os
import io
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class SarvamBatchAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_headers = {
            "API-Subscription-Key": api_key,
            "Content-Type": "application/json"
        }
        self.init_url = "https://api.sarvam.ai/speech-to-text-translate/job/init"
        self.status_url_template = "https://api.sarvam.ai/speech-to-text-translate/job/{job_id}/status"
        self.submit_url = "https://api.sarvam.ai/speech-to-text-translate/job"
        
    async def transcribe_audio(self, audio_data: bytes, source_lang: str = "hi") -> Tuple[bool, str]:
        """Process audio through the Sarvam Batch API"""
        try:
            # Step 1: Initialize a job
            job_info = await self._initialize_job()
            if not job_info:
                return False, "Failed to initialize job"
                
            job_id = job_info.get("job_id")
            input_storage_path = job_info.get("input_storage_path")
            
            if not job_id or not input_storage_path:
                return False, "Invalid job information returned from API"
                
            logger.info(f"Initialized job {job_id} with input path {input_storage_path}")
            
            # Step 2: Upload audio to Azure blob storage
            upload_success = await self._upload_audio(input_storage_path, audio_data)
            if not upload_success:
                return False, "Failed to upload audio to storage"
                
            logger.info(f"Successfully uploaded audio for job {job_id}")
            
            # Step 3: Start the job
            start_success = await self._start_job(job_id)
            if not start_success:
                return False, "Failed to start transcription job"
                
            logger.info(f"Successfully started job {job_id}")
            
            # Step 4: Poll for job completion
            transcript, error = await self._poll_job_status(job_id)
            if error:
                return False, f"Error during job processing: {error}"
                
            # Return the transcript
            return True, transcript
                
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            return False, f"Error during transcription: {str(e)}"
            
    async def _initialize_job(self) -> Optional[Dict[str, Any]]:
        """Initialize a new job and return job details"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.init_url,
                    headers=self.base_headers,
                    timeout=30
                ) as response:
                    if response.status != 202:
                        error_text = await response.text()
                        logger.error(f"Error initializing job: Status {response.status}, Response: {error_text}")
                        return None
                        
                    result = await response.json()
                    return result
        except Exception as e:
            logger.error(f"Error in _initialize_job: {e}")
            return None
            
    async def _upload_audio(self, storage_path: str, audio_data: bytes) -> bool:
        """Upload audio data to Azure storage"""
        try:
            # Convert audio to WAV format
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            # Set to mono and 16kHz for best transcription results
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export as WAV
            audio_buffer = io.BytesIO()
            audio.export(audio_buffer, format="wav")
            wav_data = audio_buffer.getvalue()
            
            # Create the audio.wav file at the storage path
            upload_url = f"{storage_path}/audio.wav"
            
            # Upload using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    upload_url,
                    data=wav_data,
                    headers={
                        "x-ms-blob-type": "BlockBlob",
                        "Content-Type": "audio/wav"
                    },
                    timeout=120  # Longer timeout for audio upload
                ) as response:
                    if response.status not in (200, 201):
                        error_text = await response.text()
                        logger.error(f"Error uploading audio: Status {response.status}, Response: {error_text}")
                        return False
                        
                    return True
        except Exception as e:
            logger.error(f"Error in _upload_audio: {e}")
            return False
    
    async def _start_job(self, job_id: str) -> bool:
        """Start the transcription job"""
        try:
            data = {
                "job_id": job_id,
                "job_parameters": {
                    "source_lang": "hi",
                    "task_type": "transcribe"
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.submit_url,
                    headers=self.base_headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status not in (200, 201, 202):
                        error_text = await response.text()
                        logger.error(f"Error starting job: Status {response.status}, Response: {error_text}")
                        return False
                        
                    return True
        except Exception as e:
            logger.error(f"Error in _start_job: {e}")
            return False
    
    async def _poll_job_status(self, job_id: str, max_polls: int = 60, poll_interval: int = 5) -> Tuple[str, Optional[str]]:
        """Poll for job completion and return transcript or error"""
        status_url = self.status_url_template.format(job_id=job_id)
        
        for _ in range(max_polls):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        status_url,
                        headers=self.base_headers,
                        timeout=30
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Error checking job status: Status {response.status}, Response: {error_text}")
                            await asyncio.sleep(poll_interval)
                            continue
                            
                        result = await response.json()
                        job_status = result.get("status")
                        
                        logger.info(f"Job status: {job_status}")
                        
                        if job_status == "completed":
                            # Get transcript from output
                            output_path = result.get("output_storage_path")
                            if not output_path:
                                return "", "No output path available"
                                
                            transcript = await self._get_transcript(output_path)
                            return transcript, None
                            
                        elif job_status == "failed":
                            error = result.get("error", "Unknown error")
                            return "", f"Job failed: {error}"
                            
                        # Still processing, wait for next poll
                        await asyncio.sleep(poll_interval)
                        
            except Exception as e:
                logger.error(f"Error polling job status: {e}")
                await asyncio.sleep(poll_interval)
                
        # If we get here, we've timed out
        return "", "Job polling timed out"
    
    async def _get_transcript(self, output_path: str) -> str:
        """Get the transcript from the output storage"""
        try:
            # The output path should have a transcript.json file
            transcript_url = f"{output_path}/transcript.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(transcript_url, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error getting transcript: Status {response.status}, Response: {error_text}")
                        return ""
                        
                    result = await response.json()
                    
                    # The transcript format might be different, we'll handle multiple possible formats
                    if isinstance(result, dict):
                        # Try common transcript fields
                        if "text" in result:
                            return result["text"]
                        elif "transcript" in result:
                            return result["transcript"]
                        elif "transcription" in result:
                            return result["transcription"]
                        # If none of those, dump the entire result as a string
                        return json.dumps(result)
                    elif isinstance(result, list):
                        # Might be an array of segments
                        texts = []
                        for item in result:
                            if isinstance(item, dict):
                                if "text" in item:
                                    texts.append(item["text"])
                                elif "transcript" in item:
                                    texts.append(item["transcript"])
                        if texts:
                            return " ".join(texts)
                        # If no texts found, dump the entire result
                        return json.dumps(result)
                    else:
                        # Just return the result as a string
                        return str(result)
                        
        except Exception as e:
            logger.error(f"Error getting transcript: {e}")
            return ""
