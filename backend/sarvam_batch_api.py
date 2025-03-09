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
            # Log the storage path for debugging
            logger.info(f"Storage path for upload: {storage_path}")
            
            # Try to convert audio data but handle errors gracefully
            try:
                # Convert audio to WAV format
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                # Set to mono and 16kHz for best transcription results
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Export as WAV
                audio_buffer = io.BytesIO()
                audio.export(audio_buffer, format="wav")
                wav_data = audio_buffer.getvalue()
                logger.info(f"Successfully converted audio to WAV format, size: {len(wav_data)} bytes")
            except Exception as e:
                logger.warning(f"Failed to convert audio, using original data: {e}")
                # If conversion fails, use the original data
                wav_data = audio_data
            
            # Create the audio.wav file at the storage path
            # Use the exact URL format required by Azure Blob Storage
            upload_url = f"{storage_path}/audio.wav"
            logger.info(f"Upload URL: {upload_url}")
            
            # Try with different headers and methods if the first attempt fails
            
            # Method 1: aiohttp PUT with Azure headers
            try:
                logger.info("Trying upload method 1: aiohttp PUT with Azure headers")
                async with aiohttp.ClientSession() as session:
                    async with session.put(
                        upload_url,
                        data=wav_data,
                        headers={
                            "x-ms-blob-type": "BlockBlob",
                            "Content-Type": "audio/wav",
                            "x-ms-version": "2020-04-08"
                        },
                        timeout=120  # Longer timeout for audio upload
                    ) as response:
                        if response.status in (200, 201, 204):
                            logger.info(f"Upload successful with method 1: {response.status}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.warning(f"Method 1 failed: Status {response.status}, Response: {error_text}")
            except Exception as e:
                logger.warning(f"Method 1 exception: {e}")
            
            # Method 2: requests PUT (synchronous but more reliable)
            try:
                logger.info("Trying upload method 2: requests PUT (synchronous)")
                import requests
                response = requests.put(
                    upload_url,
                    data=wav_data,
                    headers={
                        "x-ms-blob-type": "BlockBlob",
                        "Content-Type": "audio/wav"
                    },
                    timeout=120
                )
                if response.status_code in (200, 201, 204):
                    logger.info(f"Upload successful with method 2: {response.status_code}")
                    return True
                else:
                    logger.warning(f"Method 2 failed: Status {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.warning(f"Method 2 exception: {e}")
            
            # Method 3: Try with different content type
            try:
                logger.info("Trying upload method 3: different content type")
                response = requests.put(
                    upload_url,
                    data=wav_data,
                    headers={
                        "x-ms-blob-type": "BlockBlob",
                        "Content-Type": "application/octet-stream"
                    },
                    timeout=120
                )
                if response.status_code in (200, 201, 204):
                    logger.info(f"Upload successful with method 3: {response.status_code}")
                    return True
                else:
                    logger.warning(f"Method 3 failed: Status {response.status_code}, Response: {response.text}")
            except Exception as e:
                logger.warning(f"Method 3 exception: {e}")
            
            # If we get here, all methods failed
            logger.error("All upload methods failed")
            return False
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
        logger.info(f"Polling job status at URL: {status_url}")
        
        for poll_attempt in range(max_polls):
            try:
                logger.info(f"Poll attempt {poll_attempt + 1}/{max_polls}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        status_url,
                        headers=self.base_headers,
                        timeout=30
                    ) as response:
                        response_text = await response.text()
                        logger.info(f"Job status response: Status {response.status}, Response: {response_text[:200]}...")
                        
                        if response.status != 200:
                            logger.error(f"Error checking job status: Status {response.status}, Response: {response_text}")
                            # If we get a 404, it might mean the job doesn't exist
                            if response.status == 404 and poll_attempt > 3:
                                return "", "Job not found after multiple attempts. It may have been deleted or never created."
                            
                            await asyncio.sleep(poll_interval)
                            continue
                        
                        try:
                            result = json.loads(response_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse job status response: {e}")
                            await asyncio.sleep(poll_interval)
                            continue
                            
                        job_status = result.get("status", "").lower()  # Normalize to lowercase
                        logger.info(f"Job status: {job_status}")
                        
                        # Handle different status values
                        if job_status in ("completed", "done", "success", "succeeded"):
                            # Get transcript from output
                            output_path = result.get("output_storage_path")
                            if not output_path:
                                logger.error("No output path in completed job response")
                                # Check if the result itself contains the transcript
                                if "text" in result:
                                    return result["text"], None
                                elif "transcript" in result:
                                    return result["transcript"], None
                                elif "result" in result:
                                    return str(result["result"]), None
                                else:
                                    # Return the full result as a last resort
                                    return str(result), None
                                    
                            logger.info(f"Job completed, fetching transcript from: {output_path}")
                            transcript = await self._get_transcript(output_path)
                            return transcript, None
                            
                        elif job_status in ("failed", "error", "failure"):
                            error = result.get("error", "Unknown error")
                            logger.error(f"Job failed: {error}")
                            return "", f"Job failed: {error}"
                            
                        elif job_status in ("processing", "running", "in_progress", "pending"):
                            logger.info(f"Job {job_id} still processing, waiting {poll_interval} seconds...")
                            # Still processing, wait for next poll
                            await asyncio.sleep(poll_interval)
                            
                        else:
                            logger.warning(f"Unknown job status: {job_status}")
                            await asyncio.sleep(poll_interval)
                        
            except Exception as e:
                logger.error(f"Error polling job status: {e}")
                await asyncio.sleep(poll_interval)
                
        # If we get here, we've timed out
        return "", "Job polling timed out"
    
    async def _get_transcript(self, output_path: str) -> str:
        """Get the transcript from the output storage"""
        try:
            logger.info(f"Getting transcript from output path: {output_path}")
            
            # Try several possible transcript file locations/names
            transcript_paths = [
                f"{output_path}/transcript.json",
                f"{output_path}/text.json",
                f"{output_path}/output.json",
                f"{output_path}/result.json",
                # Also try without .json extension
                f"{output_path}/transcript",
                f"{output_path}/text",
                # Try txt format
                f"{output_path}/transcript.txt",
                f"{output_path}/text.txt"
            ]
            
            for transcript_url in transcript_paths:
                logger.info(f"Trying to get transcript from: {transcript_url}")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(transcript_url, timeout=30) as response:
                            if response.status == 200:
                                logger.info(f"Successfully got response from {transcript_url}")
                                content_type = response.headers.get('Content-Type', '')
                                
                                if 'json' in content_type:
                                    # Handle JSON content
                                    try:
                                        result = await response.json()
                                        logger.info(f"Parsed JSON result: {str(result)[:200]}...")
                                        
                                        # The transcript format might be different, we'll handle multiple possible formats
                                        if isinstance(result, dict):
                                            # Try common transcript fields
                                            if "text" in result:
                                                return result["text"]
                                            elif "transcript" in result:
                                                return result["transcript"]
                                            elif "transcription" in result:
                                                return result["transcription"]
                                            elif "output" in result:
                                                return str(result["output"])
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
                                    except json.JSONDecodeError:
                                        # If it's not valid JSON, treat as text
                                        text = await response.text()
                                        logger.info(f"Invalid JSON, treating as text: {text[:200]}...")
                                        return text
                                else:
                                    # Handle non-JSON content (e.g., plain text)
                                    text = await response.text()
                                    logger.info(f"Got plain text: {text[:200]}...")
                                    return text
                            else:
                                logger.warning(f"Failed to get transcript from {transcript_url}: Status {response.status}")
                except Exception as e:
                    logger.warning(f"Error accessing {transcript_url}: {e}")
            
            # If we've tried all paths and none worked, return an error message
            logger.error("Failed to get transcript from any of the expected paths")
            return "Transcription completed but unable to retrieve transcript content."
                        
        except Exception as e:
            logger.error(f"Error in _get_transcript: {e}")
            return f"Error retrieving transcript: {str(e)}"
