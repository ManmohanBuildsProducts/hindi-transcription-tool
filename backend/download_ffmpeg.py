#!/usr/bin/env python3
"""
Download and set up ffmpeg for use in Render or similar environments
where apt-get installation is not possible due to read-only filesystem.
"""

import os
import sys
import stat
import logging
import requests
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FFMPEG_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
DOWNLOAD_PATH = "/tmp/ffmpeg.tar.xz"
EXTRACT_PATH = "/tmp/ffmpeg-extract"
FFMPEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")

def download_ffmpeg():
    """Download ffmpeg static build"""
    logger.info(f"Downloading ffmpeg from {FFMPEG_URL}")
    try:
        response = requests.get(FFMPEG_URL, stream=True)
        response.raise_for_status()
        
        with open(DOWNLOAD_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded ffmpeg to {DOWNLOAD_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to download ffmpeg: {e}")
        return False

def extract_ffmpeg():
    """Extract the downloaded tarball"""
    logger.info(f"Extracting ffmpeg from {DOWNLOAD_PATH}")
    try:
        Path(EXTRACT_PATH).mkdir(parents=True, exist_ok=True)
        with tarfile.open(DOWNLOAD_PATH) as tar:
            tar.extractall(path=EXTRACT_PATH)
        
        logger.info(f"Extracted ffmpeg to {EXTRACT_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract ffmpeg: {e}")
        return False

def setup_ffmpeg():
    """Set up ffmpeg in a directory accessible to the application"""
    try:
        # Create bin directory if it doesn't exist
        Path(FFMPEG_DIR).mkdir(parents=True, exist_ok=True)
        
        # Find the ffmpeg and ffprobe binaries in the extracted directory
        extracted_dir = next(Path(EXTRACT_PATH).glob("ffmpeg-*"))
        ffmpeg_path = extracted_dir / "ffmpeg"
        ffprobe_path = extracted_dir / "ffprobe"
        
        # Copy binaries to our bin directory
        target_ffmpeg = os.path.join(FFMPEG_DIR, "ffmpeg")
        target_ffprobe = os.path.join(FFMPEG_DIR, "ffprobe")
        
        # Create symbolic links
        if os.path.exists(target_ffmpeg):
            os.remove(target_ffmpeg)
        if os.path.exists(target_ffprobe):
            os.remove(target_ffprobe)
            
        # Copy files
        with open(ffmpeg_path, 'rb') as src, open(target_ffmpeg, 'wb') as dst:
            dst.write(src.read())
        with open(ffprobe_path, 'rb') as src, open(target_ffprobe, 'wb') as dst:
            dst.write(src.read())
        
        # Make executable
        os.chmod(target_ffmpeg, os.stat(target_ffmpeg).st_mode | stat.S_IEXEC)
        os.chmod(target_ffprobe, os.stat(target_ffprobe).st_mode | stat.S_IEXEC)
        
        logger.info(f"ffmpeg and ffprobe set up in {FFMPEG_DIR}")
        
        # Update PATH environment variable
        if FFMPEG_DIR not in os.environ['PATH']:
            os.environ['PATH'] = f"{FFMPEG_DIR}:{os.environ['PATH']}"
            logger.info(f"Updated PATH: {os.environ['PATH']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up ffmpeg: {e}")
        return False

def main():
    """Main function to download and set up ffmpeg"""
    if os.path.exists(os.path.join(FFMPEG_DIR, "ffmpeg")):
        logger.info("ffmpeg already set up, skipping download")
        return 0
    
    if not download_ffmpeg():
        return 1
    
    if not extract_ffmpeg():
        return 1
    
    if not setup_ffmpeg():
        return 1
    
    logger.info("ffmpeg setup completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
