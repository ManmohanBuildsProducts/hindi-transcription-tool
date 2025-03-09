import unittest
import requests
import io
import time
from pathlib import Path

class TestHindiTranscriptionAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8001"
    
    def setUp(self):
        # Check if API is healthy
        try:
            response = requests.get(f"{self.BASE_URL}/api")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["service"], "Hindi Audio Transcription API")
        except Exception as e:
            self.fail(f"API is not running: {str(e)}")

    def test_test_recording(self):
        """Test the test recording endpoint"""
        print("\nTesting test recording functionality...")
        
        # Create a test recording
        files = {
            'audio': ('test_recording', io.BytesIO(b'test'), 'audio/wav')
        }
        data = {'source': 'microphone'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        
        recording_id = response.json()["recording_id"]
        self.assertIsNotNone(recording_id)
        
        # Wait for processing to complete (max 5 seconds)
        max_retries = 5
        for _ in range(max_retries):
            response = requests.get(f"{self.BASE_URL}/api/recordings/{recording_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(1)
        
        # Verify test transcript
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("नमस्ते", data["transcript"])
        print("✅ Test recording functionality working")

    def test_list_recordings(self):
        """Test listing recordings"""
        print("\nTesting list recordings endpoint...")
        response = requests.get(f"{self.BASE_URL}/api/recordings")
        self.assertEqual(response.status_code, 200)
        recordings = response.json()
        self.assertIsInstance(recordings, list)
        print("✅ List recordings endpoint working")

    def test_invalid_audio_source(self):
        """Test invalid audio source validation"""
        print("\nTesting invalid audio source validation...")
        files = {
            'audio': ('test.wav', io.BytesIO(b'test'), 'audio/wav')
        }
        data = {'source': 'invalid_source'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid audio source", response.json()["detail"])
        print("✅ Invalid audio source validation working")

    def test_invalid_audio_format(self):
        """Test invalid audio format validation"""
        print("\nTesting invalid audio format validation...")
        files = {
            'audio': ('test.txt', io.BytesIO(b'test'), 'text/plain')
        }
        data = {'source': 'microphone'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported audio format", response.json()["detail"])
        print("✅ Invalid audio format validation working")

    def test_delete_recording(self):
        """Test deleting a recording"""
        print("\nTesting delete recording functionality...")
        
        # First create a test recording
        files = {
            'audio': ('test_recording', io.BytesIO(b'test'), 'audio/wav')
        }
        data = {'source': 'microphone'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        recording_id = response.json()["recording_id"]
        
        # Delete the recording
        response = requests.delete(f"{self.BASE_URL}/api/recordings/{recording_id}")
        self.assertEqual(response.status_code, 200)
        
        # Verify recording is deleted
        response = requests.get(f"{self.BASE_URL}/api/recordings/{recording_id}")
        self.assertEqual(response.status_code, 404)
        print("✅ Delete recording functionality working")

if __name__ == '__main__':
    unittest.main(verbosity=2)