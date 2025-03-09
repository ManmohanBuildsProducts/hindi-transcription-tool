import requests
import unittest
import io
import time

class TestHindiTranscriptionAPI(unittest.TestCase):
    BASE_URL = "http://localhost:8001"
    
    def test_01_api_health(self):
        """Test API health endpoint"""
        print("\nTesting API health...")
        response = requests.get(f"{self.BASE_URL}/api")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "Hindi Audio Transcription API")
        print("✅ API health check passed")

    def test_02_list_recordings(self):
        """Test listing recordings"""
        print("\nTesting list recordings endpoint...")
        response = requests.get(f"{self.BASE_URL}/api/recordings")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recordings", data)
        self.assertIsInstance(data["recordings"], list)
        print("✅ List recordings endpoint working")

    def test_03_test_recording(self):
        """Test the test recording functionality"""
        print("\nTesting test recording functionality...")
        
        # Create a test recording
        files = {
            'audio': ('test_recording', io.BytesIO(b'test audio content'), 'audio/wav')
        }
        data = {'source': 'microphone'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        print(f"Response: {response.status_code} - {response.text}")
        self.assertEqual(response.status_code, 200)
        
        recording_id = response.json().get("recording_id")
        self.assertIsNotNone(recording_id)
        print(f"Created recording with ID: {recording_id}")
        
        # Poll for completion
        max_attempts = 10
        for attempt in range(max_attempts):
            response = requests.get(f"{self.BASE_URL}/api/recordings/{recording_id}")
            if response.status_code == 200:
                data = response.json()
                print(f"Attempt {attempt+1}: Status = {data['status']}")
                if data["status"] == "completed":
                    self.assertIn("नमस्ते", data["transcript"])
                    print("✅ Test recording transcribed successfully")
                    return
            time.sleep(1)
            
        self.fail("Transcription did not complete in time")

    def test_04_invalid_source(self):
        """Test invalid audio source validation"""
        print("\nTesting invalid audio source...")
        files = {
            'audio': ('test.wav', io.BytesIO(b'test'), 'audio/wav')
        }
        data = {'source': 'invalid_source'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        self.assertEqual(response.status_code, 400)
        print("✅ Invalid source validation working")

    def test_05_invalid_audio(self):
        """Test invalid audio format"""
        print("\nTesting invalid audio format...")
        files = {
            'audio': ('test.txt', io.BytesIO(b'not audio'), 'text/plain')
        }
        data = {'source': 'microphone'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        self.assertEqual(response.status_code, 415)
        print("✅ Invalid audio format validation working")

    def test_06_delete_recording(self):
        """Test deleting a recording"""
        print("\nTesting delete recording...")
        
        # First create a recording
        files = {
            'audio': ('test_recording', io.BytesIO(b'test audio'), 'audio/wav')
        }
        data = {'source': 'microphone'}
        
        response = requests.post(f"{self.BASE_URL}/api/recordings", files=files, data=data)
        self.assertEqual(response.status_code, 200, f"Failed to create test recording: {response.text}")
        recording_id = response.json().get("recording_id")
        
        # Delete it
        response = requests.delete(f"{self.BASE_URL}/api/recordings/{recording_id}")
        self.assertEqual(response.status_code, 200)
        
        # Verify it's gone
        response = requests.get(f"{self.BASE_URL}/api/recordings/{recording_id}")
        self.assertEqual(response.status_code, 404)
        print("✅ Delete functionality working")

if __name__ == '__main__':
    unittest.main(verbosity=2)
