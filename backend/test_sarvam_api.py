import requests
import base64
import json
import sys
import os

def test_sarvam_direct_endpoint():
    """Test the Sarvam AI direct endpoint"""
    API_KEY = "ec7650e8-3560-48c7-8c69-649f1c659680"
    
    # Try direct endpoint with Bearer token
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": {
            "input_type": "text",
            "input_data": "Hello, this is a test."
        },
        "task": "stt",
        "config": {
            "language": "hi"
        }
    }
    
    print(f"Testing with Bearer token: {API_KEY}")
    try:
        response = requests.post(
            "https://api.sarvam.ai/direct/v1",
            headers=headers,
            json=data
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try direct endpoint with x-api-key
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"\nTesting with x-api-key: {API_KEY}")
    try:
        response = requests.post(
            "https://api.sarvam.ai/direct/v1",
            headers=headers,
            json=data
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try v2 API
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"\nTesting v2 API with x-api-key: {API_KEY}")
    try:
        response = requests.post(
            "https://api.sarvam.ai/v2",
            headers=headers,
            json=data
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try API for batch
    batch_headers = {
        "API-Subscription-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"\nTesting batch API with API-Subscription-Key: {API_KEY}")
    try:
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text-translate/job/init",
            headers=batch_headers
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_other_apis():
    # Try some common API patterns to see if any work
    API_KEY = "ec7650e8-3560-48c7-8c69-649f1c659680"
    
    endpoints = [
        "https://api.sarvam.ai/v1/asr",
        "https://api.sarvam.ai/v1/transcribe",
        "https://api.sarvam.ai/v1/transcribe/batch",
        "https://api.sarvam.ai/v1/audio/transcribe",
        "https://api.sarvam.ai/v1/audio/asr",
        "https://api.sarvam.ai/v1/stt"
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "audio": {
                "data": base64.b64encode(b"test audio data").decode('utf-8')
            },
            "config": {
                "language_code": "hi"
            }
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=5
            )
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
        
        # Also try with x-api-key header
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=5
            )
            print(f"With x-api-key - Status code: {response.status_code}")
            print(f"With x-api-key - Response: {response.text[:200]}...")
        except Exception as e:
            print(f"With x-api-key - Error: {e}")

if __name__ == "__main__":
    print("Testing Sarvam API endpoints...")
    test_sarvam_direct_endpoint()
    test_other_apis()
