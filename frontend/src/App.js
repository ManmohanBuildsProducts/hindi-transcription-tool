import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Get API base URL from environment variable
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [recordingId, setRecordingId] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const [error, setError] = useState(null);
  const [deviceStatus, setDeviceStatus] = useState({
    microphone: 'unchecked',
    systemAudio: 'unchecked'
  });
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioSource, setAudioSource] = useState('microphone'); // 'microphone', 'system', or 'combined'
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Recording references
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const durationTimer = useRef(null);
  const micStream = useRef(null);
  const systemStream = useRef(null);
  const combinedStream = useRef(null);
  
  useEffect(() => {
    fetchRecordings();
    checkAudioDevices();
  }, []);

  useEffect(() => {
    if (isRecording) {
      durationTimer.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
    } else {
      clearInterval(durationTimer.current);
      setRecordingDuration(0);
    }
    return () => clearInterval(durationTimer.current);
  }, [isRecording]);

  const fetchRecordings = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/recordings`);
      if (!response.ok) throw new Error('Failed to fetch recordings');
      const data = await response.json();
      setRecordings(data.recordings || []);
    } catch (error) {
      console.error('Error fetching recordings:', error);
      setError('Failed to load recordings. Please check your network connection.');
    }
  };

  const checkAudioDevices = async () => {
    try {
      // Check browser support for basic audio
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setDeviceStatus({
          microphone: 'unavailable',
          systemAudio: 'unavailable'
        });
        setError('Audio recording is not supported in your browser. Please use Chrome, Firefox, or Edge.');
        return false;
      }

      // Check microphone availability
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        
        // Stop all tracks after checking
        stream.getTracks().forEach(track => track.stop());
        
        setDeviceStatus(prev => ({
          ...prev,
          microphone: 'available'
        }));
      } catch (err) {
        console.error('Microphone check error:', err);
        
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setError('Microphone access denied. Please allow microphone access in your browser settings.');
        } else {
          setError('Could not access microphone. Please check your microphone connection.');
        }
        
        setDeviceStatus(prev => ({
          ...prev,
          microphone: 'unavailable'
        }));
      }

      // Check system audio availability (via desktop capture)
      try {
        if (navigator.mediaDevices.getDisplayMedia) {
          setDeviceStatus(prev => ({
            ...prev,
            systemAudio: 'available'
          }));
        } else {
          setDeviceStatus(prev => ({
            ...prev,
            systemAudio: 'unavailable'
          }));
          console.warn('System audio capture not supported in this browser');
        }
      } catch (err) {
        console.error('System audio check error:', err);
        setDeviceStatus(prev => ({
          ...prev,
          systemAudio: 'unavailable'
        }));
      }

      return true;
    } catch (err) {
      console.error('Device check error:', err);
      setError('Could not check audio devices. Please reload the page.');
      setDeviceStatus({
        microphone: 'unavailable',
        systemAudio: 'unavailable'
      });
      return false;
    }
  };

  const getMicrophoneStream = async () => {
    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
    } catch (err) {
      console.error('Error getting microphone stream:', err);
      throw new Error('Could not access microphone');
    }
  };

  const getSystemAudioStream = async () => {
    try {
      // Request desktop capture with audio
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true
      });
      
      // If no audio track, throw error
      if (!stream.getAudioTracks().length) {
        stream.getTracks().forEach(track => track.stop());
        throw new Error('No system audio selected. Please select "Share audio" when sharing your screen.');
      }
      
      // Stop video tracks, we only need audio
      stream.getVideoTracks().forEach(track => track.stop());
      
      return stream;
    } catch (err) {
      console.error('Error getting system audio stream:', err);
      throw new Error(err.message || 'Could not access system audio');
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      audioChunks.current = [];
      
      // Stop any existing streams
      if (micStream.current) {
        micStream.current.getTracks().forEach(track => track.stop());
      }
      if (systemStream.current) {
        systemStream.current.getTracks().forEach(track => track.stop());
      }
      if (combinedStream.current) {
        combinedStream.current.getTracks().forEach(track => track.stop());
      }
      
      micStream.current = null;
      systemStream.current = null;
      combinedStream.current = null;

      let stream;
      
      // Get the appropriate stream based on audio source
      if (audioSource === 'microphone') {
        micStream.current = await getMicrophoneStream();
        stream = micStream.current;
      } 
      else if (audioSource === 'system') {
        systemStream.current = await getSystemAudioStream();
        stream = systemStream.current;
      }
      else if (audioSource === 'combined') {
        // Get both streams
        micStream.current = await getMicrophoneStream();
        systemStream.current = await getSystemAudioStream();
        
        // Combine audio tracks from both streams
        const audioContext = new AudioContext();
        const micSource = audioContext.createMediaStreamSource(micStream.current);
        const systemSource = audioContext.createMediaStreamSource(systemStream.current);
        const destination = audioContext.createMediaStreamDestination();
        
        // Connect sources to destination
        micSource.connect(destination);
        systemSource.connect(destination);
        
        // Use the combined stream
        combinedStream.current = destination.stream;
        stream = combinedStream.current;
      }

      // Determine MIME type based on browser support
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp3'
      ];
      
      const supportedType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type));
      
      if (!supportedType) {
        throw new Error('Your browser does not support any compatible audio format');
      }

      // Create media recorder
      const options = {
        mimeType: supportedType,
        audioBitsPerSecond: 128000
      };
      
      mediaRecorder.current = new MediaRecorder(stream, options);
      
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };
      
      // Get data every 30 seconds to ensure we don't lose data if browser crashes
      mediaRecorder.current.start(30000);
      setIsRecording(true);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      setError(error.message || 'Error starting recording');
      setIsRecording(false);
      
      // Clean up any streams
      if (micStream.current) {
        micStream.current.getTracks().forEach(track => track.stop());
      }
      if (systemStream.current) {
        systemStream.current.getTracks().forEach(track => track.stop());
      }
      if (combinedStream.current) {
        combinedStream.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  const stopRecording = async () => {
    try {
      if (!mediaRecorder.current || mediaRecorder.current.state === 'inactive') {
        return;
      }

      setIsRecording(false);
      setIsProcessing(true);
      
      // Stop the recorder and get final chunk
      mediaRecorder.current.stop();
      
      // Wait for final dataavailable event
      await new Promise(resolve => {
        mediaRecorder.current.addEventListener('stop', resolve, { once: true });
      });
      
      // Stop all tracks in all streams
      if (micStream.current) {
        micStream.current.getTracks().forEach(track => track.stop());
      }
      if (systemStream.current) {
        systemStream.current.getTracks().forEach(track => track.stop());
      }
      if (combinedStream.current) {
        combinedStream.current.getTracks().forEach(track => track.stop());
      }

      // Create a single blob from all chunks
      const audioBlob = new Blob(audioChunks.current, { 
        type: mediaRecorder.current.mimeType 
      });

      // Upload recording
      const formData = new FormData();
      formData.append('audio', audioBlob);
      formData.append('source', audioSource);

      const response = await fetch(`${BACKEND_URL}/api/recordings`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to upload recording: ${errorText}`);
      }

      const result = await response.json();
      setRecordingId(result.recording_id);

      // Poll for completion
      await pollRecordingStatus(result.recording_id);
      
      // Refresh recordings list
      await fetchRecordings();
      
      // Reset
      audioChunks.current = [];

    } catch (error) {
      console.error('Error stopping recording:', error);
      setError(error.message || 'Error processing recording');
    } finally {
      setIsProcessing(false);
      mediaRecorder.current = null;
      micStream.current = null;
      systemStream.current = null;
      combinedStream.current = null;
    }
  };

  const pollRecordingStatus = async (id) => {
    let attempts = 0;
    const maxAttempts = 120; // 2 minutes max waiting time (1 sec interval)
    
    while (attempts < maxAttempts) {
      try {
        const statusResponse = await fetch(`${BACKEND_URL}/api/recordings/${id}`);
        
        if (!statusResponse.ok) {
          attempts++;
          await new Promise(resolve => setTimeout(resolve, 1000));
          continue;
        }
        
        const statusData = await statusResponse.json();
        
        // If processing complete or failed, exit polling
        if (statusData.status === 'completed' || statusData.status === 'failed') {
          return statusData;
        }
        
        // If still processing, continue polling
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error('Error polling status:', error);
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    throw new Error('Transcription is taking longer than expected. Check recordings list for status.');
  };

  const deleteRecording = async (id) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/recordings/${id}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete recording');
      }
      
      // Update recordings list
      setRecordings(recordings.filter(recording => recording.id !== id));
      
    } catch (error) {
      console.error('Error deleting recording:', error);
      setError('Failed to delete recording');
    }
  };

  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return [
      hours > 0 ? hours : null,
      minutes.toString().padStart(hours > 0 ? 2 : 1, '0'),
      secs.toString().padStart(2, '0')
    ].filter(Boolean).join(':');
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">हिंदी ऑडियो ट्रांसक्रिप्शन</h1>
          <p className="text-gray-600 mt-2">Record audio and transcribe Hindi in real-time</p>
        </header>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          {/* Device Status */}
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${
                deviceStatus.microphone === 'available' ? 'bg-green-500' :
                deviceStatus.microphone === 'unavailable' ? 'bg-red-500' :
                'bg-yellow-500'
              }`}></div>
              <span className="text-sm text-gray-600">
                {deviceStatus.microphone === 'available' ? 'Microphone Ready' :
                 deviceStatus.microphone === 'unavailable' ? 'Microphone Not Available' :
                 'Checking Microphone...'}
              </span>
            </div>
            
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${
                deviceStatus.systemAudio === 'available' ? 'bg-green-500' :
                deviceStatus.systemAudio === 'unavailable' ? 'bg-red-500' :
                'bg-yellow-500'
              }`}></div>
              <span className="text-sm text-gray-600">
                {deviceStatus.systemAudio === 'available' ? 'System Audio Ready' :
                 deviceStatus.systemAudio === 'unavailable' ? 'System Audio Not Available' :
                 'Checking System Audio...'}
              </span>
            </div>
          </div>

          {/* Audio Source Selection */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex rounded-md shadow-sm" role="group">
              <button
                type="button"
                onClick={() => setAudioSource('microphone')}
                disabled={deviceStatus.microphone !== 'available' || isRecording}
                className={`px-4 py-2 text-sm font-medium ${
                  deviceStatus.microphone !== 'available' ? 'opacity-50 cursor-not-allowed ' : ''
                } ${
                  audioSource === 'microphone'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } border border-gray-200 rounded-l-lg`}
              >
                Microphone
              </button>
              <button
                type="button"
                onClick={() => setAudioSource('system')}
                disabled={deviceStatus.systemAudio !== 'available' || isRecording}
                className={`px-4 py-2 text-sm font-medium ${
                  deviceStatus.systemAudio !== 'available' ? 'opacity-50 cursor-not-allowed ' : ''
                } ${
                  audioSource === 'system'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } border-t border-b border-gray-200`}
              >
                System Audio
              </button>
              <button
                type="button"
                onClick={() => setAudioSource('combined')}
                disabled={deviceStatus.microphone !== 'available' || deviceStatus.systemAudio !== 'available' || isRecording}
                className={`px-4 py-2 text-sm font-medium ${
                  deviceStatus.microphone !== 'available' || deviceStatus.systemAudio !== 'available' 
                    ? 'opacity-50 cursor-not-allowed ' 
                    : ''
                } ${
                  audioSource === 'combined'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } border border-gray-200 rounded-r-lg`}
              >
                Combined
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Recording Controls */}
          <div className="flex flex-col items-center">
            <div className="flex space-x-4 mb-4">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={
                  (audioSource === 'microphone' && deviceStatus.microphone !== 'available') ||
                  (audioSource === 'system' && deviceStatus.systemAudio !== 'available') ||
                  (audioSource === 'combined' && (deviceStatus.microphone !== 'available' || deviceStatus.systemAudio !== 'available')) ||
                  isProcessing
                }
                className={`relative inline-flex items-center px-6 py-3 rounded-full text-white font-medium transition-all ${
                  isProcessing 
                    ? 'bg-gray-400 cursor-not-allowed'
                    : isRecording
                      ? 'bg-red-500 hover:bg-red-600'
                      : 'bg-blue-500 hover:bg-blue-600'
                }`}
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </>
                ) : isRecording ? (
                  <>
                    <span className="animate-pulse mr-2 text-red-200">●</span>
                    Stop Recording
                  </>
                ) : (
                  'Start Recording'
                )}
              </button>
              
              {/* Quick Test Button */}
              <button
                onClick={async () => {
                  try {
                    setIsProcessing(true);
                    setError(null);
                    
                    // Create a test FormData with an empty file
                    const formData = new FormData();
                    const testBlob = new Blob(["test"], { type: "text/plain" });
                    const testFile = new File([testBlob], "test_recording", { type: "text/plain" });
                    formData.append('audio', testFile);
                    formData.append('source', audioSource);
                    
                    // Submit test recording request
                    const response = await fetch(`${BACKEND_URL}/api/recordings`, {
                      method: 'POST',
                      body: formData
                    });
                    
                    if (!response.ok) {
                      const errorText = await response.text();
                      throw new Error(`Test failed: ${errorText}`);
                    }
                    
                    const result = await response.json();
                    
                    // Poll for completion
                    await pollRecordingStatus(result.recording_id);
                    
                    // Refresh recordings list
                    await fetchRecordings();
                    
                    // Reset state
                    setIsProcessing(false);
                  } catch (error) {
                    console.error('Error in test:', error);
                    setError(error.message || 'Test recording failed');
                    setIsProcessing(false);
                  }
                }}
                disabled={isProcessing || isRecording}
                className="px-5 py-3 bg-green-500 hover:bg-green-600 text-white font-medium rounded-full disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                Test Transcription
              </button>
            </div>

            {isRecording && (
              <div className="mt-4 text-center">
                <div className="text-lg font-semibold text-gray-700">
                  {formatTime(recordingDuration)}
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {audioSource === 'microphone' ? 'Recording from microphone' :
                   audioSource === 'system' ? 'Recording system audio' :
                   'Recording microphone and system audio'}
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  For long recordings, processing may take several minutes
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Transcriptions List */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-6 flex items-center justify-between">
            <span>Transcriptions</span>
            <button 
              onClick={fetchRecordings}
              className="text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1 rounded"
            >
              Refresh
            </button>
          </h2>
          <div className="space-y-6">
            {recordings.length > 0 ? (
              recordings.map((recording) => (
                <div key={recording.id} className="border rounded-lg p-4 shadow-sm">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="text-sm text-gray-500">
                        {formatDate(recording.timestamp)}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <p className="text-xs px-2 py-0.5 bg-gray-100 rounded text-gray-700">
                          {recording.source}
                        </p>
                        <p className="text-xs text-gray-600">
                          {Math.round(recording.duration)}s
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        recording.status === 'completed' ? 'bg-green-100 text-green-800' :
                        recording.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {recording.status}
                      </span>
                      <button
                        onClick={() => deleteRecording(recording.id)}
                        className="text-gray-400 hover:text-red-500"
                        title="Delete recording"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  {recording.status === 'processing' && (
                    <div className="mt-3">
                      <div className="flex items-center">
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-blue-600 h-2.5 rounded-full" 
                            style={{ width: `${recording.progress}%` }}
                          ></div>
                        </div>
                        <span className="ml-2 text-xs text-gray-500">
                          {recording.progress}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Processing chunk {recording.chunks_processed} of {recording.chunks_total}
                      </p>
                    </div>
                  )}

                  {recording.status === 'completed' && recording.transcript && (
                    <div className="mt-3">
                      <div className="bg-gray-50 rounded p-4 font-hindi">
                        <p className="text-gray-800 whitespace-pre-wrap text-base leading-relaxed">
                          {recording.transcript}
                        </p>
                      </div>
                      <div className="mt-2 flex gap-2">
                        <button 
                          onClick={() => navigator.clipboard.writeText(recording.transcript)}
                          className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                          </svg>
                          Copy Text
                        </button>
                        <button 
                          onClick={() => {
                            const blob = new Blob([recording.transcript], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `transcription-${recording.id.substring(0, 8)}.txt`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                          }}
                          className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                          Download
                        </button>
                      </div>
                    </div>
                  )}

                  {recording.status === 'failed' && (
                    <div className="bg-red-50 p-4 rounded mt-3">
                      <p className="text-red-700 text-sm">
                        {recording.error || 'Failed to process recording'}
                      </p>
                    </div>
                  )}

                  {recording.warning && (
                    <div className="bg-yellow-50 p-3 rounded mt-3">
                      <p className="text-yellow-700 text-xs">
                        Warning: {recording.warning}
                      </p>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>No recordings yet</p>
                <p className="text-sm mt-2">Start recording to see transcriptions here</p>
              </div>
            )}
          </div>
        </div>
        
        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Hindi Audio Transcription Tool</p>
          <p className="mt-1">Powered by Sarvam AI</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
