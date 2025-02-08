import React, { useEffect, useRef, useState } from 'react';
import { Activity, Moon, Sun, Upload } from 'lucide-react';

interface BPData {
  systolic: number;
  diastolic: number;
  status: string;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [bpData, setBpData] = useState<BPData | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState<string>('');
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  // Draw face overlay
  const drawFaceOverlay = (x: number, y: number, width: number, height: number) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;

    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    // Clear previous drawings
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw circle
    const centerX = x + width / 2;
    const centerY = y + height / 2;
    const radius = Math.min(width, height) / 2;

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Add pulse animation
    if (isScanning) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius + 5, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(0, 255, 0, 0.2)';
      ctx.stroke();
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640,
          height: 480,
          facingMode: 'user'
        } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();

        // Set overlay canvas dimensions
        if (overlayCanvasRef.current) {
          overlayCanvasRef.current.width = 640;
          overlayCanvasRef.current.height = 480;
        }
      }
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(track => track.stop());
  };

  const captureFrame = async () => {
    if (!canvasRef.current || !videoRef.current) return;
    
    const context = canvasRef.current.getContext('2d');
    if (!context) return;

    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    
    context.drawImage(videoRef.current, 0, 0);

    try {
      const blob = await new Promise<Blob>((resolve) => {
        canvasRef.current?.toBlob((b) => {
          if (b) resolve(b);
        }, 'image/jpeg');
      });

      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to process image');

      const data: BPData = await response.json();
      setBpData(data);

      // Update face overlay with mock coordinates (you'll get these from backend)
      drawFaceOverlay(220, 140, 200, 200);
    } catch (err) {
      setError('Failed to process image. Please try again.');
    }
  };

  const startScanning = () => {
    setIsScanning(true);
    setError('');
    const scanInterval = setInterval(async () => {
      await captureFrame();
    }, 1000);

    setTimeout(() => {
      clearInterval(scanInterval);
      setIsScanning(false);
      // Clear overlay when scanning stops
      const ctx = overlayCanvasRef.current?.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, 640, 480);
    }, 30000);
  };

  const handleVideoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.includes('video/')) {
      setError('Please upload a video file');
      return;
    }

    setUploadedVideo(file);
    setIsScanning(true);
    setError('');

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5000/analyze-video', {
        method: 'POST',
        body: formData,
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          );
          setUploadProgress(progress);
        },
      });

      if (!response.ok) throw new Error('Failed to process video');

      const data = await response.json();
      setBpData(data);
    } catch (err) {
      setError('Failed to process video. Please try again.');
    } finally {
      setIsScanning(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-white text-black'}`}>
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Blood Pressure Monitor</h1>
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600"
          >
            {isDarkMode ? '🌞' : '🌙'}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="relative">
            <video
              ref={videoRef}
              className="w-full rounded-lg shadow-lg"
              autoPlay
              playsInline
              muted
            />
            <canvas 
              ref={overlayCanvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
            />
            <canvas ref={canvasRef} className="hidden" />
          </div>

          <div className="space-y-6">
            <div className="flex gap-4">
              <button
                onClick={startScanning}
                disabled={isScanning}
                className={`flex-1 py-3 rounded-lg font-semibold ${
                  isScanning
                    ? 'bg-gray-500 cursor-not-allowed'
                    : 'bg-blue-500 hover:bg-blue-600'
                }`}
              >
                {isScanning ? 'Scanning...' : 'Start Scan'}
              </button>

              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isScanning}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-semibold ${
                  isScanning
                    ? 'bg-gray-500 cursor-not-allowed'
                    : 'bg-green-500 hover:bg-green-600'
                }`}
              >
                <Upload size={20} />
                Upload Video
              </button>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                className="hidden"
              />
            </div>

            {uploadProgress > 0 && (
              <div className="w-full bg-gray-700 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-500 text-white rounded-lg">
                {error}
              </div>
            )}

            {bpData && (
              <div className="p-6 bg-gray-800 rounded-lg">
                <h2 className="text-xl font-semibold mb-4">Results</h2>
                <div className="space-y-2">
                  <p>Systolic: {bpData.systolic.toFixed(0)} mmHg</p>
                  <p>Diastolic: {bpData.diastolic.toFixed(0)} mmHg</p>
                  <p className={`font-bold ${
                    bpData.status === 'Normal' ? 'text-green-500' : 
                    bpData.status === 'Elevated' ? 'text-yellow-500' : 'text-red-500'
                  }`}>
                    Status: {bpData.status}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;