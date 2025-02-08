from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from scipy.signal import butter, filtfilt
import base64
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load pre-trained BP estimation model (replace with your model)
class BPEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

model = BPEstimator()
model.load_state_dict(torch.load("bp_model.pth"))  # Load your trained model
model.eval()

# Preprocess rPPG signals
def preprocess_signals(signal_value):
    # Create a synthetic signal of appropriate length
    min_length = 128  # Minimum length required for the model
    synthetic_signal = np.full(min_length, signal_value)
    
    # Apply simple moving average to smooth the signal
    window_size = 5
    smoothed = np.convolve(synthetic_signal, np.ones(window_size)/window_size, mode='valid')
    
    # Normalize the signal
    normalized = (smoothed - np.mean(smoothed)) / (np.std(smoothed) + 1e-6)
    
    return normalized

# Extract rPPG signals from a video frame
def extract_rppg(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 2)  # Reduced min neighbors from 4 to 2
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Extract mean values from face ROI
        g_signal = np.mean(face_roi[:, :, 1])  # Green channel
        return g_signal, {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
    else:
        return np.mean(frame[:, :, 1]), None  # Fallback to full frame green channel

def process_video(video_path):
    """Process video file for blood pressure estimation"""
    cap = cv2.VideoCapture(video_path)
    signals = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract rPPG from frame
        signal, _ = extract_rppg(frame)
        signals.append(signal)
    
    cap.release()
    
    # Process collected signals
    if len(signals) > 0:
        # Average the signals
        avg_signal = np.mean(signals)
        processed_signal = preprocess_signals(avg_signal)
        signal_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(signal_tensor)
            systolic, diastolic = prediction[0].numpy()
            
        return float(systolic), float(diastolic)
    
    return None, None

@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        
        # Create temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            
            # Process the video
            systolic, diastolic = process_video(tmp_file.name)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            if systolic is None or diastolic is None:
                return jsonify({"error": "Could not process video"}), 400
                
            status = get_bp_status(systolic, diastolic)
            
            return jsonify({
                "systolic": systolic,
                "diastolic": diastolic,
                "status": status
            })

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return jsonify({"error": "Failed to process video"}), 500

def get_bp_status(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    else:
        return "Stage 2 Hypertension"

if __name__ == "__main__":
    app.run(debug=True)