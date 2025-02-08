import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from scipy.signal import butter, filtfilt

# Configuration
config = {
    "video_path": "3cf596e2bcc34862abc89bd2eca4a985_1.mp4",
    "json_path": "3cf596e2bcc34862abc89bd2eca4a985.json",
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "bp_model.pth"  # Path to save the trained model
}

# Load annotations
with open(config["json_path"], 'r') as f:
    annotations = json.load(f)

def extract_bp_values(annotations):
    """Deep search for BP values in complex JSON structures"""
    bp_key_variants = [
        ('bp_sys', 'bp_dia'),
        ('systolic', 'diastolic'),
        ('BP_SYS', 'BP_DIA'),
        ('blood_pressure_systolic', 'blood_pressure_diastolic')
    ]

    def deep_search(node, path=None):
        if path is None:
            path = []
            
        if isinstance(node, dict):
            for sys_key, dia_key in bp_key_variants:
                if sys_key in node and dia_key in node:
                    sys_val = node[sys_key].get('value', node[sys_key]) if isinstance(node[sys_key], (dict, list)) else node[sys_key]
                    dia_val = node[dia_key].get('value', node[dia_key]) if isinstance(node[dia_key], (dict, list)) else node[dia_key]
                    
                    if isinstance(sys_val, list): sys_val = sys_val[0]
                    if isinstance(dia_val, list): dia_val = dia_val[0]
                        
                    if sys_val is not None and dia_val is not None:
                        print(f"Found BP values at path: {' → '.join(path + [sys_key, dia_key])}")
                        return float(sys_val), float(dia_val)
            
            for k, v in node.items():
                result = deep_search(v, path + [k])
                if result: return result
                
        elif isinstance(node, list):
            for i, item in enumerate(node):
                result = deep_search(item, path + [str(i)])
                if result: return result
                
        return None

    result = deep_search(annotations)
    
    if result:
        return result
    else:
        print("Full annotation structure:")
        print(json.dumps(annotations, indent=2))
        raise ValueError("BP values not found. Check structure above and add appropriate keys.")

def extract_rppg(video_path):
    """Improved rPPG signal extraction with proper frame handling"""
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    signals = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = gray_frame[y:y+h, x:x+w]
            signals.append(np.mean(roi))
            
    cap.release()
    return np.array(signals)

def preprocess_signals(signals):
    """Enhanced signal processing pipeline"""
    b, a = butter(3, [0.5, 4], btype='bandpass', fs=30)
    filtered = filtfilt(b, a, signals)
    window_size = 5
    smoothed = np.convolve(filtered, np.ones(window_size)/window_size, mode='same')
    return (smoothed - np.mean(smoothed)) / np.std(smoothed)

class BPDataset(Dataset):
    def __init__(self, signals, bp_values, window_size=150):
        self.window_size = window_size
        self.signals = self.create_windows(signals)
        self.bp_values = np.tile(bp_values, (len(self.signals), 1))

    def create_windows(self, signals):
        """Convert 1D signal to sliding windows"""
        windows = []
        for i in range(len(signals) - self.window_size + 1):
            windows.append(signals[i:i+self.window_size])
        return np.array(windows)

    def __len__(self): return len(self.signals)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0),  # Add channel dimension
            torch.tensor(self.bp_values[idx], dtype=torch.float32)
        )

class BPEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),  # Input: (batch, 1, 150)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output: (32, 75)
            
            nn.Conv1d(32, 64, 3, padding=1),  # Output: (64, 75)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output: (64, 37)
            
            nn.Conv1d(64, 128, 3, padding=1),  # Output: (128, 37)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output: (128, 1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.cnn(x)  # Output shape: (batch, 128, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128)
        return self.regressor(x)

def train_model(train_loader, val_loader, config):
    model = BPEstimator().to(config["device"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    criterion = nn.HuberLoss()

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for signals, bp in train_loader:
            signals, bp = signals.to(config["device"]), bp.to(config["device"])
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, bp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for signals, bp in val_loader:
                signals, bp = signals.to(config["device"]), bp.to(config["device"])
                val_loss += criterion(model(signals), bp).item()
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model  # Return the trained model

def main():
    # Extract and process signals
    signals = extract_rppg(config["video_path"])
    if len(signals) < 150:
        raise ValueError(f"Signal too short ({len(signals)} frames). Need at least 150 frames.")
    signals = preprocess_signals(signals)

    # Extract BP values
    try:
        bp_sys, bp_dia = extract_bp_values(annotations)
        print(f"Extracted BP values - SYS: {bp_sys}, DIA: {bp_dia}")
    except ValueError as e:
        print(f"BP Extraction Error: {str(e)}")
        return

    # Create dataset
    dataset = BPDataset(signals, np.array([bp_sys, bp_dia]))
    train_size = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, len(dataset)-train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])

    # Train and save model
    model = train_model(train_loader, val_loader, config)
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"\nModel saved successfully to {config['model_save_path']}")

if __name__ == "__main__":
    main()