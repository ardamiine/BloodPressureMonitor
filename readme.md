# Blood Pressure Monitor - Video Analysis System

A web application that estimates blood pressure through facial video analysis, supporting both real-time camera feed and video upload.

[Download the pre-trained model (bp_model.pth)](https://www.dropbox.com/scl/fo/hamsbv7dhyhr7pewrfqz1/AP6-mpUieqW4du3keMI1a38?rlkey=u0c7luvuy60sjw2vhqgs8nyy7&e=1&dl=0)

## Prerequisites

Before installation, ensure you have:
- Node.js (v16+)
- Python (3.8+)
- npm or yarn
- pip (Python package manager)
- Git
- Pre-trained model file (bp_model.pth) - Download from link above

## Installation Guide

### 1. Create Project Structure
```bash
mkdir twise-night
cd twise-night
mkdir frontend backend
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install required packages
pip install -r requirements.txt

# Create app.py and paste the backend code
touch app.py
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Running the Application

### 1. Start Backend Server
```bash
# In the backend directory with virtual environment activated
python app.py

# The backend will start on http://localhost:5000
```

### 2. Start Frontend Development Server
```bash
# In the frontend directory
npm run dev
# The frontend will be available at http://localhost:5173
```

## Usage

### Live Camera Analysis
1. Open the application in your browser
2. Grant camera permissions when prompted
3. Click "Start Scan" button
4. Remain still for 30 seconds during scanning
5. View your blood pressure results

### Video Upload Analysis
1. Click "Upload Video" button
2. Select a video file from your device
3. Wait for upload and processing to complete
4. View your blood pressure results

## Features
- Real-time facial blood pressure analysis
- Video file upload and analysis
- Dark/Light mode toggle
- Progress tracking for video uploads
- Visual feedback during scanning
- Blood pressure classification

## Troubleshooting

### Common Issues

1. **Backend Won't Start**
```bash
# Check if port 5000 is in use
# For Linux/Mac:
lsof -i :5000
# For Windows:
netstat -ano | findstr :5000

# Try different port if needed
# Edit app.py:
app.run(debug=True, port=5001)
```

2. **Frontend Dependencies Issues**
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules
npm install
```

3. **Camera Access Issues**
- Ensure browser has camera permissions
- Check if camera is being used by another application
- Try a different browser (Chrome recommended)

4. **Video Upload Problems**
- Ensure video file is in supported format (MP4 recommended)
- Check file size (keep under 100MB for best results)
- Verify stable internet connection

## Project Structure
```
twise-night/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── index.html
├── backend/
│   ├── app.py
│   ├── bp_model.pth
│   └── requirements.txt
└── README.md
```

## Development Notes

### Required Browser Permissions
- Camera access for live scanning
- File system access for video upload

### Supported Formats
- Video: MP4, WebM, MOV
- Maximum recommended file size: 100MB
- Minimum video quality: 720p

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari (limited support)
- Edge

## Security Considerations
- Backend runs on localhost only
- CORS enabled for development
- Temporary files are automatically cleaned up
- No data persistence implemented

## Support

For issues and questions:
1. Check the troubleshooting section
2. Create an issue in the repository
3. Contact the development team

## License

This project is licensed under the MIT License.
