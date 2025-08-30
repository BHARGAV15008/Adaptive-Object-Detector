# Adaptive Object Detector

A real-time object detection system with interactive learning capabilities, built using Flask and YOLOv5. This application allows users to detect objects through a webcam feed and dynamically teach the system to recognize new objects through an intuitive web interface.

## 🚀 Features

- **Real-time Object Detection**: Live webcam feed with object detection using YOLOv5
- **Interactive Learning**: Label unknown objects directly through the web interface
- **Dynamic Model Retraining**: Automatically retrains the model when new labels are added
- **Live Object Counting**: Real-time display of detected object counts
- **Web-based Interface**: Clean, responsive web UI for monitoring and interaction
- **Custom Dataset Support**: Ability to train on custom datasets
- **Model Persistence**: Saves retrained models for future use

## 🏗️ Project Structure

```
adaptive-object-detector/
├── app.py                  # Main Flask application
├── Model.py               # Model loading and management
├── objectDetection.py     # Object detection utilities
├── appUtils.py           # Helper functions for labels
├── customDataset.py      # Custom dataset class for training
├── requirements.txt      # Python dependencies
├── datasets/
│   └── customeLabels.json # Custom object labels storage
├── retrainedModels/
│   ├── best.pt           # Retrained model weights
│   └── yolov5s-seg.pt   # Segmentation model
├── static/
│   ├── style.css         # Frontend styling
│   └── scripts.js        # Frontend JavaScript
└── templates/
    └── index.html        # Main web interface
```

## 🛠️ Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: YOLOv5 (PyTorch-based object detection)
- **Computer Vision**: OpenCV for image processing
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning**: PyTorch for model training and inference

## 📋 Prerequisites

- Python 3.7+
- Webcam or camera device
- CUDA-compatible GPU (optional, for faster inference)

## ⚙️ Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/BHARGAV15008/Adaptive-Object-Detector.git
   cd adaptive-object-detector
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify directory structure**
   Ensure the following directories exist:
   - `datasets/`
   - `retrainedModels/`
   - `static/`
   - `templates/`

## 🚀 Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

3. **Using the system**:
   - The webcam feed will start automatically
   - Known objects will be detected and labeled with bounding boxes
   - Unknown objects will prompt for user labeling
   - Object counts are displayed in real-time
   - When prompted, enter labels for unknown objects to teach the system

## 🔧 How It Works

### Core Components

1. **Model Management (`Model.py`)**:
   - Loads YOLOv5 models (pre-trained or custom)
   - Handles model retraining with new labels
   - Manages model persistence

2. **Object Detection (`objectDetection.py`)**:
   - Processes video frames for object detection
   - Draws bounding boxes and labels
   - Handles unknown object identification

3. **Web Application (`app.py`)**:
   - Serves the web interface
   - Streams video feed to browser
   - Handles user labeling requests
   - Updates object counts in real-time

4. **Frontend Interface**:
   - Displays live video feed
   - Shows object count statistics
   - Provides labeling form for unknown objects

### Learning Process

1. **Object Detection**: The system detects objects in the video feed
2. **Label Check**: Checks if object is known (has a label)
3. **User Interaction**: If unknown, prompts user for label
4. **Model Update**: Saves new label and triggers model retraining
5. **Improved Detection**: Updated model can now recognize the new object type

## 📊 API Endpoints

- `GET /` - Main application interface
- `GET /video_feed` - Real-time video stream
- `GET /object_count` - Current object count data (JSON)
- `POST /label_object` - Submit new object labels

## 🔧 Configuration

### Model Configuration
- Default model: YOLOv5s (downloaded automatically)
- Custom models: Place in `retrainedModels/` directory
- Update `model_path` in `Model.py` to use different models

### Camera Configuration
- Default: Built-in webcam (index 0)
- Change camera source in `app.py`, line 44: `cv2.VideoCapture(0)`
- For video files: `cv2.VideoCapture('path/to/video.mp4')`

## 🎯 Use Cases

- **Educational**: Learn about object detection and machine learning
- **Prototyping**: Quickly create custom object detection systems
- **Research**: Experiment with interactive learning approaches
- **Security**: Custom object detection for specific environments
- **Inventory**: Count and track specific objects in real-time

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

2. **Model loading errors**:
   - Check internet connection for initial YOLOv5 download
   - Verify PyTorch installation
   - Ensure sufficient memory for model loading

3. **Slow performance**:
   - Use GPU acceleration if available
   - Reduce video resolution
   - Close unnecessary applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open-source. Please check the license file for details.

## 🔮 Future Enhancements

- Multiple camera support
- Object tracking across frames
- Export trained models
- Batch image processing
- Mobile app integration
- Cloud deployment options
- Advanced metrics and analytics

## 📞 Support

For issues, questions, or contributions, please open an issue in the repository or contact the development team.

---

**Note**: This application requires a webcam and will download YOLOv5 models on first run. Ensure you have a stable internet connection for initial setup.
