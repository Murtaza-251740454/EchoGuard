# Voice Security - Audio Anomaly Detection System

A real-time audio anomaly detection system using deep learning that can identify various audio events and security threats through voice analysis.

## 🌟 Features

- **Real-time Audio Processing**: Process audio files and detect anomalies in real-time
- **Multiple Prediction Modes**: 
  - Single prediction for complete audio files
  - Streaming prediction for continuous monitoring
  - Continuous prediction with sliding window analysis
- **RESTful API**: FastAPI-based backend for easy integration
- **History Tracking**: Automatic logging of all predictions with timestamps
- **High Accuracy**: CNN-based model using MFCC features for robust audio classification
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🏗️ Architecture

The system consists of:
- **Deep Learning Model**: CNN architecture specialized for MFCC feature analysis
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients) for audio representation
- **API Backend**: FastAPI server for handling requests and responses
- **Data Storage**: CSV-based history logging system

## 📋 Requirements

- Python 3.8+
- PyTorch
- torchaudio
- FastAPI
- pandas
- numpy
- joblib
- pydub
- uvicorn

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VoiceSecurity.git
   cd VoiceSecurity
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchaudio fastapi pandas numpy joblib pydub uvicorn python-multipart
   ```

4. **Set up directories**
   ```bash
   mkdir data models
   ```

5. **Add your trained models**
   - Place your trained CNN model: `models/cnn_mfcc_model.pth`
   - Place your label encoder: `models/label_encoder.pkl`

## 🎯 Usage

### Starting the Server

```bash
python backend.py
```

The API server will start on `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```
Returns server status.

#### 2. Single Prediction
```bash
POST /predict
```
Upload an audio file (.wav, .mp3) for single prediction.

#### 3. Streaming Prediction
```bash
POST /stream_predict
```
Upload an audio file for prediction with history logging.

#### 4. Continuous Prediction
```bash
POST /continuous_predict
```
Process audio with sliding window analysis for detailed timeline predictions.

#### 5. Get History
```bash
GET /history
```
Retrieve prediction history sorted by timestamp (most recent first).

### Example Usage

#### Using curl:
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@your_audio_file.wav"

# Get history
curl -X GET "http://localhost:8000/history"
```

#### Using Python requests:
```python
import requests

# Single prediction
with open('audio_file.wav', 'rb') as f:
    files = {'audio': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())

# Get history
response = requests.get('http://localhost:8000/history')
print(response.json())
```

## 📁 Project Structure

```
VoiceSecurity/
├── backend.py          # FastAPI server and API endpoints
├── main.py            # Core prediction functions and model loading
├── CNN.py             # CNN model architecture definition
├── models/            # Trained models directory
│   ├── cnn_mfcc_model.pth    # Trained CNN model
│   └── label_encoder.pkl     # Label encoder for class mapping
├── data/              # Data storage
│   └── history.csv    # Prediction history log
└── README.md          # This file
```

## 🔧 Configuration

### Model Parameters
- **MFCC Features**: 40 coefficients
- **Window Size**: 2048 FFT points
- **Hop Length**: 512 samples
- **Mel Filters**: 40
- **Max Sequence Length**: 174 frames

### Prediction Thresholds
- **Confidence Threshold**: 0.8 (adjustable)
- **Anomaly Detection**: Predictions below threshold marked as "No Anomaly"

## 📊 Model Training

The system uses a CNN model trained on MFCC features. To train your own model:

1. Prepare your audio dataset
2. Extract MFCC features using the provided `extract_mfcc` function
3. Train the CNN model defined in `CNN.py`
4. Save the model and label encoder to the `models/` directory

## 🔍 Monitoring and Logging

The system automatically logs all predictions with:
- Predicted class
- Confidence score
- Timestamp
- History accessible via `/history` endpoint

## 🚨 Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Missing model files
- Audio processing errors
- File I/O operations


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Include system information and error logs

## 🔮 Future Enhancements

- [ ] Real-time WebSocket streaming
- [ ] Mobile app integration
- [ ] Advanced visualization dashboard
- [ ] Multi-model ensemble predictions
- [ ] Cloud deployment support
- [ ] Real-time alert system