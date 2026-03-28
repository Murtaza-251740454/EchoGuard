import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import joblib
import os
from CNN import CNNForMFCC  # Assuming CNNForMFCC is defined in CNN.py

# Instantiate model and load weights correctly
cnn = CNNForMFCC(num_classes=10)  # Adjust num_classes as per your model
cnn.load_state_dict(torch.load("models/cnn_mfcc_model.pth", map_location='cpu'))
cnn.eval()
label_encoder = joblib.load("models/label_encoder.pkl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = cnn.to(device)

# MFCC extraction function (returns 2D MFCC for CNN)
def extract_mfcc(file_path, n_mfcc=40, n_fft=2048, n_mels=40, hop_length=512, max_len=174):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.to(device)

        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length,
                'mel_scale': 'htk'
            }
        ).to(device)

        mfcc = mfcc_transform(waveform)  # shape: (channel, n_mfcc, time)
        mfcc = mfcc.mean(dim=0)  # average over channels if stereo

        # Pad or truncate to max_len
        if mfcc.shape[1] < max_len:
            pad_amount = max_len - mfcc.shape[1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc.cpu().numpy()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
# Predict function using CNN
async def single_predict_audio(file_path, threshold=0.8):
    print(f"🔊 Extracting Features")
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "⚠️ Could not extract features"

    # Convert to tensor, add batch and channel dimensions, move to device
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        print(f"🔊 Making Prediction")
        output = cnn(mfcc)  # shape: [1, num_classes]
        probs = torch.softmax(output, dim=1)  # shape: [1, num_classes]
        max_prob, pred_idx = torch.max(probs, dim=1)
        predicted_label = label_encoder.inverse_transform(pred_idx.cpu().numpy())

        return predicted_label[0], max_prob.item()
        
# Predict function using CNN
async def predict_audio(file_path, threshold=0.8):
    print(f"🔊 Extracting Features")
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "⚠️ Could not extract features"

    # Convert to tensor, add batch and channel dimensions, move to device
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        print(f"🔊 Making Prediction")
        output = cnn(mfcc)  # shape: [1, num_classes]
        probs = torch.softmax(output, dim=1)  # shape: [1, num_classes]
        max_prob, pred_idx = torch.max(probs, dim=1)
        predicted_label = label_encoder.inverse_transform(pred_idx.cpu().numpy())

        if max_prob.item() < threshold or predicted_label[0] == "dog_bark":
            return "No Anomaly",0.0
        else:
            return predicted_label[0], max_prob.item()

async def continuous_predict(file_path, window_size=2.0, hop_size=1.0, threshold=0.8, n_mfcc=40, n_fft=2048, n_mels=40, hop_length=512, max_len=174):
    """
    Process an audio file in a streaming fashion, making predictions on overlapping windows.
    
    Args:
        file_path (str): Path to the audio file
        window_size (float): Size of the analysis window in seconds
        hop_size (float): Hop size between windows in seconds
        threshold (float): Confidence threshold for predictions
        n_mfcc (int): Number of MFCC coefficients
        n_fft (int): FFT window size
        n_mels (int): Number of mel filters
        hop_length (int): Hop length for MFCC computation
        max_len (int): Maximum length of MFCC time frames
        
    Returns:
        list: List of tuples containing (start_time, end_time, prediction, confidence)
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.to(device)
        
        # Calculate samples per window and hop
        window_samples = int(window_size * sample_rate)
        hop_samples = int(hop_size * sample_rate)
        
        # Initialize MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length,
                'mel_scale': 'htk'
            }
        ).to(device)
        
        predictions = []
        total_samples = waveform.shape[1]
        
        print(f"🔊 Processing audio stream from {file_path}")
        
        # Slide window over audio
        for start_idx in range(0, total_samples - window_samples + 1, hop_samples):
            end_idx = start_idx + window_samples
            start_time = start_idx / sample_rate
            end_time = end_idx / sample_rate
            
            # Extract window
            window = waveform[:, start_idx:end_idx]
            
            # Compute MFCC
            mfcc = mfcc_transform(window)
            mfcc = mfcc.mean(dim=0)  # Average over channels if stereo
            
            # Pad or truncate to max_len
            if mfcc.shape[1] < max_len:
                pad_amount = max_len - mfcc.shape[1]
                mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
            else:
                mfcc = mfcc[:, :max_len]
            
            # Convert to tensor and add batch/channel dimensions
            mfcc = mfcc.unsqueeze(0).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = cnn(mfcc)
                probs = torch.softmax(output, dim=1)
                max_prob, pred_idx = torch.max(probs, dim=1)
                
                confidence = max_prob.item()
                if confidence < threshold:
                    prediction = "No Anomaly"
                else:
                    prediction = label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
                
                predictions.append((start_time, end_time, prediction, confidence))
                print(f"Window [{start_time:.2f}s - {end_time:.2f}s]: {prediction} (confidence: {confidence:.3f})")
        
        print("✅ Stream processing completed")
        return predictions
    
    except Exception as e:
        print(f"⚠️ Error processing stream: {e}")
        return []
