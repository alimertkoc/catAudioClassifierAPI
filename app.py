from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tensorflow as tf
import tempfile
import os

app = FastAPI(title="Cat Meow Classifier API")

# Load the trained model
MODEL_PATH = "./modelv0.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Define label mapping
LABELS = ['Waiting For Food', 'Isolated in unfamiliar Environment', 'Brushing']

# Feature extraction functions
def extract_features(data, sample_rate):
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    
    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    
    # Root Mean Square Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    
    # Combine all features
    features = np.hstack((zcr, chroma_stft, mfcc, rms, mel))
    return features

def preprocess_audio(file_path):
    try:
        # Load audio file
        data, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6, sr=44100)
        
        # Extract features
        features = extract_features(data, sample_rate)
        
        # Reshape for model input
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        
        return features
    except Exception as e:
        print(f"Error in preprocessing audio: {e}")
        raise e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV, MP3, and MPEG are supported.")
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            temp_name = tmp.name
        
        # Preprocess the audio file
        features = preprocess_audio(temp_name)
        
        # Remove the temporary file
        os.remove(temp_name)
        
        # Perform prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = LABELS[predicted_class]
        
        # Prepare the response
        response = {
            "filename": file.filename,
            "predicted_label": predicted_label,
            "confidence": float(np.max(prediction) * 100)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)