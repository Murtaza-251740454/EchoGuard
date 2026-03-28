from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from main import predict_audio, single_predict_audio
from main import continuous_predict as cp 
import os
from tempfile import NamedTemporaryFile
import uvicorn
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history = pd.read_csv("data/history.csv")
@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    print("in the predict api")
    if not audio.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .wav and .mp3 are supported.")

    try:
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio.read())
            temp_file_path = temp_file.name
        
        print(f"Temporary file created at: {temp_file_path}")
        print(f"File size: {os.path.getsize(temp_file_path)} bytes")

        # Predict the class
        predicted_class, confidence = await single_predict_audio(temp_file_path)
        print(f"Predicted class: {predicted_class}")
        return JSONResponse(content={"predicted_class": predicted_class, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/continuous_predict")
async def continuous_predict(audio: UploadFile = File(...)):
    print("in the continuous_predict api")
    if not audio.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .wav and .mp3 are supported.")

    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio.read())
            temp_file_path = temp_file.name

        # Process the entire audio file (e.g., predict on the whole file or split into chunks)
        predicted_classes = await cp(temp_file_path)  # Await async function
        return JSONResponse(content={"predicted_class": predicted_classes})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/health")
async def health():
    return {"status": "healthy"}
    
@app.post("/stream_predict")
async def stream_predict(audio: UploadFile = File(...)):
    print("in the predict api")
    if not audio.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .wav and .mp3 are supported.")

    try:
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await audio.read())
            temp_file_path = temp_file.name
        
        print(f"Temporary file created at: {temp_file_path}")
        print(f"File size: {os.path.getsize(temp_file_path)} bytes")

        # Predict the class
        predicted_class,confidence = await predict_audio(temp_file_path)
        print(f"Predicted class: {predicted_class}")
        entry = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "timestamp": pd.Timestamp.now()
        }

        try:
            # Debugging: Check if the file exists
            if not os.path.exists("data/history.csv"):  
                print("File does not exist. Creating a new file.")
                history = pd.DataFrame(columns=["predicted_class", "timestamp"])
            else:
                # Read the existing history file
                history = pd.read_csv("data/history.csv")
                print(f"Loaded history file with {len(history)} rows.")

            # Append the new entry using pd.concat
            new_entry_df = pd.DataFrame([entry])  # Convert the entry to a DataFrame
            history = pd.concat([history, new_entry_df], ignore_index=True)
            print(f"Appended new entry: {entry}")

            # Save the updated DataFrame back to the same file
            history.to_csv("data/history.csv", index=False)
            print("Data successfully saved to history.csv.")
        except Exception as e:
            print(f"Error while updating history.csv: {str(e)}")
            
        return JSONResponse(content={"predicted_class": predicted_class,"confidence": confidence, "timestamp": str(entry["timestamp"])})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/history")
async def get_history():
    """
    Retrieve the prediction history.
    """
    try:
        if not os.path.exists("data/history.csv"):
            return JSONResponse(content={"history": []})
        
        history = pd.read_csv("data/history.csv")

        history['timestamp'] = pd.to_datetime(history['timestamp'])
        df_sorted = history.sort_values('timestamp', ascending=False)
        df_sorted['timestamp'] = df_sorted['timestamp'].astype(str)

        history_list = df_sorted.to_dict(orient='records')
        return JSONResponse(content={"history": history_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")
        
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
