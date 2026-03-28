import gradio as gr
import requests
import sounddevice as sd
import numpy as np
import queue
import logging
import threading
from pydub import AudioSegment
import tempfile
import os
import io
import wave


logging.basicConfig(level=logging.INFO)

BASE_URL = "http://127.0.0.1:8000/"

# Global stop event for stream control
stop_event = threading.Event()

def predict_audio(file_path):
    if file_path is None:
        return "Please upload an audio file."
    
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.frame_rate != 48000:
            audio = audio.set_frame_rate(48000)
        if audio.channels != 2:
            audio = audio.set_channels(2)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            print(f"Exported chunk to temporary file: {temp_file.name}")
            print(type(temp_file))
            with open(temp_file.name, "rb") as audio_file:
                files = {"audio": (os.path.basename(temp_file.name), audio_file, "audio/wav")}
                response = requests.post(f"{BASE_URL}predict", files=files)
        
        if response.status_code == 200:
            return f"✅ Prediction: {response.json().get('predicted_class', 'Unknown')}"
        else:
            return f"❌ Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def start_stream(prediction_list):
    global stop_event
    stop_event.clear()  # Reset stop event
    try:
        response = requests.get(f"{BASE_URL}health")
        if response.status_code != 200:
            prediction_list.append("Error: Backend API is not available.")
            yield "\n".join(prediction_list)
            return
    except Exception as e:
        prediction_list.append(f"Error: Could not connect to backend API - {str(e)}")
        yield "\n".join(prediction_list)
        return

    logging.info("Streaming started.")

    
    sample_rate = 48000
    channels = 2
    dtype = "int16"
    bytes_per_sample = 2
    chunk_duration = 3
    chunk_frames = int(sample_rate * chunk_duration)
    chunk_bytes = chunk_frames * channels * bytes_per_sample

    audio_buffer = bytearray()
    chunk_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            logging.error(status)
            prediction_list.append(f"Audio Error: {status}")
        if not stop_event.is_set():
            audio_buffer.extend(indata.tobytes())

    def create_chunks():
        logging.info("Creating audio chunks.")
        while not stop_event.is_set():
            while len(audio_buffer) < chunk_bytes and not stop_event.is_set():
                continue
            if stop_event.is_set():
                break
            chunk = bytes(audio_buffer[:chunk_bytes])
            del audio_buffer[:chunk_bytes]
            chunk_queue.put(chunk)
            logging.info(f"Created chunk of size {len(chunk)} bytes.")

    def send_chunks():
        logging.info("Sending chunks to backend API.")
        while not stop_event.is_set():
            try:
                chunk = chunk_queue.get(timeout=1.0)
                if chunk is None or stop_event.is_set():
                    logging.info("Received None chunk or stop event set. Exiting send_chunks.")
                    break
                
                logging.info(f"Processing chunk of size {len(chunk)} bytes.")
                
                # Verify chunk is not empty
                if not chunk:
                    logging.warning("Empty chunk received. Skipping.")
                    prediction_list.append("Error: Empty chunk received")
                    continue
                
                # Create temporary WAV file from raw PCM data
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        logging.debug(f"Creating temporary WAV file: {temp_file_path}")
                        
                        # Write raw PCM data with WAV header
                        with wave.open(temp_file_path, 'wb') as wav_file:
                            wav_file.setnchannels(channels)
                            wav_file.setsampwidth(bytes_per_sample)
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(chunk)
                            logging.debug(f"Wrote {len(chunk)} bytes as PCM with WAV header")
                        
                        # Send the WAV file to the API with correct field name
                        with open(temp_file_path, "rb") as audio_file:
                            files = {"audio": (os.path.basename(temp_file_path), audio_file, "audio/wav")}  # Changed 'audio_chunk' to 'audio'
                            logging.info(f"Sending processed chunk to {BASE_URL}stream_predict")
                            response = requests.post(f"{BASE_URL}stream_predict", files=files)
                
                    # Process response
                    if response.status_code == 200:
                        data = response.json()
                        prediction_list.append(
                            f"Predicted Class: {data.get('predicted_class', 'Unknown')}"
                        )
                        
                    else:
                        prediction_list.append(
                            f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"
                        )
                
                except Exception as e:
                    logging.error(f"Failed to process or send chunk: {str(e)}")
                    prediction_list.append(f"Error processing chunk: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            logging.debug(f"Deleted temporary file: {temp_file_path}")
                        except Exception as e:
                            logging.error(f"Failed to delete temporary file {temp_file_path}: {str(e)}")
                
                # Keep only the last 10 predictions
                if len(prediction_list) > 10:
                    prediction_list[:] = prediction_list[-10:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Unexpected error in send_chunks: {str(e)}")
                prediction_list.append(f"Error processing chunk: {str(e)}")
                if len(prediction_list) > 10:
                    prediction_list[:] = prediction_list[-10:]
            
            yield "\n".join(prediction_list)

    create_thread = threading.Thread(target=create_chunks, daemon=True)
    send_thread = threading.Thread(target=send_chunks, daemon=True)

    create_thread.start()
    send_thread.start()
    logging.info("Threads started for audio processing.")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype, callback=audio_callback):
            while not stop_event.is_set():
                result = next(send_chunks())
                logging.info(f"Yielding result: {result}")
                yield result
    except Exception as e:
        prediction_list.append(f"Streaming Error: {str(e)}")
        yield "\n".join(prediction_list)
    finally:
        stop_event.set()
        chunk_queue.put(None)
def stop_stream():
    global stop_event
    stop_event.set()
    return "Streaming stopped."

def continuous_predict_audio(file_path):
    print("Starting continuous prediction for uploaded audio file...")
    if file_path is None:
        return "Please upload an audio file."
    
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.frame_rate != 48000:
            audio = audio.set_frame_rate(48000)
        if audio.channels != 2:
            audio = audio.set_channels(2)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)

        print("Exporting audio to temporary file for continuous prediction...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            with open(temp_file.name, "rb") as audio_file:
                files = {"audio": (os.path.basename(temp_file.name), audio_file, "audio/wav")}
                logging.info(f"Sending file {temp_file.name} for continuous prediction...")
                logging.info("file sent")
                response = requests.post(f"{BASE_URL}continuous_predict", files=files)
        
        if response.status_code == 200:
            predictions = response.json().get('predicted_class', [])
            formatted_output = "\n".join([
                f"{start:.1f}-{end:.1f}s: {label} ({confidence:.2f})"
                for start, end, label, confidence in predictions
            ])
            return f"✅ Prediction: {formatted_output}"
        else:
            return f"❌ Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(css="""
#header {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
    color: #1f2937;
}

#subheader {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 30px;
}

#predict-button {
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    color: white;
    font-size: 18px;
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
    transition: background 0.3s ease;
}

#predict-button:hover {
    background: linear-gradient(135deg, #2563eb, #0891b2);
}

#dashboard {
    background-color: #f9fafb;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

#prediction-list {
    height: 300px;
    overflow-y: scroll;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 10px;
    border-radius: 8px;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}
""") as demo:
    
    with gr.Column(elem_id="dashboard"):
        gr.HTML('<div id="header">🎤 Voice Security Dashboard</div>')
        gr.HTML('<div id="subheader">Upload an audio file or stream live audio to predict its class using our machine learning model.</div>')

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(label="🎵 Upload Audio", type="filepath")
                gr.Markdown("Supported formats: `.wav`, `.mp3`, etc.")
            with gr.Column(scale=1):
                predict_button = gr.Button("🔍 Predict", elem_id="predict-button")
        
        output_text = gr.Textbox(label="📋 Prediction Result", interactive=False, lines=2)

        with gr.Accordion("📘 Instructions", open=False):
            gr.Markdown("""
                1. Click on the 'Upload Audio' area to choose your file or start streaming.
                2. Press the 'Predict' button to analyze an uploaded file or 'Start Voice Processing' for live audio.
                3. Press 'Stop Voice Processing' to stop streaming.
                4. Ensure your audio is clear and matches the expected format (48kHz, stereo, 16-bit PCM) for best results.
            """)
        
        gr.Markdown("---")
        gr.Markdown("© 2025 SecureVoice AI | All rights reserved.", elem_id="footer")

    predict_button.click(predict_audio, inputs=[audio_input], outputs=[output_text])

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🎤 Continuous Voice Processing")
#             
            start_button = gr.Button("Start Voice Processing")
#             
            stop_button = gr.Button("Stop Voice Processing")
        with gr.Column(scale=1):
            prediction_list = gr.State([])
            prediction_box = gr.Textbox(label="📜 Continuous Predictions", elem_id="prediction-list", interactive=False, lines=10)
            start_button.click(
                start_stream,
                inputs=[prediction_list],
                outputs=[prediction_box],
                show_progress=True,
                queue=True
            )
            stop_button.click(
                fn=stop_stream,
                outputs=[prediction_box]
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Long Audio for Continuous Prediction")
#             
            audio_file = gr.File(label="Select Audio File", file_count="single", type="filepath")
#             
            upload_button = gr.Button("Upload and Predict")
            continuous_result = gr.Textbox(label="Continuous Prediction Result", lines=2, interactive=False)
            
    upload_button.click(
                fn=continuous_predict_audio,
                inputs=[audio_file],
                outputs=[continuous_result],
                show_progress=True
            )

if __name__ == "__main__":
    demo.launch()