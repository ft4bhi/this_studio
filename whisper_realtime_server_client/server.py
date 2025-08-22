from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load Whisper model (using GPU if available)
model = WhisperModel("base", device="cuda", compute_type="float16")

@app.get("/")
async def root():
    return {"message": "Whisper Realtime Server is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()  # Receive raw PCM16 audio
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, _ = model.transcribe(audio, beam_size=5)
            text = " ".join([segment.text for segment in segments])

            await websocket.send_text(text)

    except Exception as e:
        print("Connection closed:", e)
        await websocket.close()
