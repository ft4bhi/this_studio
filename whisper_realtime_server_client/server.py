# server.py
import asyncio
import traceback
import numpy as np
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# ====== CONFIG ======
SAMPLE_RATE = 16000
MIN_SECONDS_PER_INFER = 2.0
MIN_SAMPLES = int(SAMPLE_RATE * MIN_SECONDS_PER_INFER)

# Use environment variables with fallbacks
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ====== APP ======
app = FastAPI()

print("🚀 Starting Whisper Realtime Server...")
print(f"📦 Loading model: {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})")

try:
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Model loaded successfully")
except Exception as e:
    print("💥 Failed to load model:", repr(e))
    print("🔄 Falling back to CPU with int8...")
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✅ Fallback model loaded successfully")
    except Exception as e2:
        print("💥 Fallback also failed:", repr(e2))
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    return {"message": "Whisper Realtime Server is running"}

@app.websocket("/ws")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected:", websocket.client)

    buffer = np.empty(0, dtype=np.float32)
    chunk_count = 0
    total_samples = 0

    try:
        while True:
            try:
                # receive audio
                chunk = await websocket.receive_bytes()
                chunk_count += 1
                print(f"🎧 Received chunk #{chunk_count}, {len(chunk)} bytes")
            except WebSocketDisconnect:
                print("❌ Client disconnected normally")
                break
            except Exception as e:
                print("⚠️ receive_bytes failed:", repr(e))
                break

            if not chunk:
                print("⚠️ Empty chunk received, skipping...")
                continue

            try:
                # convert to float32 [-1,1]
                audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                print("💥 Failed to decode audio chunk:", repr(e))
                continue

            if audio.size == 0:
                print("⚠️ Zero-size audio array after conversion")
                continue

            # append to buffer
            buffer = np.concatenate((buffer, audio))
            total_samples += audio.size
            print(f"📊 Buffer length: {buffer.size} samples, Total received: {total_samples}")

            # decode every ~2s
            if buffer.size >= MIN_SAMPLES:
                to_decode = buffer.copy()
                buffer = np.empty(0, dtype=np.float32)

                def do_transcribe(a: np.ndarray) -> str:
                    try:
                        print(f"🧠 Transcribing {a.size} samples (~{a.size/SAMPLE_RATE:.2f}s)")
                        segments, _ = model.transcribe(
                            a,
                            beam_size=5,
                            vad_filter=True,
                            language=None,
                        )
                        result = " ".join(s.text for s in segments).strip()
                        print(f"✅ Transcription result: '{result}'")
                        return result
                    except Exception as e:
                        print("💥 Transcription error:", repr(e))
                        traceback.print_exc()
                        return ""

                try:
                    text = await asyncio.to_thread(do_transcribe, to_decode)
                    if text:
                        await websocket.send_text(text)
                        print(f"📤 Sent transcript: '{text}'")
                    else:
                        print("🔇 No transcription result (silence or error)")
                except Exception as e:
                    print("💥 Error during transcription or sending:", repr(e))
                    break

    except WebSocketDisconnect:
        print("❌ Client disconnected:", websocket.client)
    except Exception as e:
        print("💥 Unexpected error:", repr(e))
        traceback.print_exc()
    finally:
        print("🔒 Connection cleanup done")