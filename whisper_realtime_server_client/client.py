import asyncio
import websockets
import sounddevice as sd
import numpy as np

SERVER_URL = "wss://8000-01k30sxs5twvzkfhvr36583w80.cloudspaces.litng.ai/ws"
  # change this if server is remote

loop = asyncio.get_event_loop()  # main event loop


async def send_audio(ws):
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        audio_bytes = indata.tobytes()
        # Send safely from another thread
        asyncio.run_coroutine_threadsafe(ws.send(audio_bytes), loop)

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=callback):
        print("🎙️ Connected to server. Start speaking...")
        await asyncio.Future()  # run forever


async def main():
    async with websockets.connect(SERVER_URL) as ws:
        await send_audio(ws)


if __name__ == "__main__":
    loop.run_until_complete(main())
