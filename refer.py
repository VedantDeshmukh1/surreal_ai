import asyncio
import os
import numpy as np
import sounddevice as sd
import websockets
import json
from pyht import TTSOptions
from contextual_memory import create_conversation_chain, get_response
from pyht.async_client import AsyncClient
from pyht.protos import api_pb2
from groq import Groq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# === SYNC EXAMPLE ===

def play_audio(data):
    buffer = np.frombuffer(data, dtype=np.float16)
    sd.play(buffer, samplerate=24000, channels=1, dtype='float16')
    sd.wait()

def get_groq_response(prompt):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise responses without any formatting."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
    )
    end_time = time.time()
    
    response = chat_completion.choices[0].message.content
    latency = end_time - start_time
    
    print(f"Groq Latency: {latency:.2f} seconds")
    return response

async def initialize_apis(client, options):
    print("Initializing APIs...")
    start_time = time.time()

    # Initialize Groq
    dummy_prompt = "Hello"
    await asyncio.to_thread(get_groq_response, dummy_prompt)

    # Initialize PlayHT
    dummy_text = "Initialization complete"
    async for _ in client.tts(dummy_text, options):
        pass  # We just need to iterate through the generator

    print(f"APIs initialized in {time.time() - start_time:.2f} seconds")

class AudioManager:
    def __init__(self):
        self.input_stream = None

    def create_input_stream(self):
        if self.input_stream is None:
            self.input_stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16')
            self.input_stream.start()
        return self.input_stream

    def close_streams(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

    def terminate(self):
        self.close_streams()

audio_manager = AudioManager()

async def async_get_speech_input():
    url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
    
    stream = audio_manager.create_input_stream()

    print("Listening... (Speak your question)")

    async with websockets.connect(
        url,
        extra_headers={
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"
        },
    ) as ws:
        async def sender(ws):
            try:
                while True:
                    try:
                        data, _ = stream.read(1024)
                        await ws.send(data.tobytes())
                    except OSError as e:
                        print(f"Error reading from audio stream: {e}")
                        break
                    await asyncio.sleep(0.01)  # Add a small delay to prevent overwhelming the buffer
            except asyncio.CancelledError:
                pass
            finally:
                await ws.send(json.dumps({"type": "CloseStream"}))

        async def receiver(ws):
            transcript = ""
            async for msg in ws:
                res = json.loads(msg)
                if res.get("is_final"):
                    transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                    if transcript:
                        print(f"You said: {transcript}")
                        return transcript

        sender_task = asyncio.create_task(sender(ws))
        try:
            transcript = await asyncio.wait_for(receiver(ws), timeout=10)
        except asyncio.TimeoutError:
            transcript = None
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

    return transcript

async def async_play_audio(data):
    buffer = np.array([], dtype=np.int16)
    start_time = time.time()
    stream = None
    first_chunk = True
    total_samples = 0

    async for chunk in data:
        if first_chunk:
            print("First audio byte received in:", time.time() - start_time)
            first_chunk = False
        
        chunk_array = np.frombuffer(chunk, dtype=np.int16)
        buffer = np.append(buffer, chunk_array)
        total_samples += len(chunk_array)

        if len(buffer) >= 3600:  # Start playing after accumulating 0.15 seconds of audio
            if stream is None:
                stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
                stream.start()
            
            stream.write(buffer)
            buffer = np.array([], dtype=np.int16)

        yield chunk  # Yield each chunk as it's received

    # Play any remaining audio
    if len(buffer) > 0 and stream is not None:
        stream.write(buffer)

    if stream:
        stream.stop()
        stream.close()

    print(f"Total audio samples: {total_samples}")
    print(f"Audio duration: {total_samples / 24000:.2f} seconds")
    print("Audio playback completed")

# Remove the main() function and the if __name__ == "__main__" block