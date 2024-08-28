from __future__ import annotations
import select
import sys
from typing import AsyncGenerator, AsyncIterable, Generator, Iterable, Literal
import asyncio
import time
import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import pyaudio
import numpy as np
import sounddevice as sd
from pyht import TTSOptions
import simpleaudio as sa
import websockets
import json

from pyht.async_client import AsyncClient
from pyht.protos import api_pb2

from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === SYNC EXAMPLE ===


def play_audio(data: Generator[bytes, None, None] | Iterable[bytes]):
    buff_size = 5242880
    ptr = 0
    start_time = time.time()
    buffer = np.empty(buff_size, np.float16)
    audio = None
    for i, chunk in enumerate(data):
        if i == 0:
            start_time = time.time()
            continue  # Drop the first response, we don't want a header.
        elif i == 1:
            print("First audio byte received in:", time.time() - start_time)
        for sample in np.frombuffer(chunk, np.float16):
            buffer[ptr] = sample
            ptr += 1
        if i == 5:
            # Give a 4 sample worth of breathing room before starting
            # playback
            audio = sa.play_buffer(buffer, 1, 2, 24000)
    approx_run_time = ptr / 24_000
    time.sleep(max(approx_run_time - time.time() + start_time, 0))
    if audio is not None:
        audio.stop()

def get_groq_response(prompt):
    client = Groq(api_key="KEY")
    
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
        self.p = pyaudio.PyAudio()
        self.input_stream = None

    def create_input_stream(self):
        if self.input_stream is None:
            self.input_stream = self.p.open(format=pyaudio.paInt16,
                                            channels=1,
                                            rate=16000,
                                            input=True,
                                            frames_per_buffer=1024)
        return self.input_stream

    def close_streams(self):
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

    def terminate(self):
        self.close_streams()
        self.p.terminate()

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
                        data = await asyncio.to_thread(stream.read, 1024, exception_on_overflow=False)
                        await ws.send(data)
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
async def async_play_audio(data: AsyncGenerator[bytes, None] | AsyncIterable[bytes]):
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
        print(f"Received chunk of size: {len(chunk_array)} samples")

        if len(buffer) >= 7200:  # Start playing after accumulating 0.2 seconds of audio
            if stream is None:
                stream = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
                stream.start()
            
            stream.write(buffer)
            buffer = np.array([], dtype=np.int16)

    # Play any remaining audio
    if len(buffer) > 0 and stream is not None:
        stream.write(buffer)

    if stream:
        stream.stop()
        stream.close()

    print(f"Total audio samples: {total_samples}")
    print(f"Audio duration: {total_samples / 24000:.2f} seconds")
    print("Audio playback completed")

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

from contextual_memory import create_conversation_chain, get_response

async def main():
    # Load values from environment variables
    user = "KEY"
    key = "KEY"
    voice = "s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json"
    quality = "standard"
    interactive = True
    use_speech_input = True

    # Setup the client
    client = AsyncClient(user, key)

    # Set the speech options
    options = TTSOptions(voice=voice, format=api_pb2.FORMAT_WAV, quality=quality, speed=0.8, sample_rate=24000, voice_guidance=0.5, temperature=0.5)

    try:
        # Initialize APIs
        await initialize_apis(client, options)

        # Initialize the conversation chain
        conversation = create_conversation_chain(os.getenv('GROQ_API_KEY'))

        # Maybe play around with an interactive session.
        if interactive:
            print("Starting interactive session.")
            print("Speak your question or say 'quit' to exit.")
            while True:
                if use_speech_input:
                    try:
                        t = await async_get_speech_input()
                        if t is None:
                            print("No speech detected. Please try again.")
                            continue
                        if t.lower() == 'quit':
                            break
                    except Exception as e:
                        print(f"An error occurred during speech input: {e}")
                        continue
                else:
                    t = input("> ")
                    if t.lower() == 'quit':
                        break
                
                start_time = time.time()
                try:
                    # Use the conversation chain instead of direct Groq API call
                    groq_response = await asyncio.to_thread(get_response, conversation, t)
                    print("Groq Response:", groq_response)
                    await async_play_audio(client.tts(groq_response, options))
                    print(f"Total response time: {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    print(f"An error occurred while processing the response: {e}")
                    print(f"Error type: {type(e)}")
                    print(f"Error args: {e.args}")
    finally:
        # Cleanup
        try:
            await client.close()
        except AttributeError:
            # If _stop_lease_loop doesn't exist, we can't stop it, so we'll just pass
            pass
        audio_manager.terminate()

    return 0

# === ASYNC EXAMPLE ===


if __name__ == "__main__":
    asyncio.run(main())