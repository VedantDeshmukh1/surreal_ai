from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv
from refer import (
    AsyncClient,
    TTSOptions,
    api_pb2,
    get_groq_response,
    async_play_audio,
    AudioManager,
    async_get_speech_input,
    initialize_apis
)
from contextual_memory import create_conversation_chain, get_response

load_dotenv()

app = FastAPI()

# Mount static files


# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Global variables
client = AsyncClient("USERID", "KEY")
options = TTSOptions(
    voice="s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json",
    format=api_pb2.FORMAT_WAV,
    quality="standard",
    speed=0.8,
    sample_rate=24000,
    voice_guidance=0.5,
    temperature=0.5
)
api_key = "GROQ key"
conversation = create_conversation_chain(api_key)
audio_manager = AudioManager()

class Message(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    await initialize_apis(client, options)

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await client.close()
    except AttributeError:
        # If _stop_lease_loop doesn't exist, we can't stop it, so we'll just pass
        pass
    audio_manager.terminate()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data.lower() == 'quit':
                await websocket.send_json({"type": "text", "content": "Goodbye!"})
                break

            start_time = asyncio.get_event_loop().time()
            try:
                groq_response = await asyncio.to_thread(get_response, conversation, data)
                await websocket.send_json({"type": "text", "content": groq_response})

                # Convert text to speech
                audio_data = client.tts(groq_response, options)
                async for chunk in audio_data:
                    await websocket.send_bytes(chunk)
                
                # Send a message to indicate end of audio stream
                await websocket.send_json({"type": "audio_end"})

                end_time = asyncio.get_event_loop().time()
                await websocket.send_json({"type": "info", "content": f"Total response time: {end_time - start_time:.2f} seconds"})
            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.websocket("/speech")
async def speech_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data['action'] == 'start_listening':
                await websocket.send_json({"type": "info", "content": "Listening... (Speak your question)"})
                transcript = await async_get_speech_input()
                
                if transcript is None:
                    await websocket.send_json({"type": "error", "content": "No speech detected. Please try again."})
                    continue
                
                if transcript.lower() == 'quit':
                    await websocket.send_json({"type": "text", "content": "Goodbye!"})
                    break

                await websocket.send_json({"type": "text", "content": f"You said: {transcript}"})

                start_time = asyncio.get_event_loop().time()
                try:
                    groq_response = await asyncio.to_thread(get_response, conversation, transcript)
                    await websocket.send_json({"type": "text", "content": groq_response})

                    # Convert text to speech
                    audio_data = client.tts(groq_response, options)
                    async for chunk in audio_data:
                        await websocket.send_bytes(chunk)
                    
                    # Send a message to indicate end of audio stream
                    await websocket.send_json({"type": "audio_end"})

                    end_time = asyncio.get_event_loop().time()
                    await websocket.send_json({"type": "info", "content": f"Total response time: {end_time - start_time:.2f} seconds"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "content": str(e)})

    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
