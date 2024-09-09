from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import os
import json
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
from supabase import create_client, Client

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Global variables
client = AsyncClient(os.getenv('PLAY_HT_USER_ID'), os.getenv('PLAY_HT_API_KEY'))
options = TTSOptions(
    voice="s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json",
    format=api_pb2.FORMAT_WAV,
    quality="standard",
    speed=0.8,
    sample_rate=24000,
    voice_guidance=0.5,
    temperature=0.5
)
api_key =  os.getenv('GROQ_API_KEY')
audio_manager = AudioManager()

class UserSession:
    def __init__(self, email):
        self.email = email
        self.conversation = None
        self.chat_history = []
        self.role = None

active_sessions = {}

@app.on_event("startup")
async def startup_event():
    await initialize_apis(client, options)

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await client.close()
    except AttributeError:
        pass
    audio_manager.terminate()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.post("/submit-email")
async def submit_email(request: Request, email: str = Form(...)):
    try:
        # Insert email into Supabase table
        response = supabase.table("surreal_users").insert({"email": str(email)}).execute()
        
        if response.data:
            print(f"Email inserted successfully: {email}")
            return RedirectResponse(url=f"/chat?email={email}", status_code=303)
        else:
            print(f"Failed to insert email: {email}")
            raise HTTPException(status_code=500, detail="Failed to submit email")
    except Exception as e:
        print(f"Error inserting email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit email: {str(e)}")

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    # Get the email from the query parameters
    email = request.query_params.get("email")
    
    # If no email is provided, redirect to the landing page
    if not email:
        return RedirectResponse(url="/", status_code=303)
    
    # Check if the email exists in the database
    result = supabase.table("surreal_users").select("email").eq("email", email).execute()
    
    # If the email doesn't exist in the database, redirect to the landing page
    if not result.data:
        return RedirectResponse(url="/", status_code=303)
    
    # If email exists, render the chatbot page
    return templates.TemplateResponse("chatbot.html", {"request": request, "email": email})

DEFAULT_PROMPT = """
You are Surreal AI, acting as a {role}. Your responses should be:
- Professional and knowledgeable in the field of {role}
- Accurate and based on current information relevant to the {role}
- Helpful and informative to the best of your abilities
- Clear and easy to understand for the general public
- Ethical and responsible within the context of your role as a {role}
- Cautious about providing specific advice without proper context or qualifications

Provide information and assistance appropriate to your role as a {role}. If asked about topics outside your expertise or role, politely redirect the conversation to areas relevant to your role as a {role}.

Remember, you're providing general information and should encourage users to seek professional, in-person consultation for specific or critical matters related to your role.
"""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_session = None
    try:
        while True:
            data = await websocket.receive_json()
            
            if data['type'] == 'email':
                user_email = data['content']
                if user_email not in active_sessions:
                    user_session = UserSession(user_email)
                    active_sessions[user_email] = user_session
                    # Load existing chat history from Supabase
                    result = supabase.table("surreal_users").select("conversation").eq("email", user_email).execute()
                    if result.data and result.data[0]['conversation']:
                        user_session.chat_history = json.loads(result.data[0]['conversation'])
                        for msg in user_session.chat_history:
                            await websocket.send_json({"type": "history", "content": msg})
                else:
                    user_session = active_sessions[user_email]
                continue

            elif data['type'] == 'system_prompt':
                if user_session:
                    user_session.role = data['content']
                    user_session.conversation = create_conversation_chain(api_key, role=user_session.role)
                continue

            if not user_session:
                await websocket.send_json({"type": "error", "content": "Session not initialized"})
                continue

            if data['type'] in ['text', 'speech']:
                user_input = data['content']
                if user_input.lower() == 'quit':
                    await websocket.send_json({"type": "text", "content": "Goodbye!"})
                    break

                start_time = asyncio.get_event_loop().time()
                try:
                    print("Getting Groq response...")
                    if user_session.conversation is None:
                        if user_session.role is None:
                            user_session.role = "general assistant"
                        user_session.conversation = create_conversation_chain(api_key, role=user_session.role)
                    
                    groq_response = await asyncio.to_thread(get_response, user_session.conversation, user_input)
                    print(f"Groq response: {groq_response}")
                    await websocket.send_json({"type": "text", "content": groq_response})

                    # Add user message to chat history
                    user_session.chat_history.append({"role": "user", "content": user_input})

                    # Add assistant message to chat history
                    user_session.chat_history.append({"role": "assistant", "content": groq_response})

                    # Save chat history to Supabase after each response
                    try:
                        supabase.table("surreal_users").upsert({
                            "email": user_session.email,
                            "conversation": json.dumps(user_session.chat_history)
                        }).execute()
                        print("Chat history saved to Supabase")
                    except Exception as e:
                        print(f"Error saving chat history: {str(e)}")

                    print("Converting text to speech...")
                    audio_data = await asyncio.to_thread(client.tts, groq_response, options)
                    
                    first_audio_byte_time = None
                    async for chunk in async_play_audio(audio_data):
                        if first_audio_byte_time is None:
                            first_audio_byte_time = asyncio.get_event_loop().time()
                            latency = first_audio_byte_time - start_time
                            await websocket.send_json({"type": "info", "content": f"Latency: {latency:.2f} seconds"})
                        await websocket.send_bytes(chunk)
                    
                    await websocket.send_json({"type": "audio_end"})

                    end_time = asyncio.get_event_loop().time()
                    await websocket.send_json({"type": "info", "content": f"Total response time: {end_time - start_time:.2f} seconds"})
                except Exception as e:
                    error_message = f"Error in processing response: {str(e)}"
                    print(error_message)
                    await websocket.send_json({"type": "error", "content": error_message})
                    continue  # Skip the rest of the loop and start over

            else:
                print(f"Unsupported message type: {data['type']}")

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error in WebSocket: {str(e)}")
    finally:
        if user_session:
            del active_sessions[user_session.email]

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

    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)