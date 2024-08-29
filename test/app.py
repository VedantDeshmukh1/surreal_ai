import os
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # Add this line
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)  # Add this line

app = FastAPI()

# Mount static files
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class UserCredentials(BaseModel):
    email: str
    password: str

@app.middleware("http")
async def add_authentication(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path in ["/", "/signup", "/login"]:
        return await call_next(request)
    token = request.headers.get("authorization", "").replace("Bearer ", "")
    if not token:
        return Response("Unauthorized", status_code=401)
    try:
        auth = supabase.auth.get_user(token)
        request.state.user_id = auth.user.id
        supabase.postgrest.auth(token)
    except Exception:
        return Response("Invalid user token", status_code=401)
    return await call_next(request)

@app.post("/signup")
async def signup(user: UserCredentials):
    try:
        # Sign up the user using the admin client
        response = supabase_admin.auth.admin.create_user({
            "email": user.email,
            "password": user.password,
            "email_confirm": True  # Automatically confirm the email
        })
        
        if response.user is None:
            raise HTTPException(status_code=400, detail="User creation failed")
        
        # Extract username from email
        username = user.email.split('@')[0]
        
        # Insert user data into the 'users' table
        user_data = {
            'user_id': response.user.id,
            'username': username,
            'email': user.email
        }
        supabase_admin.table('users').insert(user_data).execute()
        
        return {"message": "User created successfully", "user": response.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(user: UserCredentials):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        return {"message": "Login successful", "token": response.session.access_token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/protected")
async def protected_route(request: Request):
    return {"message": "This is a protected route", "user_id": request.state.user_id}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)