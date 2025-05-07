from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the router from endpoints.py
app.include_router(router, prefix="/api")