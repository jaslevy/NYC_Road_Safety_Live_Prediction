from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import router

app = FastAPI(title="NYC Road Safety Weather API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add router with /api prefix
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "NYC Road Safety Live Prediction API"}
