from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import fraud, ops

app = FastAPI(title="AI MedGuard API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
def root():
    return {"message": "AI MedGuard API", "version": "1.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "name": "AI MedGuard API",
        "version": "1.0",
        "description": "Healthcare analytics platform for fraud detection, operational monitoring, and compliance",
        "endpoints": ["/health", "/info", "/fraud", "/ops"],
    }


app.include_router(fraud.router)
app.include_router(ops.router)

# Run: uvicorn api.app:app --reload
