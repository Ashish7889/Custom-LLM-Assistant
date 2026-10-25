"""
api/app.py -- FastAPI application factory and entrypoint for FinSage.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config import cfg

app = FastAPI(
    title=cfg.app_name,
    version=cfg.app_version,
    description="Evidence-Grounded Self-Verifying Financial Assistant",
)

# CORS setup for web client and external harnesses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    print(f"Starting {cfg.app_name} v{cfg.app_version} on http://{cfg.host}:{cfg.port}")
    uvicorn.run("api.app:app", host=cfg.host, port=cfg.port, reload=cfg.debug)
