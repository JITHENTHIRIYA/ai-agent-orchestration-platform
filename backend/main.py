"""
main.py — FastAPI application entry point.

Creates the FastAPI app instance and registers top-level routes.
Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from config import settings

app = FastAPI(
    title="AntiGravity Backend",
    description="FastAPI backend powering the AntiGravity AI agent platform.",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """
    Health-check endpoint.

    Returns a simple JSON payload confirming the service is running.
    Useful for uptime monitors, container orchestrators, and load balancers.
    """
    return {"status": "ok"}
