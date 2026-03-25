"""
config.py — Application configuration module.

Loads environment variables from a .env file using python-dotenv and exposes
them as a Pydantic settings object for type-safe access throughout the app.
"""

from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load variables from .env file into the process environment
load_dotenv()


class Settings(BaseModel):
    """
    Central configuration container.

    Reads required secrets and connection details from environment variables
    so that sensitive values are never hard-coded in source control.
    """

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "")


# Singleton settings instance used by the rest of the application
settings = Settings()
