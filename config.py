import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

load_dotenv(".env")

class Settings(BaseSettings):
    TEAM_NAME: str = "Vera AI Team"
    TEAM_MEMBERS: List[str] = ["Vera Developer"]
    MODEL: str = "gemini-flash-latest"
    APPROACH: str = "Context-driven state machine with rule-based routing and LLM composition"
    CONTACT_EMAIL: str = "vera@example.com"
    VERSION: str = "1.0.0"
    SUBMITTED_AT: str = "2026-09-27T14:16:18+05:30"
    
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-flash-latest"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
