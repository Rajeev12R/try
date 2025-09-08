# config.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Use the direct Supabase connection string
DB_URL = os.getenv("SUPABASE_DB_URL")


engine = create_engine(DB_URL, echo=True)

# Chroma config
CHROMA_SERVER_URL = os.getenv("CHROMA_SERVER_URL", "http://localhost:8001")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")

# Google GenAI key
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")