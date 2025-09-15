# config.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Use the direct Supabase connection string
DB_URL = os.getenv("SUPABASE_DB_URL")
engine = create_engine(DB_URL, echo=True)

# Local Chroma config - SIMPLIFIED
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db_storage")  # Local folder

# Google GenAI key (optional)
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")