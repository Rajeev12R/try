# test_connections.py
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import chromadb

load_dotenv()

# Test database connection
try:
    engine = create_engine(os.getenv("SUPABASE_DB_URL"))
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print("✅ Database connection successful")
        print(f"PostgreSQL version: {result.scalar()}")
except Exception as e:
    print(f"❌ Database connection failed: {e}")

# Test ChromaDB connection
try:
    chroma_client = chromadb.HttpClient(
        host="3614c461-db46-4572-ae67-8ad8e3c7e842-chroma-server.chroma.cloud",
        port=443,
        ssl=True,
        headers={"Authorization": f"Bearer {os.getenv('CHROMA_API_KEY')}"}
    )
    collections = chroma_client.list_collections()
    print("✅ ChromaDB connection successful")
    print(f"Collections: {collections}")
except Exception as e:
    print(f"❌ ChromaDB connection failed: {e}")