# backend/app/chroma_client.py - MODERN VERSION
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIRECTORY
import os

def get_chroma_client():
    """Get a local ChromaDB client with persistence - Modern version"""
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    # MODERN: Use PersistentClient with latest ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIRECTORY,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    return client

def get_or_create_collection(collection_name="argo_profiles"):
    """Get or create a ChromaDB collection"""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    return collection