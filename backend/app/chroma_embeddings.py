import os
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    USE_GOOGLE_EMBEDDINGS = True
except ImportError:
    USE_GOOGLE_EMBEDDINGS = False

try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False

# ChromaDB Cloud configuration
import chromadb
from config import CHROMA_SERVER_URL, CHROMA_API_KEY, GENAI_API_KEY
from models import ProfileMetadata, Measurement, FloatMetadata
from sqlalchemy.orm import sessionmaker, joinedload
from config import engine
import tqdm
from datetime import datetime
import numpy as np

SessionLocal = sessionmaker(bind=engine)

def get_chroma_client():
    """Get ChromaDB client based on configuration"""
    if "chroma.cloud" in CHROMA_SERVER_URL:
        # ChromaDB Cloud configuration
        return chromadb.HttpClient(
            host=CHROMA_SERVER_URL.replace("https://", "").replace("http://", ""),
            port=443 if "https" in CHROMA_SERVER_URL else 80,
            ssl="https" in CHROMA_SERVER_URL,
            headers={"Authorization": f"Bearer {CHROMA_API_KEY}"} if CHROMA_API_KEY else {}
        )
    else:
        # Local ChromaDB server
        return chromadb.HttpClient(host=CHROMA_SERVER_URL)

def get_embedding_model():
    """Get the appropriate embedding model based on available packages"""
    if USE_GOOGLE_EMBEDDINGS and GENAI_API_KEY:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GENAI_API_KEY
        )
    elif USE_SENTENCE_TRANSFORMERS:
        return SentenceTransformer('all-MiniLM-L6-v2')
    else:
        raise ImportError("No embedding model available. Install either langchain-google-genai or sentence-transformers")

def generate_embeddings():
    session = SessionLocal()
    
    try:
        # Query profiles with related data using proper relationships
        profiles = session.query(ProfileMetadata).options(
            joinedload(ProfileMetadata.measurements),
            joinedload(ProfileMetadata.float_metadata)
        ).all()

        if not profiles:
            print("No profiles found in database. Upload some NetCDF files first.")
            return

        # Chroma client
        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection(name="argo_profiles")

        # Embedding model
        embedding_model = get_embedding_model()

        print(f"Processing {len(profiles)} profiles...")

        for profile in tqdm.tqdm(profiles):
            # Get basic profile information
            platform_number = profile.platform_number
            cycle_number = profile.cycle_number
            date = profile.juld if profile.juld else profile.juld_location
            latitude = profile.latitude
            longitude = profile.longitude
            direction = profile.direction
            
            # Get float metadata through relationship
            float_meta = profile.float_metadata
            platform_type = float_meta.platform_type if float_meta else None
            project_name = float_meta.project_name if float_meta else None
            data_centre = float_meta.data_centre if float_meta else None
            
            # Get measurements through relationship
            measurements = profile.measurements
            if measurements:
                # Count valid measurements
                temp_count = sum(1 for m in measurements if m.temp is not None)
                psal_count = sum(1 for m in measurements if m.psal is not None)
                pres_vals = [m.pres for m in measurements if m.pres is not None]
                pres_min = min(pres_vals) if pres_vals else None
                pres_max = max(pres_vals) if pres_vals else None
                
                # Calculate average values if available
                temp_vals = [m.temp for m in measurements if m.temp is not None]
                temp_avg = sum(temp_vals) / len(temp_vals) if temp_vals else None
                
                psal_vals = [m.psal for m in measurements if m.psal is not None]
                psal_avg = sum(psal_vals) / len(psal_vals) if psal_vals else None
            else:
                temp_count = psal_count = 0
                pres_min = pres_max = temp_avg = psal_avg = None
            
            # Create a comprehensive summary for embedding
            summary_text = f"""
            ARGO float {platform_number} cycle {cycle_number} in {project_name if project_name else 'unknown project'}.
            Profile collected on {date.strftime('%Y-%m-%d') if date else 'unknown date'} at location {latitude:.2f}°N, {longitude:.2f}°E.
            Platform type: {platform_type if platform_type else 'unknown'}, direction: {direction if direction else 'unknown'}.
            Data processed by {data_centre if data_centre else 'unknown'} data center.
            Profile contains {len(measurements) if measurements else 0} measurements with:
            - {temp_count} temperature measurements{', average: ' + f'{temp_avg:.2f}°C' if temp_avg else ''}
            - {psal_count} salinity measurements{', average: ' + f'{psal_avg:.2f} PSU' if psal_avg else ''}
            Pressure range: {pres_min:.1f} to {pres_max:.1f} dbar.
            """
            
            # Create metadata for ChromaDB
            metadata = {
                "platform_number": platform_number,
                "cycle_number": str(cycle_number) if cycle_number else "unknown",
                "date": date.strftime('%Y-%m-%d') if date else "unknown",
                "latitude": str(latitude) if latitude else "unknown",
                "longitude": str(longitude) if longitude else "unknown",
                "direction": direction if direction else "unknown",
                "platform_type": platform_type if platform_type else "unknown",
                "project_name": project_name if project_name else "unknown",
                "data_centre": data_centre if data_centre else "unknown",
                "temperature_measurements": str(temp_count),
                "salinity_measurements": str(psal_count),
                "pressure_range": f"{pres_min:.1f}-{pres_max:.1f} dbar" if pres_min and pres_max else "unknown",
                "average_temperature": f"{temp_avg:.2f}°C" if temp_avg else "unknown",
                "average_salinity": f"{psal_avg:.2f} PSU" if psal_avg else "unknown"
            }
            
            # Generate embedding based on available model
            if isinstance(embedding_model, SentenceTransformer):
                vector = embedding_model.encode(summary_text).tolist()
            else:
                vector = embedding_model.embed_query(summary_text)
            
            # Create a unique ID
            profile_id = f"{platform_number}_cycle_{cycle_number}"
            
            # Add to ChromaDB
            try:
                collection.add(
                    ids=[profile_id],
                    embeddings=[vector],
                    metadatas=[metadata],
                    documents=[summary_text]
                )
            except Exception as e:
                print(f"Warning: Failed to add profile {profile_id} to ChromaDB: {e}")
                continue

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise
    finally:
        session.close()
    
    print("Embeddings generated and stored in ChromaDB!")

def query_similar_profiles(query_text, n_results=5):
    """Query ChromaDB for similar profiles based on natural language query"""
    try:
        # Chroma client
        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection(name="argo_profiles")
        
        # Embedding model
        embedding_model = get_embedding_model()
        
        # Generate embedding for query
        if isinstance(embedding_model, SentenceTransformer):
            query_vector = embedding_model.encode(query_text).tolist()
        else:
            query_vector = embedding_model.embed_query(query_text)
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        return results
        
    except Exception as e:
        print(f"Error querying similar profiles: {e}")
        raise

if __name__ == "__main__":
    print("Testing embedding generation...")
    generate_embeddings()
    
    # Example query
    print("\nTesting similarity search...")
    try:
        results = query_similar_profiles("Show me salinity profiles near the equator in March", n_results=3)
        print("Similar profiles found:", results)
    except Exception as e:
        print(f"Error in similarity search: {e}")