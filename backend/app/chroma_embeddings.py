import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from sqlalchemy.orm import sessionmaker
from config import engine
from models import ProfileMetadata, FloatMetadata, Measurement
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import re
import logging
from tqdm import tqdm
import asyncio
import aiohttp
import backoff

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                allow_reset=True, 
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Use the default embedding function (more lightweight)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="argo_data",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "description": "ARGO float profile data"}
        )
        
        self.session_factory = sessionmaker(bind=engine)
    
    def generate_rich_semantic_summary(self, profile: ProfileMetadata, float_meta: FloatMetadata, measurements: List[Measurement]) -> str:
        """Generate a comprehensive semantic summary with measurement statistics"""
        summary_parts = []
        
        # Basic information
        summary_parts.append(f"ARGO float {profile.platform_number}, cycle {profile.cycle_number}")
        
        # Location and time with more context
        if profile.latitude and profile.longitude:
            lat_dir = "N" if profile.latitude >= 0 else "S"
            lon_dir = "E" if profile.longitude >= 0 else "W"
            summary_parts.append(f"located at {abs(profile.latitude):.4f}°{lat_dir}, {abs(profile.longitude):.4f}°{lon_dir}")
        
        if profile.juld:
            date_str = profile.juld.strftime("%B %d, %Y")
            summary_parts.append(f"on {date_str}")
        
        # Float metadata with more details
        if float_meta.platform_type:
            summary_parts.append(f"Platform type: {float_meta.platform_type}")
        
        if float_meta.project_name:
            summary_parts.append(f"Research project: {float_meta.project_name}")
        
        if float_meta.pi_name:
            summary_parts.append(f"Principal Investigator: {float_meta.pi_name}")
        
        if float_meta.data_centre:
            summary_parts.append(f"Data center: {float_meta.data_centre}")
        
        # Measurement statistics
        if measurements:
            pres_values = [m.pres for m in measurements if m.pres is not None]
            temp_values = [m.temp for m in measurements if m.temp is not None]
            psal_values = [m.psal for m in measurements if m.psal is not None]
            
            if pres_values:
                summary_parts.append(f"Pressure range: {min(pres_values):.1f} to {max(pres_values):.1f} dbar")
            
            if temp_values:
                summary_parts.append(f"Temperature range: {min(temp_values):.2f} to {max(temp_values):.2f} °C")
            
            if psal_values:
                summary_parts.append(f"Salinity range: {min(psal_values):.3f} to {max(psal_values):.3f} PSU")
        
        # Data quality information
        quality_info = []
        if profile.profile_pres_qc and profile.profile_pres_qc != '0':
            quality_info.append(f"pressure({profile.profile_pres_qc})")
        
        if profile.profile_temp_qc and profile.profile_temp_qc != '0':
            quality_info.append(f"temperature({profile.profile_temp_qc})")
        
        if profile.profile_psal_qc and profile.profile_psal_qc != '0':
            quality_info.append(f"salinity({profile.profile_psal_qc})")
        
        if quality_info:
            summary_parts.append(f"Quality flags: {', '.join(quality_info)}")
        
        # Measurement parameters
        if profile.station_parameters:
            params = [p for p in profile.station_parameters if p and p.strip()]
            if params:
                summary_parts.append(f"Measured parameters: {', '.join(params)}")
        
        return ". ".join(summary_parts)
    
    def extract_comprehensive_metadata(self, profile: ProfileMetadata, float_meta: FloatMetadata, measurements: List[Measurement]) -> Dict[str, Any]:
        """Extract comprehensive metadata for filtering and analysis"""
        metadata = {
            "platform_number": profile.platform_number,
            "cycle_number": int(profile.cycle_number) if profile.cycle_number else 0,
            "latitude": float(profile.latitude) if profile.latitude else 0.0,
            "longitude": float(profile.longitude) if profile.longitude else 0.0,
            "juld": profile.juld.isoformat() if profile.juld else "",
            "juld_timestamp": profile.juld.timestamp() if profile.juld else 0,
            "data_type": profile.data_type or "",
            "direction": profile.direction or "",
            "platform_type": float_meta.platform_type or "",
            "project_name": float_meta.project_name or "",
            "pi_name": float_meta.pi_name or "",
            "data_centre": float_meta.data_centre or "",
            "data_mode": float_meta.data_mode or "",
            "firmware_version": float_meta.firmware_version or "",
            "position_qc": profile.position_qc or "",
            "profile_pres_qc": profile.profile_pres_qc or "",
            "profile_temp_qc": profile.profile_temp_qc or "",
            "profile_psal_qc": profile.profile_psal_qc or "",
        }
        
        # Add measurement statistics
        if measurements:
            pres_values = [m.pres for m in measurements if m.pres is not None]
            temp_values = [m.temp for m in measurements if m.temp is not None]
            psal_values = [m.psal for m in measurements if m.psal is not None]
            
            metadata["max_pressure"] = max(pres_values) if pres_values else 0.0
            metadata["min_pressure"] = min(pres_values) if pres_values else 0.0
            metadata["avg_temperature"] = sum(temp_values)/len(temp_values) if temp_values else 0.0
            metadata["avg_salinity"] = sum(psal_values)/len(psal_values) if psal_values else 0.0
            metadata["measurement_count"] = len(measurements)
        
        # Add station parameters as a list
        if profile.station_parameters:
            metadata["parameters"] = [p for p in profile.station_parameters if p and p.strip()]
        else:
            metadata["parameters"] = []
        
        # Add geographic region information
        if profile.latitude and profile.longitude:
            metadata["ocean_region"] = self._get_ocean_region(profile.latitude, profile.longitude)
            metadata["hemisphere"] = "Northern" if profile.latitude >= 0 else "Southern"
        
        return metadata
    
    def _get_ocean_region(self, latitude: float, longitude: float) -> str:
        """Determine ocean region based on coordinates"""
        if -60 <= latitude <= 60:  # Tropical and temperate regions
            if 20 <= longitude <= 160:
                return "Western Pacific"
            elif 160 <= longitude <= 220:
                return "Central Pacific"
            elif 220 <= longitude <= 300:
                return "Eastern Pacific"
            elif 300 <= longitude <= 360 or 0 <= longitude <= 20:
                return "Atlantic Ocean"
            elif 20 <= longitude <= 100:
                return "Indian Ocean"
        elif latitude > 60:
            return "Arctic Ocean"
        elif latitude < -60:
            return "Southern Ocean"
        
        return "Global Ocean"
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def index_profiles(self, batch_size: int = 50, max_workers: int = 4):
        """Index all profiles in the database with retry logic and batching"""
        logger.info("Starting profile indexing...")
        
        # First, reset the collection to start fresh
        try:
            self.client.reset()
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.warning(f"Could not reset vector store: {e}")
        
        self.collection = self.client.get_or_create_collection(
            name="argo_data",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "description": "ARGO float profile data"}
        )
        
        session = self.session_factory()
        try:
            # Get all profiles with their float metadata and measurements
            profiles = session.query(ProfileMetadata).join(
                FloatMetadata, ProfileMetadata.platform_number == FloatMetadata.platform_number
            ).all()
            
            logger.info(f"Found {len(profiles)} profiles to index")
            
            documents = []
            metadatas = []
            ids = []
            
            for i, profile in enumerate(tqdm(profiles, desc="Indexing profiles")):
                float_meta = profile.float_metadata
                
                # Get measurements for this profile
                measurements = session.query(Measurement).filter(
                    Measurement.platform_number == profile.platform_number,
                    Measurement.cycle_number == profile.cycle_number
                ).all()
                
                # Generate semantic summary
                summary = self.generate_rich_semantic_summary(profile, float_meta, measurements)
                
                # Extract comprehensive metadata
                metadata = self.extract_comprehensive_metadata(profile, float_meta, measurements)
                
                # Create unique ID
                profile_id = f"{profile.platform_number}_{profile.cycle_number}"
                
                documents.append(summary)
                metadatas.append(metadata)
                ids.append(profile_id)
                
                # Add to collection in batches
                if len(documents) >= batch_size:
                    try:
                        self.collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        documents = []
                        metadatas = []
                        ids = []
                    except Exception as e:
                        logger.error(f"Error adding batch {i//batch_size}: {e}")
                        # Retry this batch
                        continue
            
            # Add any remaining documents
            if documents:
                try:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    logger.error(f"Error adding final batch: {e}")
            
            logger.info(f"Successfully indexed {len(profiles)} profiles!")
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise
        finally:
            session.close()
    
    def search_profiles(self, query: str, n_results: int = 10, filters: Dict = None, 
                       include: List[str] = ["documents", "metadatas", "distances"]) -> List[Dict]:
        """Search for profiles based on semantic similarity with advanced options"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=include
            )
            
            # Format results in a structured way
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i] if 'distances' in include else None,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in include else {},
                    'document': results['documents'][0][i] if 'documents' in include else ""
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_similar_profiles(self, profile_id: str, n_results: int = 5) -> List[Dict]:
        """Find similar profiles to a given profile ID"""
        try:
            results = self.collection.get(ids=[profile_id])
            if not results['documents']:
                return []
            
            # Use the first document as query
            similar = self.collection.query(
                query_texts=results['documents'][:1],
                n_results=n_results + 1,  # +1 to exclude self
                where={"platform_number": {"$ne": profile_id.split('_')[0]}}  # Exclude same float
            )
            
            # Filter out the original profile
            formatted_results = []
            for i in range(len(similar['ids'][0])):
                if similar['ids'][0][i] != profile_id:
                    result = {
                        'id': similar['ids'][0][i],
                        'score': 1 - similar['distances'][0][i],
                        'metadata': similar['metadatas'][0][i],
                        'document': similar['documents'][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar profiles: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about the vector store"""
        try:
            count = self.collection.count()
            # Get sample of metadata to understand structure
            sample = self.collection.get(limit=1)
            metadata_keys = list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
            
            return {
                "count": count,
                "name": self.collection.name,
                "metadata_fields": metadata_keys,
                "persistent": True
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def get_profile_by_id(self, profile_id: str) -> Optional[Dict]:
        """Get a specific profile by its ID"""
        try:
            results = self.collection.get(ids=[profile_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'metadata': results['metadatas'][0],
                    'document': results['documents'][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting profile {profile_id}: {e}")
            return None

# Global instance with error handling
try:
    vector_store = VectorStoreManager()
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None