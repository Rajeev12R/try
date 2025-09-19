from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from netcdf_processor import process_netcdf
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, desc
from config import engine
from models import Base, ProfileMetadata, Measurement, FloatMetadata
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from chroma_embeddings import vector_store
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

app = FastAPI(title="ARGO Data Processing API", 
              description="API for processing ARGO NetCDF files")

executor = ThreadPoolExecutor(max_workers=4)
SessionLocal = sessionmaker(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    Base.metadata.create_all(bind=engine)

@app.post("/upload-netcdf/")
async def upload_netcdf(file: UploadFile = File(...)):
    """
    Upload and process a NetCDF file containing ARGO float data.
    """
    file_location = None
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save uploaded file
        file_location = f"temp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the NetCDF file
        process_netcdf(file_location)
        
        return JSONResponse(
            status_code=200,
            content={"message": "NetCDF file processed successfully."}
        )
        
    except Exception as e:
        # Clean up on error
        if file_location and os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if file_location and os.path.exists(file_location):
            os.remove(file_location)

@app.get("/profiles/")
def get_profiles(
    platform_number: Optional[str] = Query(None, description="Filter by platform number"),
    min_latitude: Optional[float] = Query(None, description="Minimum latitude"),
    max_latitude: Optional[float] = Query(None, description="Maximum latitude"),
    min_longitude: Optional[float] = Query(None, description="Minimum longitude"),
    max_longitude: Optional[float] = Query(None, description="Maximum longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Number of results to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Retrieve ARGO profiles with filtering options.
    """
    session = SessionLocal()
    try:
        query = session.query(ProfileMetadata)
        
        # Apply filters
        if platform_number:
            query = query.filter(ProfileMetadata.platform_number == platform_number)
        
        if min_latitude is not None:
            query = query.filter(ProfileMetadata.latitude >= min_latitude)
        
        if max_latitude is not None:
            query = query.filter(ProfileMetadata.latitude <= max_latitude)
        
        if min_longitude is not None:
            query = query.filter(ProfileMetadata.longitude >= min_longitude)
        
        if max_longitude is not None:
            query = query.filter(ProfileMetadata.longitude <= max_longitude)
        
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                query = query.filter(ProfileMetadata.juld >= start_dt)
            except ValueError:
                pass
        
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                query = query.filter(ProfileMetadata.juld <= end_dt)
            except ValueError:
                pass
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination
        profiles = query.order_by(desc(ProfileMetadata.juld)).offset(offset).limit(limit).all()
        
        # Convert to dict for JSON response
        result = []
        for profile in profiles:
            result.append({
                "platform_number": profile.platform_number,
                "cycle_number": profile.cycle_number,
                "juld": profile.juld.isoformat() if profile.juld else None,
                "juld_location": profile.juld_location.isoformat() if profile.juld_location else None,
                "latitude": profile.latitude,
                "longitude": profile.longitude,
                "position_qc": profile.position_qc,
                "data_type": profile.data_type,
                "direction": profile.direction,
                "station_parameters": profile.station_parameters,
                "profile_pres_qc": profile.profile_pres_qc,
                "profile_temp_qc": profile.profile_temp_qc,
                "profile_psal_qc": profile.profile_psal_qc
            })
        
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "profiles": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving profiles: {str(e)}")
    finally:
        session.close()

@app.get("/measurements/{platform_number}/{cycle_number}")
def get_measurements(platform_number: str, cycle_number: int):
    """
    Retrieve measurements for a specific profile.
    """
    session = SessionLocal()
    try:
        measurements = session.query(Measurement).filter(
            Measurement.platform_number == platform_number,
            Measurement.cycle_number == cycle_number
        ).order_by(Measurement.pres).all()
        
        result = []
        for measurement in measurements:
            result.append({
                "pres": measurement.pres,
                "pres_qc": measurement.pres_qc,
                "temp": measurement.temp,
                "temp_qc": measurement.temp_qc,
                "temp_adjusted": measurement.temp_adjusted,
                "temp_adjusted_qc": measurement.temp_adjusted_qc,
                "temp_adjusted_error": measurement.temp_adjusted_error,
                "psal": measurement.psal,
                "psal_qc": measurement.psal_qc,
                "psal_adjusted": measurement.psal_adjusted,
                "psal_adjusted_qc": measurement.psal_adjusted_qc,
                "psal_adjusted_error": measurement.psal_adjusted_error
            })
        
        return {
            "platform_number": platform_number,
            "cycle_number": cycle_number,
            "measurement_count": len(result),
            "measurements": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving measurements: {str(e)}")
    finally:
        session.close()

@app.get("/floats/")
def get_floats(
    limit: int = Query(100, description="Number of results to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Retrieve list of ARGO floats with basic metadata.
    """
    session = SessionLocal()
    try:
        query = session.query(FloatMetadata)
        total_count = query.count()
        
        floats = query.offset(offset).limit(limit).all()
        
        result = []
        for float_meta in floats:
            result.append({
                "platform_number": float_meta.platform_number,
                "wmo_inst_type": float_meta.wmo_inst_type,
                "platform_type": float_meta.platform_type,
                "float_serial_no": float_meta.float_serial_no,
                "firmware_version": float_meta.firmware_version,
                "project_name": float_meta.project_name,
                "pi_name": float_meta.pi_name,
                "data_centre": float_meta.data_centre,
                "dc_reference": float_meta.dc_reference,
                "data_state_indicator": float_meta.data_state_indicator,
                "data_mode": float_meta.data_mode,
                "positioning_system": float_meta.positioning_system,
                "vertical_sampling_scheme": float_meta.vertical_sampling_scheme
            })
        
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "floats": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving floats: {str(e)}")
    finally:
        session.close()
    
@app.post("/vector-index/")
async def rebuild_vector_index(background_tasks: BackgroundTasks):
    """
    Rebuild the vector index from the database (runs in background)
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    # Run indexing in background
    background_tasks.add_task(vector_store.index_profiles)
    
    return {"message": "Vector index rebuild started in background. This may take several minutes."}

@app.get("/vector-search/")
def vector_search(
    query: str = Query(..., description="Semantic search query"),
    n_results: int = Query(10, description="Number of results to return"),
    platform_number: Optional[str] = Query(None, description="Filter by platform number"),
    min_latitude: Optional[float] = Query(None, description="Minimum latitude"),
    max_latitude: Optional[float] = Query(None, description="Maximum latitude"),
    min_longitude: Optional[float] = Query(None, description="Minimum longitude"),
    max_longitude: Optional[float] = Query(None, description="Maximum longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    ocean_region: Optional[str] = Query(None, description="Filter by ocean region"),
    min_temperature: Optional[float] = Query(None, description="Minimum average temperature"),
    max_temperature: Optional[float] = Query(None, description="Maximum average temperature"),
    min_salinity: Optional[float] = Query(None, description="Minimum average salinity"),
    max_salinity: Optional[float] = Query(None, description="Maximum average salinity"),
    data_mode: Optional[str] = Query(None, description="Filter by data mode (R, A, D)"),
):
    """
    Advanced semantic search for ARGO profiles with comprehensive filtering
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        # Build advanced filters
        filters = {}
        
        if platform_number:
            filters["platform_number"] = platform_number
        
        # Geographic filters
        lat_filter = {}
        if min_latitude is not None:
            lat_filter["$gte"] = min_latitude
        if max_latitude is not None:
            lat_filter["$lte"] = max_latitude
        if lat_filter:
            filters["latitude"] = lat_filter
        
        lon_filter = {}
        if min_longitude is not None:
            lon_filter["$gte"] = min_longitude
        if max_longitude is not None:
            lon_filter["$lte"] = max_longitude
        if lon_filter:
            filters["longitude"] = lon_filter
        
        # Date filter
        if start_date or end_date:
            try:
                date_filter = {}
                if start_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    date_filter["$gte"] = start_dt.isoformat()
                if end_date:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    date_filter["$lte"] = end_dt.isoformat()
                filters["juld"] = date_filter
            except ValueError:
                pass
        
        # Ocean region filter
        if ocean_region:
            filters["ocean_region"] = ocean_region
        
        # Measurement filters
        temp_filter = {}
        if min_temperature is not None:
            temp_filter["$gte"] = min_temperature
        if max_temperature is not None:
            temp_filter["$lte"] = max_temperature
        if temp_filter:
            filters["avg_temperature"] = temp_filter
        
        salinity_filter = {}
        if min_salinity is not None:
            salinity_filter["$gte"] = min_salinity
        if max_salinity is not None:
            salinity_filter["$lte"] = max_salinity
        if salinity_filter:
            filters["avg_salinity"] = salinity_filter
        
        # Data mode filter
        if data_mode:
            filters["data_mode"] = data_mode
        
        # Perform search
        results = vector_store.search_profiles(
            query, 
            n_results, 
            filters if filters else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Enhance results with additional database information
        enhanced_results = []
        session = SessionLocal()
        try:
            for result in results:
                platform, cycle = result['id'].split('_')
                
                # Get additional measurements from database
                measurements = session.query(Measurement).filter(
                    Measurement.platform_number == platform,
                    Measurement.cycle_number == int(cycle)
                ).order_by(Measurement.pres).all()
                
                # Add measurement data to result
                result['measurements'] = [{
                    "pres": m.pres,
                    "temp": m.temp,
                    "psal": m.psal,
                    "pres_qc": m.pres_qc,
                    "temp_qc": m.temp_qc,
                    "psal_qc": m.psal_qc
                } for m in measurements]
                
                enhanced_results.append(result)
        finally:
            session.close()
        
        return {
            "query": query,
            "filters": filters,
            "result_count": len(enhanced_results),
            "results": enhanced_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing vector search: {str(e)}")

@app.get("/similar-profiles/{profile_id}")
def get_similar_profiles(
    profile_id: str,
    n_results: int = Query(5, description="Number of similar profiles to find")
):
    """
    Find profiles similar to a given profile ID
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        similar = vector_store.get_similar_profiles(profile_id, n_results)
        return {
            "profile_id": profile_id,
            "similar_profiles": similar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar profiles: {str(e)}")

@app.get("/vector-stats/")
def get_vector_stats():
    """
    Get comprehensive statistics about the vector store
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting vector stats: {str(e)}")

@app.get("/vector-profile/{profile_id}")
def get_vector_profile(profile_id: str):
    """
    Get a specific profile from the vector store by ID
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        profile = vector_store.get_profile_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found in vector store")
        
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving profile: {str(e)}")

@app.get("/ocean-regions/")
def get_ocean_regions():
    """
    Get list of available ocean regions in the vector store
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        # Get unique ocean regions from metadata
        results = vector_store.collection.get(include=["metadatas"])
        regions = set()
        
        for metadata in results['metadatas']:
            if 'ocean_region' in metadata:
                regions.add(metadata['ocean_region'])
        
        return {"ocean_regions": sorted(list(regions))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ocean regions: {str(e)}")


@app.get("/")
def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "ARGO Data Processing API",
        "endpoints": {
            "upload_netcdf": "/upload-netcdf/",
            "get_profiles": "/profiles/",
            "get_measurements": "/measurements/{platform_number}/{cycle_number}",
            "get_floats": "/floats/"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)