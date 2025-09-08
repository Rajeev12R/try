from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from netcdf_processor import process_netcdf
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, desc
from config import engine
from models import Base, ProfileMetadata, Measurement, FloatMetadata
import os
from datetime import datetime
from typing import Optional

app = FastAPI(title="ARGO Data Processing API", 
              description="API for processing ARGO NetCDF files")

SessionLocal = sessionmaker(bind=engine)

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