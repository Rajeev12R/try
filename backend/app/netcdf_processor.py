import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker, joinedload
from sqlalchemy import func
from config import engine
from models import Base, FloatMetadata, ProfileMetadata, Measurement, Calibration, ProcessingHistory
from tqdm import tqdm
import datetime
import chromadb
from chromadb.config import Settings
import os
from sentence_transformers import SentenceTransformer

SessionLocal = sessionmaker(bind=engine)

# Initialize ChromaDB and embedding model
CHROMA_PERSIST_DIRECTORY = "chroma_db_storage"
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# ChromaDB initialization - lazy loading
chroma_client = None
collection = None

def get_chroma_client():
    """Lazy load ChromaDB client"""
    global chroma_client
    if chroma_client is None:
        print("Initializing ChromaDB client...")
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print("ChromaDB client initialized!")
    return chroma_client

def get_collection():
    """Get or create ChromaDB collection"""
    global collection
    if collection is None:
        print("Getting/creating ChromaDB collection...")
        client = get_chroma_client()
        collection = client.get_or_create_collection(name="argo_profiles")
        print("Collection ready!")
    return collection

# MODERN: Lazy loading of sentence-transformers to avoid blocking server startup
embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model only when needed"""
    global embedding_model
    if embedding_model is None:
        print("Loading sentence-transformers model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    return embedding_model

def convert_julian_day(julian_day, reference_date="1950-01-01"):
    """Convert Julian day to datetime"""
    if pd.isna(julian_day) or julian_day == 999999.0:
        return None
    try:
        base_date = pd.Timestamp(reference_date)
        return base_date + pd.Timedelta(days=julian_day)
    except:
        return None

def safe_decode(value):
    """Safely decode bytes to string or return original value"""
    if isinstance(value, bytes):
        return value.decode('utf-8').strip()
    elif isinstance(value, str):
        return value.strip()
    elif pd.isna(value) or value in [99999.0, 999999.0]:
        return None
    else:
        return str(value).strip()

def safe_float(value):
    """Safely convert to float, handling missing values"""
    if pd.isna(value) or value in [99999.0, 999999.0]:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_datetime(value):
    """Safely convert to datetime, handling NaT and invalid values - FIXED VERSION"""
    if pd.isna(value) or value in [99999.0, 999999.0]:
        return None
    try:
        if isinstance(value, (pd.Timestamp, datetime.datetime)):
            # Check if it's NaT specifically
            if pd.isna(value):
                return None
            return value
        elif isinstance(value, str):
            if value.strip() == '' or value.strip().upper() == 'NAT':
                return None
            result = pd.to_datetime(value, errors='coerce')
            return None if pd.isna(result) else result
        else:
            result = pd.to_datetime(str(value), errors='coerce')
            return None if pd.isna(result) else result
    except (ValueError, TypeError):
        return None

def create_profile_summary(profile, measurements, float_meta):
    """Create a comprehensive text summary for a profile for embedding"""
    # Get basic profile information
    platform_number = profile.platform_number
    cycle_number = profile.cycle_number
    date = profile.juld if profile.juld else profile.juld_location
    latitude = profile.latitude
    longitude = profile.longitude
    direction = profile.direction
    
    # Get float metadata
    platform_type = float_meta.platform_type if float_meta else None
    project_name = float_meta.project_name if float_meta else None
    data_centre = float_meta.data_centre if float_meta else None
    
    # Calculate statistics from measurements
    if measurements and len(measurements) > 0:
        temp_vals = [m.temp for m in measurements if m.temp is not None]
        psal_vals = [m.psal for m in measurements if m.psal is not None]
        pres_vals = [m.pres for m in measurements if m.pres is not None]
        
        temp_avg = sum(temp_vals) / len(temp_vals) if temp_vals else None
        psal_avg = sum(psal_vals) / len(psal_vals) if psal_vals else None
        pres_min = min(pres_vals) if pres_vals else None
        pres_max = max(pres_vals) if pres_vals else None
        temp_count = len(temp_vals)
        psal_count = len(psal_vals)
    else:
        temp_avg = psal_avg = pres_min = pres_max = None
        temp_count = psal_count = 0
    
    # Safely format coordinates - handle None values
    lat_str = f"{latitude:.2f}" if latitude is not None else "unknown"
    lon_str = f"{longitude:.2f}" if longitude is not None else "unknown"
    
    # Safely format pressure range
    if pres_min is not None and pres_max is not None:
        pressure_range = f"{pres_min:.1f} to {pres_max:.1f} dbar"
    else:
        pressure_range = "unknown range"
    
    # Create a comprehensive summary for embedding
    summary_text = f"""
    ARGO float {platform_number} cycle {cycle_number} in {project_name if project_name else 'unknown project'}.
    Profile collected on {date.strftime('%Y-%m-%d') if date else 'unknown date'} at location {lat_str}°N, {lon_str}°E.
    Platform type: {platform_type if platform_type else 'unknown'}, direction: {direction if direction else 'unknown'}.
    Data processed by {data_centre if data_centre else 'unknown'} data center.
    Profile contains {len(measurements) if measurements else 0} measurements with:
    - {temp_count} temperature measurements{', average: ' + f'{temp_avg:.2f}°C' if temp_avg else ''}
    - {psal_count} salinity measurements{', average: ' + f'{psal_avg:.2f} PSU' if psal_avg else ''}
    Pressure range: {pressure_range}.
    """
    
    return summary_text, {
        "platform_number": platform_number,
        "cycle_number": str(cycle_number) if cycle_number else "unknown",
        "date": date.strftime('%Y-%m-%d') if date else "unknown",
        "latitude": str(latitude) if latitude is not None else "unknown",
        "longitude": str(longitude) if longitude is not None else "unknown",
        "direction": direction if direction else "unknown",
        "platform_type": platform_type if platform_type else "unknown",
        "project_name": project_name if project_name else "unknown",
        "data_centre": data_centre if data_centre else "unknown",
        "temperature_measurements": str(temp_count),
        "salinity_measurements": str(psal_count),
        "pressure_range": f"{pres_min:.1f}-{pres_max:.1f} dbar" if pres_min is not None and pres_max is not None else "unknown",
        "average_temperature": f"{temp_avg:.2f}°C" if temp_avg is not None else "unknown",
        "average_salinity": f"{psal_avg:.2f} PSU" if psal_avg is not None else "unknown"
    }

def process_netcdf(file_path):
    ds = xr.open_dataset(file_path)
    session = SessionLocal()
    
    # Batch processing for ChromaDB
    chroma_batch_size = 100
    chroma_ids = []
    chroma_embeddings = []
    chroma_metadatas = []
    chroma_documents = []
    
    try:
        # Get reference date for Julian day conversion
        ref_date_str = ds.REFERENCE_DATE_TIME.values.item()
        if isinstance(ref_date_str, bytes):
            ref_date_str = ref_date_str.decode('utf-8').strip()
        ref_date = pd.to_datetime(ref_date_str, format='%Y%m%d%H%M%S', errors='coerce')
        if pd.isna(ref_date):
            ref_date = pd.Timestamp("1950-01-01")  # Default reference date
        
        # Process float metadata
        print("Processing float metadata...")
        platform_numbers = ds.PLATFORM_NUMBER.values
        unique_platforms = np.unique([safe_decode(p) for p in platform_numbers])
        
        float_metadata_map = {}
        for platform in unique_platforms:
            if not platform:
                continue
                
            # Check if this float already exists in the database
            existing_float = session.query(FloatMetadata).filter_by(platform_number=platform).first()
            if existing_float:
                float_metadata_map[platform] = existing_float
                continue
                
            # Find the first profile index for this platform
            platform_idx = None
            for i, p in enumerate(platform_numbers):
                if safe_decode(p) == platform:
                    platform_idx = i
                    break
                    
            if platform_idx is not None:
                float_meta = FloatMetadata(
                    platform_number=platform,
                    wmo_inst_type=safe_decode(ds.WMO_INST_TYPE.values[platform_idx]) if hasattr(ds, 'WMO_INST_TYPE') else None,
                    platform_type=safe_decode(ds.PLATFORM_TYPE.values[platform_idx]) if hasattr(ds, 'PLATFORM_TYPE') else None,
                    float_serial_no=safe_decode(ds.FLOAT_SERIAL_NO.values[platform_idx]) if hasattr(ds, 'FLOAT_SERIAL_NO') else None,
                    firmware_version=safe_decode(ds.FIRMWARE_VERSION.values[platform_idx]) if hasattr(ds, 'FIRMWARE_VERSION') else None,
                    project_name=safe_decode(ds.PROJECT_NAME.values[platform_idx]) if hasattr(ds, 'PROJECT_NAME') else None,
                    pi_name=safe_decode(ds.PI_NAME.values[platform_idx]) if hasattr(ds, 'PI_NAME') else None,
                    data_centre=safe_decode(ds.DATA_CENTRE.values[platform_idx]) if hasattr(ds, 'DATA_CENTRE') else None,
                    dc_reference=safe_decode(ds.DC_REFERENCE.values[platform_idx]) if hasattr(ds, 'DC_REFERENCE') else None,
                    data_state_indicator=safe_decode(ds.DATA_STATE_INDICATOR.values[platform_idx]) if hasattr(ds, 'DATA_STATE_INDICATOR') else None,
                    data_mode=safe_decode(ds.DATA_MODE.values[platform_idx]) if hasattr(ds, 'DATA_MODE') else None,
                    positioning_system=safe_decode(ds.POSITIONING_SYSTEM.values[platform_idx]) if hasattr(ds, 'POSITIONING_SYSTEM') else None,
                    vertical_sampling_scheme=safe_decode(ds.VERTICAL_SAMPLING_SCHEME.values[platform_idx]) if hasattr(ds, 'VERTICAL_SAMPLING_SCHEME') else None,
                )
                session.add(float_meta)
                float_metadata_map[platform] = float_meta
        
        session.commit()
        
        # Process profile metadata and measurements
        print("Processing profiles and measurements...")
        n_prof = ds.dims['N_PROF']
        
        for i in tqdm(range(n_prof), desc="Processing profiles"):
            platform = safe_decode(ds.PLATFORM_NUMBER.values[i])
            cycle_number = int(ds.CYCLE_NUMBER.values[i]) if safe_float(ds.CYCLE_NUMBER.values[i]) is not None else None
            
            # Process profile metadata
            juld = convert_julian_day(ds.JULD.values[i], ref_date)
            juld_location = convert_julian_day(ds.JULD_LOCATION.values[i], ref_date)
            
            # Get station parameters
            station_params = []
            for j in range(ds.dims['N_PARAM']):
                param = safe_decode(ds.STATION_PARAMETERS.values[i, j])
                if param and param != '':
                    station_params.append(param)
            
            profile_meta = ProfileMetadata(
                platform_number=platform,
                cycle_number=cycle_number,
                direction=safe_decode(ds.DIRECTION.values[i]) if hasattr(ds, 'DIRECTION') else None,
                juld=juld,
                juld_qc=safe_decode(ds.JULD_QC.values[i]) if hasattr(ds, 'JULD_QC') else None,
                juld_location=juld_location,
                latitude=safe_float(ds.LATITUDE.values[i]),
                longitude=safe_float(ds.LONGITUDE.values[i]),
                position_qc=safe_decode(ds.POSITION_QC.values[i]) if hasattr(ds, 'POSITION_QC') else None,
                data_type=safe_decode(ds.DATA_TYPE.values.item()) if hasattr(ds, 'DATA_TYPE') else None,
                station_parameters=station_params,
                config_mission_number=int(ds.CONFIG_MISSION_NUMBER.values[i]) if hasattr(ds, 'CONFIG_MISSION_NUMBER') and safe_float(ds.CONFIG_MISSION_NUMBER.values[i]) is not None else None,
                profile_pres_qc=safe_decode(ds.PROFILE_PRES_QC.values[i]) if hasattr(ds, 'PROFILE_PRES_QC') else None,
                profile_temp_qc=safe_decode(ds.PROFILE_TEMP_QC.values[i]) if hasattr(ds, 'PROFILE_TEMP_QC') else None,
                profile_psal_qc=safe_decode(ds.PROFILE_PSAL_QC.values[i]) if hasattr(ds, 'PROFILE_PSAL_QC') else None,
            )
            
            session.add(profile_meta)
            session.flush()
            
            # Process measurements for this profile
            n_levels = ds.dims['N_LEVELS']
            profile_measurements = []
            for level in range(n_levels):
                # Check if we have valid pressure data
                pres_val = safe_float(ds.PRES.values[i, level])
                if pres_val is None:
                    continue
                    
                measurement = Measurement(
                    platform_number=platform,
                    cycle_number=cycle_number,
                    pres=pres_val,
                    pres_qc=safe_decode(ds.PRES_QC.values[i, level]) if hasattr(ds, 'PRES_QC') else None,
                    temp=safe_float(ds.TEMP.values[i, level]) if hasattr(ds, 'TEMP') else None,
                    temp_qc=safe_decode(ds.TEMP_QC.values[i, level]) if hasattr(ds, 'TEMP_QC') else None,
                    temp_adjusted=safe_float(ds.TEMP_ADJUSTED.values[i, level]) if hasattr(ds, 'TEMP_ADJUSTED') else None,
                    temp_adjusted_qc=safe_decode(ds.TEMP_ADJUSTED_QC.values[i, level]) if hasattr(ds, 'TEMP_ADJUSTED_QC') else None,
                    temp_adjusted_error=safe_float(ds.TEMP_ADJUSTED_ERROR.values[i, level]) if hasattr(ds, 'TEMP_ADJUSTED_ERROR') else None,
                    psal=safe_float(ds.PSAL.values[i, level]) if hasattr(ds, 'PSAL') else None,
                    psal_qc=safe_decode(ds.PSAL_QC.values[i, level]) if hasattr(ds, 'PSAL_QC') else None,
                    psal_adjusted=safe_float(ds.PSAL_ADJUSTED.values[i, level]) if hasattr(ds, 'PSAL_ADJUSTED') else None,
                    psal_adjusted_qc=safe_decode(ds.PSAL_ADJUSTED_QC.values[i, level]) if hasattr(ds, 'PSAL_ADJUSTED_QC') else None,
                    psal_adjusted_error=safe_float(ds.PSAL_ADJUSTED_ERROR.values[i, level]) if hasattr(ds, 'PSAL_ADJUSTED_ERROR') else None,
                )
                session.add(measurement)
                profile_measurements.append(measurement)
            
            # Get float metadata for this profile
            float_meta = float_metadata_map.get(platform)
            
            # Create ChromaDB entry for this profile
            summary_text, metadata = create_profile_summary(
                profile_meta, 
                profile_measurements, 
                float_meta
            )
            
            # MODERN: Use sentence-transformers with lazy loading
            embedding_vector = get_embedding_model().encode(summary_text).tolist()
            
            # Add to batch
            profile_id = f"{platform}_cycle_{cycle_number}"
            chroma_ids.append(profile_id)
            chroma_embeddings.append(embedding_vector)
            chroma_metadatas.append(metadata)
            chroma_documents.append(summary_text)
            
            # Process batch if reached size
            if len(chroma_ids) >= chroma_batch_size:
                try:
                    collection = get_collection()
                    collection.add(
                        ids=chroma_ids,
                        embeddings=chroma_embeddings,
                        metadatas=chroma_metadatas,
                        documents=chroma_documents
                    )
                    print(f"✓ Added batch of {len(chroma_ids)} profiles to ChromaDB")
                    # Reset batch
                    chroma_ids, chroma_embeddings, chroma_metadatas, chroma_documents = [], [], [], []
                except Exception as e:
                    print(f"⚠️ Warning: Failed to add batch to ChromaDB: {e}")
                    # Reset batch to continue processing
                    chroma_ids, chroma_embeddings, chroma_metadatas, chroma_documents = [], [], [], []
            
            # FIXED: Process calibration data with proper datetime handling
            if hasattr(ds, 'PARAMETER') and hasattr(ds, 'SCIENTIFIC_CALIB_EQUATION'):
                n_calib = ds.dims['N_CALIB']
                n_param = ds.dims['N_PARAM']
                
                for calib_idx in range(n_calib):
                    for param_idx in range(n_param):
                        param = safe_decode(ds.PARAMETER.values[i, calib_idx, param_idx])
                        if param and param != '':
                            # FIXED: Properly handle calibration date with strict None checking
                            calib_date = None
                            if hasattr(ds, 'SCIENTIFIC_CALIB_DATE'):
                                try:
                                    calib_date_val = ds.SCIENTIFIC_CALIB_DATE.values[i, calib_idx, param_idx]
                                    calib_date = safe_datetime(calib_date_val)
                                    # Double-check that we don't have NaT
                                    if calib_date is not None and pd.isna(calib_date):
                                        calib_date = None
                                except Exception as e:
                                    print(f"Warning: Error processing calibration date: {e}")
                                    calib_date = None
                            
                            calib = Calibration(
                                platform_number=platform,
                                cycle_number=cycle_number,
                                parameter=param,
                                scientific_calib_equation=safe_decode(ds.SCIENTIFIC_CALIB_EQUATION.values[i, calib_idx, param_idx]) if hasattr(ds, 'SCIENTIFIC_CALIB_EQUATION') else None,
                                scientific_calib_coefficient=safe_decode(ds.SCIENTIFIC_CALIB_COEFFICIENT.values[i, calib_idx, param_idx]) if hasattr(ds, 'SCIENTIFIC_CALIB_COEFFICIENT') else None,
                                scientific_calib_comment=safe_decode(ds.SCIENTIFIC_CALIB_COMMENT.values[i, calib_idx, param_idx]) if hasattr(ds, 'SCIENTIFIC_CALIB_COMMENT') else None,
                                scientific_calib_date=calib_date,
                            )
                            session.add(calib)
        
        # Process any remaining items in the batch
        if chroma_ids:
            try:
                collection = get_collection()
                collection.add(
                    ids=chroma_ids,
                    embeddings=chroma_embeddings,
                    metadatas=chroma_metadatas,
                    documents=chroma_documents
                )
                print(f"✓ Added final batch of {len(chroma_ids)} profiles to ChromaDB")
            except Exception as e:
                print(f"⚠️ Warning: Failed to add final batch to ChromaDB: {e}")
        
        # FIXED: Process history data with proper datetime handling
        if hasattr(ds, 'HISTORY_INSTITUTION') and ds.dims['N_HISTORY'] > 0:
            n_history = ds.dims['N_HISTORY']
            print(f"Processing {n_history} history records...")
            
            for hist_idx in range(n_history):
                for prof_idx in range(n_prof):
                    platform_hist = safe_decode(ds.PLATFORM_NUMBER.values[prof_idx])
                    cycle_hist = int(ds.CYCLE_NUMBER.values[prof_idx]) if safe_float(ds.CYCLE_NUMBER.values[prof_idx]) is not None else None
                    
                    # FIXED: Safely handle datetime conversion for history date
                    history_date = None
                    if hasattr(ds, 'HISTORY_DATE'):
                        try:
                            history_date_val = ds.HISTORY_DATE.values[hist_idx, prof_idx]
                            history_date = safe_datetime(history_date_val)
                            # Double-check that we don't have NaT
                            if history_date is not None and pd.isna(history_date):
                                history_date = None
                        except Exception as e:
                            print(f"Warning: Error processing history date: {e}")
                            history_date = None
                    
                    history = ProcessingHistory(
                        platform_number=platform_hist,
                        cycle_number=cycle_hist,
                        history_institution=safe_decode(ds.HISTORY_INSTITUTION.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_INSTITUTION') else None,
                        history_step=safe_decode(ds.HISTORY_STEP.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_STEP') else None,
                        history_software=safe_decode(ds.HISTORY_SOFTWARE.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_SOFTWARE') else None,
                        history_software_release=safe_decode(ds.HISTORY_SOFTWARE_RELEASE.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_SOFTWARE_RELEASE') else None,
                        history_reference=safe_decode(ds.HISTORY_REFERENCE.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_REFERENCE') else None,
                        history_date=history_date,
                        history_action=safe_decode(ds.HISTORY_ACTION.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_ACTION') else None,
                        history_parameter=safe_decode(ds.HISTORY_PARAMETER.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_PARAMETER') else None,
                        history_start_pres=safe_float(ds.HISTORY_START_PRES.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_START_PRES') else None,
                        history_stop_pres=safe_float(ds.HISTORY_STOP_PRES.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_STOP_PRES') else None,
                        history_previous_value=safe_float(ds.HISTORY_PREVIOUS_VALUE.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_PREVIOUS_VALUE') else None,
                        history_qctest=safe_decode(ds.HISTORY_QCTEST.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_QCTEST') else None,
                    )
                    session.add(history)
        
        session.commit()
        print("✅ NetCDF processing complete!")
        print(f"📊 ChromaDB data stored locally in: {CHROMA_PERSIST_DIRECTORY}")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error processing NetCDF file: {e}")
        raise e
        
    finally:
        session.close()
        ds.close()

if __name__ == "__main__":
    # Example usage
    process_netcdf("data/20250826_prof.nc")