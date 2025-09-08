import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from config import engine
from models import Base, FloatMetadata, ProfileMetadata, Measurement, Calibration, ProcessingHistory
from tqdm import tqdm
import datetime

SessionLocal = sessionmaker(bind=engine)

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
    """Safely convert to datetime, handling NaT and invalid values"""
    if pd.isna(value) or value in [99999.0, 999999.0]:
        return None
    try:
        if isinstance(value, (pd.Timestamp, datetime.datetime)):
            return value
        elif isinstance(value, str):
            return pd.to_datetime(value, errors='coerce')
        else:
            return pd.to_datetime(str(value), errors='coerce')
    except (ValueError, TypeError):
        return None

def process_netcdf(file_path):
    ds = xr.open_dataset(file_path)
    session = SessionLocal()
    
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
        
        for platform in unique_platforms:
            if not platform:
                continue
                
            # Check if this float already exists in the database
            existing_float = session.query(FloatMetadata).filter_by(platform_number=platform).first()
            if existing_float:
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
        
        session.commit()
        
        # Process profile metadata and measurements
        print("Processing profiles and measurements...")
        n_prof = ds.dims['N_PROF']
        
        for i in tqdm(range(n_prof)):
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
            
            # Process calibration data if available
            if hasattr(ds, 'PARAMETER') and hasattr(ds, 'SCIENTIFIC_CALIB_EQUATION'):
                n_calib = ds.dims['N_CALIB']
                n_param = ds.dims['N_PARAM']
                
                for calib_idx in range(n_calib):
                    for param_idx in range(n_param):
                        param = safe_decode(ds.PARAMETER.values[i, calib_idx, param_idx])
                        if param and param != '':
                            # Safely handle datetime conversion for calibration date
                            calib_date = None
                            if hasattr(ds, 'SCIENTENTIFIC_CALIB_DATE'):
                                calib_date_val = ds.SCIENTIFIC_CALIB_DATE.values[i, calib_idx, param_idx]
                                calib_date = safe_datetime(calib_date_val)
                            
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
        
        # Process history data if available
        if hasattr(ds, 'HISTORY_INSTITUTION') and ds.dims['N_HISTORY'] > 0:
            n_history = ds.dims['N_HISTORY']
            print(f"Processing {n_history} history records...")
            
            for hist_idx in range(n_history):
                for prof_idx in range(n_prof):
                    platform_hist = safe_decode(ds.PLATFORM_NUMBER.values[prof_idx])
                    cycle_hist = int(ds.CYCLE_NUMBER.values[prof_idx]) if safe_float(ds.CYCLE_NUMBER.values[prof_idx]) is not None else None
                    
                    # Safely handle datetime conversion for history date
                    history_date = None
                    if hasattr(ds, 'HISTORY_DATE'):
                        history_date_val = ds.HISTORY_DATE.values[hist_idx, prof_idx]
                        history_date = safe_datetime(history_date_val)
                    
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
        print("NetCDF processing complete!")
        
    except Exception as e:
        session.rollback()
        raise e
        
    finally:
        session.close()
        ds.close()