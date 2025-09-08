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

def process_netcdf(file_path):
    ds = xr.open_dataset(file_path)
    session = SessionLocal()
    
    # Get reference date for Julian day conversion
    ref_date_str = ds.REFERENCE_DATE_TIME.values.item().decode('utf-8').strip()
    ref_date = pd.to_datetime(ref_date_str, format='%Y%m%d%H%M%S', errors='coerce')
    if pd.isna(ref_date):
        ref_date = pd.Timestamp("1950-01-01")  # Default reference date
    
    # Process float metadata
    print("Processing float metadata...")
    platform_numbers = ds.PLATFORM_NUMBER.values
    unique_platforms = np.unique([p.decode('utf-8').strip() for p in platform_numbers])
    
    for platform in unique_platforms:
        # Check if this float already exists in the database
        existing_float = session.query(FloatMetadata).filter_by(platform_number=platform).first()
        if existing_float:
            continue
            
        # Find the first profile index for this platform
        platform_idx = None
        for i, p in enumerate(platform_numbers):
            if p.decode('utf-8').strip() == platform:
                platform_idx = i
                break
                
        if platform_idx is not None:
            float_meta = FloatMetadata(
                platform_number=platform,
                wmo_inst_type=ds.WMO_INST_TYPE.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'WMO_INST_TYPE') else None,
                platform_type=ds.PLATFORM_TYPE.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'PLATFORM_TYPE') else None,
                float_serial_no=ds.FLOAT_SERIAL_NO.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'FLOAT_SERIAL_NO') else None,
                firmware_version=ds.FIRMWARE_VERSION.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'FIRMWARE_VERSION') else None,
                project_name=ds.PROJECT_NAME.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'PROJECT_NAME') else None,
                pi_name=ds.PI_NAME.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'PI_NAME') else None,
                data_centre=ds.DATA_CENTRE.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'DATA_CENTRE') else None,
                dc_reference=ds.DC_REFERENCE.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'DC_REFERENCE') else None,
                data_state_indicator=ds.DATA_STATE_INDICATOR.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'DATA_STATE_INDICATOR') else None,
                data_mode=ds.DATA_MODE.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'DATA_MODE') else None,
                positioning_system=ds.POSITIONING_SYSTEM.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'POSITIONING_SYSTEM') else None,
                vertical_sampling_scheme=ds.VERTICAL_SAMPLING_SCHEME.values[platform_idx].decode('utf-8').strip() if hasattr(ds, 'VERTICAL_SAMPLING_SCHEME') else None,
            )
            session.add(float_meta)
    
    session.commit()
    
    # Process profile metadata and measurements
    print("Processing profiles and measurements...")
    n_prof = ds.dims['N_PROF']
    
    for i in tqdm(range(n_prof)):
        platform = ds.PLATFORM_NUMBER.values[i].decode('utf-8').strip()
        cycle_number = int(ds.CYCLE_NUMBER.values[i]) if ds.CYCLE_NUMBER.values[i] != 99999 else None
        
        # Process profile metadata
        juld = convert_julian_day(ds.JULD.values[i], ref_date)
        juld_location = convert_julian_day(ds.JULD_LOCATION.values[i], ref_date)
        
        # Get station parameters
        station_params = []
        for j in range(ds.dims['N_PARAM']):
            param = ds.STATION_PARAMETERS.values[i, j].decode('utf-8').strip()
            if param and param != '':
                station_params.append(param)
        
        profile_meta = ProfileMetadata(
            platform_number=platform,
            cycle_number=cycle_number,
            direction=ds.DIRECTION.values[i].decode('utf-8').strip() if hasattr(ds, 'DIRECTION') else None,
            juld=juld,
            juld_qc=ds.JULD_QC.values[i].decode('utf-8').strip() if hasattr(ds, 'JULD_QC') else None,
            juld_location=juld_location,
            latitude=float(ds.LATITUDE.values[i]) if ds.LATITUDE.values[i] != 99999.0 else None,
            longitude=float(ds.LONGITUDE.values[i]) if ds.LONGITUDE.values[i] != 99999.0 else None,
            position_qc=ds.POSITION_QC.values[i].decode('utf-8').strip() if hasattr(ds, 'POSITION_QC') else None,
            data_type=ds.DATA_TYPE.values.item().decode('utf-8').strip() if hasattr(ds, 'DATA_TYPE') else None,
            station_parameters=station_params,
            config_mission_number=int(ds.CONFIG_MISSION_NUMBER.values[i]) if hasattr(ds, 'CONFIG_MISSION_NUMBER') and ds.CONFIG_MISSION_NUMBER.values[i] != 99999 else None,
            profile_pres_qc=ds.PROFILE_PRES_QC.values[i].decode('utf-8').strip() if hasattr(ds, 'PROFILE_PRES_QC') else None,
            profile_temp_qc=ds.PROFILE_TEMP_QC.values[i].decode('utf-8').strip() if hasattr(ds, 'PROFILE_TEMP_QC') else None,
            profile_psal_qc=ds.PROFILE_PSAL_QC.values[i].decode('utf-8').strip() if hasattr(ds, 'PROFILE_PSAL_QC') else None,
        )
        
        # Removed geometry creation to avoid PostGIS issues
        # if profile_meta.latitude is not None and profile_meta.longitude is not None:
        #     profile_meta.geometry = func.ST_SetSRID(func.ST_MakePoint(profile_meta.longitude, profile_meta.latitude), 4326)
        
        session.add(profile_meta)
        session.flush()  # Flush to get the ID if needed
        
        # Process measurements for this profile
        n_levels = ds.dims['N_LEVELS']
        for level in range(n_levels):
            # Check if we have valid pressure data
            pres_val = float(ds.PRES.values[i, level]) if ds.PRES.values[i, level] != 99999.0 else None
            if pres_val is None:
                continue
                
            measurement = Measurement(
                platform_number=platform,
                cycle_number=cycle_number,
                pres=pres_val,
                pres_qc=ds.PRES_QC.values[i, level].decode('utf-8').strip() if hasattr(ds, 'PRES_QC') else None,
                temp=float(ds.TEMP.values[i, level]) if hasattr(ds, 'TEMP') and ds.TEMP.values[i, level] != 99999.0 else None,
                temp_qc=ds.TEMP_QC.values[i, level].decode('utf-8').strip() if hasattr(ds, 'TEMP_QC') else None,
                temp_adjusted=float(ds.TEMP_ADJUSTED.values[i, level]) if hasattr(ds, 'TEMP_ADJUSTED') and ds.TEMP_ADJUSTED.values[i, level] != 99999.0 else None,
                temp_adjusted_qc=ds.TEMP_ADJUSTED_QC.values[i, level].decode('utf-8').strip() if hasattr(ds, 'TEMP_ADJUSTED_QC') else None,
                temp_adjusted_error=float(ds.TEMP_ADJUSTED_ERROR.values[i, level]) if hasattr(ds, 'TEMP_ADJUSTED_ERROR') and ds.TEMP_ADJUSTED_ERROR.values[i, level] != 99999.0 else None,
                psal=float(ds.PSAL.values[i, level]) if hasattr(ds, 'PSAL') and ds.PSAL.values[i, level] != 99999.0 else None,
                psal_qc=ds.PSAL_QC.values[i, level].decode('utf-8').strip() if hasattr(ds, 'PSAL_QC') else None,
                psal_adjusted=float(ds.PSAL_ADJUSTED.values[i, level]) if hasattr(ds, 'PSAL_ADJUSTED') and ds.PSAL_ADJUSTED.values[i, level] != 99999.0 else None,
                psal_adjusted_qc=ds.PSAL_ADJUSTED_QC.values[i, level].decode('utf-8').strip() if hasattr(ds, 'PSAL_ADJUSTED_QC') else None,
                psal_adjusted_error=float(ds.PSAL_ADJUSTED_ERROR.values[i, level]) if hasattr(ds, 'PSAL_ADJUSTED_ERROR') and ds.PSAL_ADJUSTED_ERROR.values[i, level] != 99999.0 else None,
            )
            session.add(measurement)
        
        # Process calibration data if available
        if hasattr(ds, 'PARAMETER') and hasattr(ds, 'SCIENTIFIC_CALIB_EQUATION'):
            n_calib = ds.dims['N_CALIB']
            n_param = ds.dims['N_PARAM']
            
            for calib_idx in range(n_calib):
                for param_idx in range(n_param):
                    param = ds.PARAMETER.values[i, calib_idx, param_idx].decode('utf-8').strip()
                    if param and param != '':
                        calib = Calibration(
                            platform_number=platform,
                            cycle_number=cycle_number,
                            parameter=param,
                            scientific_calib_equation=ds.SCIENTIFIC_CALIB_EQUATION.values[i, calib_idx, param_idx].decode('utf-8').strip() if hasattr(ds, 'SCIENTIFIC_CALIB_EQUATION') else None,
                            scientific_calib_coefficient=ds.SCIENTIFIC_CALIB_COEFFICIENT.values[i, calib_idx, param_idx].decode('utf-8').strip() if hasattr(ds, 'SCIENTIFIC_CALIB_COEFFICIENT') else None,
                            scientific_calib_comment=ds.SCIENTIFIC_CALIB_COMMENT.values[i, calib_idx, param_idx].decode('utf-8').strip() if hasattr(ds, 'SCIENTIFIC_CALIB_COMMENT') else None,
                            scientific_calib_date=pd.to_datetime(ds.SCIENTIFIC_CALIB_DATE.values[i, calib_idx, param_idx].decode('utf-8').strip(), format='%Y%m%d%H%M%S', errors='coerce') if hasattr(ds, 'SCIENTIFIC_CALIB_DATE') else None,
                        )
                        session.add(calib)
    
    # Process history data if available
    if hasattr(ds, 'HISTORY_INSTITUTION') and ds.dims['N_HISTORY'] > 0:
        n_history = ds.dims['N_HISTORY']
        print(f"Processing {n_history} history records...")
        
        for hist_idx in range(n_history):
            for prof_idx in range(n_prof):
                history = ProcessingHistory(
                    platform_number=ds.PLATFORM_NUMBER.values[prof_idx].decode('utf-8').strip(),
                    cycle_number=int(ds.CYCLE_NUMBER.values[prof_idx]) if ds.CYCLE_NUMBER.values[prof_idx] != 99999 else None,
                    history_institution=ds.HISTORY_INSTITUTION.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_INSTITUTION') else None,
                    history_step=ds.HISTORY_STEP.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_STEP') else None,
                    history_software=ds.HISTORY_SOFTWARE.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_SOFTWARE') else None,
                    history_software_release=ds.HISTORY_SOFTWARE_RELEASE.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_SOFTWARE_RELEASE') else None,
                    history_reference=ds.HISTORY_REFERENCE.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_REFERENCE') else None,
                    history_date=pd.to_datetime(ds.HISTORY_DATE.values[hist_idx, prof_idx].decode('utf-8').strip(), format='%Y%m%d%H%M%S', errors='coerce') if hasattr(ds, 'HISTORY_DATE') else None,
                    history_action=ds.HISTORY_ACTION.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_ACTION') else None,
                    history_parameter=ds.HISTORY_PARAMETER.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_PARAMETER') else None,
                    history_start_pres=float(ds.HISTORY_START_PRES.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_START_PRES') and ds.HISTORY_START_PRES.values[hist_idx, prof_idx] != 99999.0 else None,
                    history_stop_pres=float(ds.HISTORY_STOP_PRES.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_STOP_PRES') and ds.HISTORY_STOP_PRES.values[hist_idx, prof_idx] != 99999.0 else None,
                    history_previous_value=float(ds.HISTORY_PREVIOUS_VALUE.values[hist_idx, prof_idx]) if hasattr(ds, 'HISTORY_PREVIOUS_VALUE') and ds.HISTORY_PREVIOUS_VALUE.values[hist_idx, prof_idx] != 99999.0 else None,
                    history_qctest=ds.HISTORY_QCTEST.values[hist_idx, prof_idx].decode('utf-8').strip() if hasattr(ds, 'HISTORY_QCTEST') else None,
                )
                session.add(history)
    
    session.commit()
    session.close()
    ds.close()
    print("NetCDF processing complete!")