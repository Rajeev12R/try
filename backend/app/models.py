from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

class FloatMetadata(Base):
    __tablename__ = "float_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_number = Column(String(8), nullable=False, unique=True, index=True)
    wmo_inst_type = Column(String(4))
    platform_type = Column(String(32))
    float_serial_no = Column(String(32))
    firmware_version = Column(String(32))
    project_name = Column(String(64))
    pi_name = Column(String(64))
    data_centre = Column(String(2))
    dc_reference = Column(String(32))
    data_state_indicator = Column(String(4))
    data_mode = Column(String(1))
    positioning_system = Column(String(8))
    vertical_sampling_scheme = Column(String(256))
    
    # Relationships
    profiles = relationship("ProfileMetadata", back_populates="float_metadata")

class ProfileMetadata(Base):
    __tablename__ = "profile_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_number = Column(String(8), ForeignKey('float_metadata.platform_number'), nullable=False, index=True)
    cycle_number = Column(Integer, index=True)
    direction = Column(String(1))
    juld = Column(DateTime, index=True) 
    juld_qc = Column(String(1))
    juld_location = Column(DateTime)  
    latitude = Column(Float, index=True)
    longitude = Column(Float, index=True)
    position_qc = Column(String(1))
    data_type = Column(String(16))
    station_parameters = Column(ARRAY(String(16))) 
    config_mission_number = Column(Integer)
    profile_pres_qc = Column(String(1))
    profile_temp_qc = Column(String(1))
    profile_psal_qc = Column(String(1))
    
    # Relationships
    float_metadata = relationship("FloatMetadata", back_populates="profiles")
    measurements = relationship(
        "Measurement", 
        back_populates="profile",
        foreign_keys="[Measurement.platform_number, Measurement.cycle_number]",
        primaryjoin="and_(ProfileMetadata.platform_number == Measurement.platform_number, "
                   "ProfileMetadata.cycle_number == Measurement.cycle_number)"
    )
    calibrations = relationship(
        "Calibration", 
        back_populates="profile",
        foreign_keys="[Calibration.platform_number, Calibration.cycle_number]",
        primaryjoin="and_(ProfileMetadata.platform_number == Calibration.platform_number, "
                   "ProfileMetadata.cycle_number == Calibration.cycle_number)"
    )
    processing_history = relationship(
        "ProcessingHistory", 
        back_populates="profile",
        foreign_keys="[ProcessingHistory.platform_number, ProcessingHistory.cycle_number]",
        primaryjoin="and_(ProfileMetadata.platform_number == ProcessingHistory.platform_number, "
                   "ProfileMetadata.cycle_number == ProcessingHistory.cycle_number)"
    )

class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_number = Column(String(8), nullable=False, index=True)
    cycle_number = Column(Integer, index=True)
    pres = Column(Float, index=True)
    pres_qc = Column(String(1))
    temp = Column(Float)
    temp_qc = Column(String(1))
    temp_adjusted = Column(Float)
    temp_adjusted_qc = Column(String(1))
    temp_adjusted_error = Column(Float)
    psal = Column(Float)
    psal_qc = Column(String(1))
    psal_adjusted = Column(Float)
    psal_adjusted_qc = Column(String(1))
    psal_adjusted_error = Column(Float)
    
    # Relationship
    profile = relationship(
        "ProfileMetadata", 
        back_populates="measurements",
        foreign_keys=[platform_number, cycle_number],
        primaryjoin="and_(Measurement.platform_number == ProfileMetadata.platform_number, "
                   "Measurement.cycle_number == ProfileMetadata.cycle_number)",
        viewonly=True
    )

class Calibration(Base):
    __tablename__ = "calibrations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_number = Column(String(8), nullable=False, index=True)
    cycle_number = Column(Integer, index=True)
    parameter = Column(String(16))
    scientific_calib_equation = Column(Text)
    scientific_calib_coefficient = Column(Text)
    scientific_calib_comment = Column(Text)
    scientific_calib_date = Column(DateTime)
    
    # Relationship
    profile = relationship(
        "ProfileMetadata", 
        back_populates="calibrations",
        foreign_keys=[platform_number, cycle_number],
        primaryjoin="and_(Calibration.platform_number == ProfileMetadata.platform_number, "
                   "Calibration.cycle_number == ProfileMetadata.cycle_number)",
        viewonly=True
    )

class ProcessingHistory(Base):
    __tablename__ = "processing_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_number = Column(String(8), nullable=False, index=True)
    cycle_number = Column(Integer, index=True)
    history_institution = Column(String(4))
    history_step = Column(String(4))
    history_software = Column(String(4))
    history_software_release = Column(String(4))
    history_reference = Column(String(64))
    history_date = Column(DateTime)
    history_action = Column(String(4))
    history_parameter = Column(String(16))
    history_start_pres = Column(Float)
    history_stop_pres = Column(Float)
    history_previous_value = Column(Float)
    history_qctest = Column(String(16))
    
    # Relationship
    profile = relationship(
        "ProfileMetadata", 
        back_populates="processing_history",
        foreign_keys=[platform_number, cycle_number],
        primaryjoin="and_(ProcessingHistory.platform_number == ProfileMetadata.platform_number, "
                   "ProcessingHistory.cycle_number == ProfileMetadata.cycle_number)",
        viewonly=True
    )