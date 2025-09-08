#!/usr/bin/env python3
"""
Simple database initialization script that creates tables without PostGIS dependencies
"""

import os
from config import engine
from models import Base
from sqlalchemy import text  # Add this import

def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    print("Database URL:", os.getenv("SUPABASE_DB_URL", "Not found"))
    
    try:
        # Drop and recreate all tables (be careful in production!)
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
        
        print("Creating new tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully!")
        
        # List created tables
        print("\nCreated tables:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise

def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as conn:
            # Use text() for proper SQL execution
            result = conn.execute(text("SELECT 1 as test"))
            print("✅ Database connection test passed")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ARGO Database Initialization")
    print("=" * 40)
    
    # Test connection first
    if test_connection():
        create_tables()
        print("\n🎉 Database setup complete!")
    else:
        print("\n💥 Database setup failed!")