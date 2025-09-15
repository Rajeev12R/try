# quick_test.py
import requests
import os
import sys

def test_upload():
    # Use the correct path to your NetCDF file
    file_path = "data/20250826_prof.nc"
    
    # Get the absolute path to be sure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_file_path = os.path.join(current_dir, file_path)
    
    print(f"🔍 Looking for file: {absolute_file_path}")
    
    if not os.path.exists(absolute_file_path):
        print(f"❌ File not found: {absolute_file_path}")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        
        # Check if data directory exists
        data_dir = os.path.join(current_dir, 'data')
        if os.path.exists(data_dir):
            print("Files in data directory:", os.listdir(data_dir))
        else:
            print("❌ 'data' directory does not exist")
            
        return False
    
    print(f"✅ File found: {absolute_file_path}")
    print(f"📊 File size: {os.path.getsize(absolute_file_path):,} bytes")
    
    try:
        with open(absolute_file_path, 'rb') as f:
            files = {'file': (os.path.basename(absolute_file_path), f, 'application/netcdf')}
            
            print("📤 Uploading to http://localhost:8000/upload-netcdf/ ...")
            print("⏳ This may take several minutes for large files...")
            
            response = requests.post(
                "http://localhost:8000/upload-netcdf/", 
                files=files,
                timeout=600  # 10 minute timeout for large files
            )
        
        print(f"📋 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ Success! Response: {result}")
                return True
            except Exception as json_error:
                print(f"📝 Response Text: {response.text}")
                print(f"⚠️ JSON parse error: {json_error}")
                return True  # Still consider it success if upload worked
        else:
            print(f"❌ Server returned error: {response.status_code}")
            print(f"📝 Response Text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out! The file might be too large or server is not responding.")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error! Make sure your FastAPI server is running on localhost:8000")
        print("💡 Run this command first: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting NetCDF upload test...")
    print("📁 Current directory:", os.getcwd())
    print("📁 Script directory:", os.path.dirname(os.path.abspath(__file__)))
    
    # Add current directory to path in case we need to import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    success = test_upload()
    
    if success:
        print("🎉 Upload test completed successfully!")
        print("\n📝 Next steps:")
        print("1. Check your database to see if data was inserted")
        print("2. Check the chroma_db_storage folder for embeddings")
        print("3. Test the API endpoints to query the data")
    else:
        print("💥 Upload test failed!")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure your FastAPI server is running")
        print("2. Check the file path is correct")
        print("3. Check if the NetCDF file is valid")