# quick_test.py
import requests
import os

def test_upload():
    # Use the correct path to your NetCDF file
    file_path = "data/20250826_prof.nc"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        if os.path.exists('data'):
            print("Files in data directory:", os.listdir('data'))
        return False
    
    print(f"📤 Uploading: {file_path}")
    print(f"📊 File size: {os.path.getsize(file_path):,} bytes")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/netcdf')}
            response = requests.post(
                "http://localhost:8000/upload-netcdf/", 
                files=files,
                timeout=300  # 5 minute timeout
            )
        
        print(f"📋 Status Code: {response.status_code}")
        
        try:
            result = response.json()
            print(f"✅ Response: {result}")
            return True
        except:
            print(f"📝 Response Text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting NetCDF upload test...")
    print("📁 Current directory:", os.getcwd())
    
    if test_upload():
        print("🎉 Upload test completed successfully!")
    else:
        print("💥 Upload test failed!")