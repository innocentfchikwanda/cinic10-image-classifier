#!/usr/bin/env python3
"""
Simple test script to verify the API is working with a test image
"""
import requests
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a simple 32x32 test image"""
    # Create a simple pattern that might look like an airplane
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Add some structure to make it more airplane-like
    img_array[10:22, 5:27] = [135, 206, 235]  # Sky blue background
    img_array[14:18, 8:24] = [192, 192, 192]  # Gray airplane body
    img_array[15:17, 6:26] = [169, 169, 169]  # Darker gray for wings
    
    return Image.fromarray(img_array)

def test_prediction():
    """Test the prediction endpoint"""
    # Create test image
    test_img = create_test_image()
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send to API
    files = {'file': ('test.png', img_bytes, 'image/png')}
    
    try:
        response = requests.post('http://localhost:8000/api/predict', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Model: {result.get('model', 'Unknown')}")
            print("Top 3 Predictions:")
            for i, pred in enumerate(result['results'][:3], 1):
                print(f"  {i}. {pred['label']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
            print(f"Processing time: {result.get('processingTime', 'N/A')} seconds")
            return True
        else:
            print(f"❌ API Test Failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ API Test Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Image Classification API...")
    success = test_prediction()
    if success:
        print("\n🎉 All tests passed! The application is ready to use.")
    else:
        print("\n💥 Tests failed. Please check the server logs.")
