#!/usr/bin/env python3
"""
LM Studio Connection Test
Run this to diagnose connection issues
"""

import httpx
import json
import sys

def test_connection():
    print("=" * 60)
    print("LM STUDIO CONNECTION TEST")
    print("=" * 60)
    print()
    
    # Test 1: Basic connectivity
    print("Test 1: Checking if LM Studio is reachable...")
    try:
        response = httpx.get("http://localhost:1234/v1/models", timeout=5.0)
        
        if response.status_code == 200:
            print("✅ SUCCESS - LM Studio is running!")
            data = response.json()
            print(f"   Models available: {len(data.get('data', []))}")
            if data.get('data'):
                for model in data['data']:
                    print(f"   - {model.get('id', 'unknown')}")
        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            return False
            
    except httpx.ConnectError:
        print("❌ FAILED - Cannot connect to LM Studio")
        print()
        print("TROUBLESHOOTING STEPS:")
        print("1. Open LM Studio application")
        print("2. Go to 'Local Server' or 'Developer' tab")
        print("3. Select 'TinyLlama-1.1B-Chat-v0.6' from dropdown")
        print("4. Click 'Start Server' button")
        print("5. Wait until you see 'Server running on http://localhost:1234'")
        print()
        return False
        
    except httpx.TimeoutException:
        print("❌ FAILED - Connection timeout")
        print("   LM Studio is not responding (might be loading model)")
        return False
        
    except Exception as e:
        print(f"❌ FAILED - Error: {e}")
        return False
    
    print()
    
    # Test 2: Chat completion
    print("Test 2: Testing chat completion...")
    try:
        response = httpx.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello, connection test successful!'"}
                ],
                "temperature": 0.7,
                "max_tokens": 50,
                "stream": False
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print("✅ SUCCESS - Chat is working!")
            print(f"   Response: {answer}")
        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED - Error: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✅ ALL TESTS PASSED - LM Studio is working correctly!")
    print("=" * 60)
    print()
    print("You can now run: python main.py")
    return True


def check_port():
    """Check what's using port 1234"""
    import subprocess
    import platform
    
    print()
    print("Checking port 1234...")
    print()
    
    system = platform.system()
    
    try:
        if system == "Darwin" or system == "Linux":
            result = subprocess.run(
                ["lsof", "-i", ":1234"],
                capture_output=True,
                text=True
            )
            if result.stdout:
                print("Port 1234 is in use by:")
                print(result.stdout)
            else:
                print("Port 1234 is free (nothing is using it)")
                
        elif system == "Windows":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True
            )
            lines = [line for line in result.stdout.split('\n') if ':1234' in line]
            if lines:
                print("Port 1234 is in use:")
                for line in lines:
                    print(line)
            else:
                print("Port 1234 is free (nothing is using it)")
    except Exception as e:
        print(f"Could not check port: {e}")


if __name__ == "__main__":
    success = test_connection()
    
    if not success:
        check_port()
        print()
        print("💡 TIP: Make sure LM Studio server is started before running this test")
        print("💡 TIP: Check LM_STUDIO_FIX.md for detailed troubleshooting")
        sys.exit(1)
    
    sys.exit(0)