import httpx
import json

LM_STUDIO_URL = "https://unmashed-automatically-hortense.ngrok-free.dev/v1/chat/completions"

def test_lm_studio():
    """Test LM Studio connection and response"""
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions directly and concisely."
        },
        {
            "role": "user",
            "content": "What is 2+2? Answer with just the number."
        }
    ]
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                LM_STUDIO_URL,
                json={
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "stream": False
                }
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                print(f"\nExtracted Answer: {answer}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_lm_studio()