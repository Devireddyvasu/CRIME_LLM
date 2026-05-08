import requests

class APILLM:
    def __init__(self, repo_id, hf_token):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.api_url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt):
        print(f"Sending prompt to HF Inference API ({self.repo_id})...")
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                return str(result)
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"API Exception: {str(e)}"
            print(error_msg)
            return error_msg
