import os
import time
from groq import Groq

current_key_index = 0
client = None


def get_next_key(api_keys):
    global current_key_index, client
    current_key_index = (current_key_index + 1) % len(api_keys)
    os.environ["GROQ_API_KEY"] = api_keys[current_key_index]
    client = Groq()


def initialize_client(api_keys: list):
    get_next_key(api_keys)


def completion(
        prompt: str,
        model: str,
        temperature: float = 0,
        top_p: float = 1.0,
        max_retries: int = 100
) -> str:
    global client
    if client is None:
        initialize_client()

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            get_next_key()
            time.sleep(300)

    raise RuntimeError("Max retries reached. Unable to get a response.")
