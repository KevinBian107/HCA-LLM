import requests
import tiktoken
import time

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

def LLaMA_response(messages, model_name, url="http://localhost:11434/api/generate"):
    """
    messages: list of message dictionaries following ChatCompletion format
    model_name: name of the LLaMA model
    url: endpoint where LLaMA is hosted
    """
    prompt = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )
    data = {
        "model": model_name,
        "prompt": prompt,
        # "max_tokens": 2000,
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
    }

    try:
        response = requests.post(url=url, json=data)

        if response.status_code == 200:
            response_text = response.json().get("response", "")
            token_num_count = sum(
                len(enc.encode(msg["content"])) for msg in messages
            ) + len(enc.encode(response_text))
            #print(f"Token_num_count: {token_num_count}")
            return response_text, token_num_count
        else:
            print("Error:", response.status_code, response.json())
            return None, 0
    except Exception as e:
        print(f"API call failed: {e}")
        return None, 0


# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello, how are you? What is your parameter size"}
# ]
# response, token_count = LLaMA_response(messages, model_name, url)
# print("Response:", response)
# print("Token count:", token_count)
