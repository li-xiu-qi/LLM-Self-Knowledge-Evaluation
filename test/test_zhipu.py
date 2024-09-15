import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("API_KEY")
base_url = os.environ.get("BASE_URL")
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)



def chat(messages: list[dict]):
    completion = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        top_p=0.7,
        temperature=0.9
    )

    response = completion.choices[0].message.content

    return response


if __name__ == "__main__":
    messages = [{"role": "user", "content": "你好"}]
    res = chat(messages)
    print(res)
