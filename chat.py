from openai import OpenAI, AsyncOpenAI
from config import API_KEY, BASE_URL, MODEL_NAME


class LLM:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name or MODEL_NAME
        self.client = OpenAI(api_key=api_key or API_KEY,
                             base_url=base_url or BASE_URL)

    def __call__(self, messages: list[dict], **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

        response = completion.choices[0].message.content

        return response


class AsyncLLM:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name or MODEL_NAME
        self.client = AsyncOpenAI(api_key=api_key or API_KEY,
                                  base_url=base_url or BASE_URL)

    async def __call__(self, messages: list[dict], **kwargs):
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

        response = completion.choices[0].message.content

        return response


if __name__ == "__main__":
    llm = LLM(model_name="glm-4-flash")
    messages = [{"role": "user", "content": "你好"}]
    print(llm(messages))
