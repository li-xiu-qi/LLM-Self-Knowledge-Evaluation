import os
import asyncio

import pandas as pd
from tqdm.asyncio import tqdm
from llm_parse_json import parse_json
from openai import BadRequestError
from chat import AsyncLLM


class LLMSelfKnowledge:
    base_path = os.path.dirname(__file__)
    is_known_prompt_path = os.path.join(base_path, "prompts", "is_known_prompt.txt")
    verify_answer_prompt_path = os.path.join(base_path, "prompts", "verify_answer_prompt.txt")
    output_dir = "test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def __init__(self, llm: AsyncLLM, data_set, temperatures):
        self.llm = llm
        if data_set:
            self.questions = [q for q in data_set['question']]
            self.answers = [a for a in data_set['answer']]
        self.results = []
        self.temperatures = temperatures

    async def is_known(self, question, temperature):
        model_config = {
            "temperature": temperature,
            "max_tokens": 4095
        }
        is_known_prompt = self._load_prompt(self.is_known_prompt_path)
        messages = [{"role": "system", "content": is_known_prompt},
                    {"role": "user", "content": question}]
        result = await self.llm(messages=messages, **model_config)
        return result

    async def verify_answer(self, response, standard_answer, temperature):
        model_config = {
            "temperature": temperature,
            "max_tokens": 4095
        }
        verify_answer_prompt = self._load_prompt(self.verify_answer_prompt_path)
        user_input = f"""- **Input Response**: {response}
                        - **Standard Answer**: {standard_answer}
                     """
        messages = [{"role": "system", "content": verify_answer_prompt},
                    {"role": "user", "content": user_input}]
        result = await self.llm(messages=messages, **model_config)
        return result

    async def translate(self, text, temperature):
        model_config = {
            "temperature": 0.0,
            "max_tokens": 4095
        }
        messages = [{"role": "system",
                     "content": "You are a helpful assistant that translates the following text into Chinese."},
                    {"role": "user", "content": text}]
        result = await self.llm(messages=messages, **model_config)
        return result

    async def process_data_item(self, question, answer, temperature, semaphore):
        async with semaphore:
            try:
                is_known_result = await self.is_known(question, temperature)
                if is_known_result:
                    parse_is_known_result = self.parse_result(is_known_result)
                    is_known = parse_is_known_result["is_known"]
                    is_correct = True
                    response = parse_is_known_result["answer"]
                    reason = parse_is_known_result["answer"]
                    score = 0
                    if is_known:
                        verify_answer_result = await self.verify_answer(response=response, standard_answer=answer,
                                                                        temperature=temperature)
                        parse_verify_answer_result = self.parse_result(verify_answer_result)
                        if parse_verify_answer_result:
                            is_correct = parse_verify_answer_result["is_correct"]
                            reason = parse_verify_answer_result["reason"]
                            score = parse_verify_answer_result["score"]

                    return {
                        "temperature": temperature,
                        "question": question,
                        "is_known": is_known,
                        "answer": answer,
                        "response": response,
                        "is_correct": is_correct,
                        "score": score,
                        "reason": reason,
                        "translate_reason": await self.translate(reason, temperature)
                    }
            except BadRequestError as e:
                print(e)
                print(question)
                return None

    async def process_dataset(self, semaphore, progress_bar):
        tasks = []
        for temperature in self.temperatures:
            for question, answer in zip(self.questions, self.answers):
                tasks.append(self.process_data_item(question, answer, temperature, semaphore))

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Data", leave=False):
            result = await f
            if result is not None:
                self.results.append(result)
            progress_bar.update(1)

    def _load_prompt(self, prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        return prompt

    def parse_result(self, json_str: str):
        return parse_json(json_str)

    def save_results(self):
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, "test_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
