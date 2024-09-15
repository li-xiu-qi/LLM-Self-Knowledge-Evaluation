import os
from llm_parse_json import parse_json
from chat import LLM
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


class LLMSelfKnowledge:
    base_path = os.path.dirname(__file__)
    is_known_prompt_path = os.path.join(base_path, "prompts", "is_known_prompt.txt")
    verify_answer_prompt_path = os.path.join(base_path, "prompts", "verify_answer_prompt.txt")
    def __init__(self, llm: LLM, data_set):
        self.llm = llm
        if data_set:
            self.questions = [q for q in data_set['question']]
            self.answers = [a for a in data_set['answer']]
        self.results = []

    def is_known(self, question):
        model_config = {
            "temperature": 0.0,
        }
        is_known_prompt = self._load_prompt(self.is_known_prompt_path)
        messages = [{"role": "system", "content": is_known_prompt},
                    {"role": "user", "content": question}]
        result = self.llm(messages=messages, **model_config)
        return result

    def verify_answer(self, response, standard_answer):
        model_config = {
            "temperature": 0.0,
        }
        verify_answer_prompt = self._load_prompt(self.verify_answer_prompt_path)
        user_input = f"""- **Input Response**: {response}
                        - **Standard Answer**: {standard_answer}
                     """
        messages = [{"role": "system", "content": verify_answer_prompt},
                    {"role": "user", "content": user_input}]
        result = self.llm(messages=messages, **model_config)
        return result

    def main(self):
        for question, answer in tqdm(zip(self.questions, self.answers), total=len(self.questions)):
            is_known_result = self.is_known(question)
            parse_is_known_result = self.parse_result(is_known_result)
            is_known = parse_is_known_result["is_known"]
            is_correct = True
            response = parse_is_known_result["answer"]
            if is_known:
                verify_answer_result = self.verify_answer(response=response, standard_answer=answer)
                parse_verify_answer_result = self.parse_result(verify_answer_result)
                is_correct = parse_verify_answer_result["is_correct"]

            self.results.append({
                "question": question,
                "is_known": is_known,
                "answer": answer,
                "response": response,
                "is_correct": is_correct
            })
        self.save_results()
        self.visualize_results()

    def _load_prompt(self, prompt_file_path):
        with open(prompt_file_path) as f:
            prompt = f.read()
        return prompt

    def parse_result(self, result):
        return parse_json(result)

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv("results.csv", index=False)

    def visualize_results(self):
        df = pd.DataFrame(self.results)
        known_counts = df["is_known"].value_counts()
        correct_counts = df["is_correct"].value_counts()

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        sns.barplot(x=known_counts.index, y=known_counts.values, ax=axes[0, 0], palette="viridis")
        axes[0, 0].set_title("Known Questions", fontsize=16)
        axes[0, 0].set_xlabel("Is Known", fontsize=14)
        axes[0, 0].set_ylabel("Count", fontsize=14)
        axes[0, 0].tick_params(axis='x', labelsize=12)
        axes[0, 0].tick_params(axis='y', labelsize=12)
        for i, v in enumerate(known_counts.values):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontsize=12)

        sns.barplot(x=correct_counts.index, y=correct_counts.values, ax=axes[0, 1], palette="viridis")
        axes[0, 1].set_title("Correct Answers", fontsize=16)
        axes[0, 1].set_xlabel("Is Correct", fontsize=14)
        axes[0, 1].set_ylabel("Count", fontsize=14)
        axes[0, 1].tick_params(axis='x', labelsize=12)
        axes[0, 1].tick_params(axis='y', labelsize=12)
        for i, v in enumerate(correct_counts.values):
            axes[0, 1].text(i, v + 0.5, str(v), ha='center', fontsize=12)

        axes[1, 0].pie(known_counts, labels=known_counts.index, autopct='%1.1f%%', startangle=140,
                       colors=sns.color_palette("viridis", len(known_counts)))
        axes[1, 0].set_title("Known Questions Distribution", fontsize=16)

        axes[1, 1].pie(correct_counts, labels=correct_counts.index, autopct='%1.1f%%', startangle=140,
                       colors=sns.color_palette("viridis", len(correct_counts)))
        axes[1, 1].set_title("Correct Answers Distribution", fontsize=16)

        plt.tight_layout()
        plt.savefig("test_results_visualization.png")
        plt.show()