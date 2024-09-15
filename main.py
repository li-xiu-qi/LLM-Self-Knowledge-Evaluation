from datasets import load_dataset

from chat import LLM
from config import DATASET, MODEL_NAME, DATA_TYPE
from self_knowledge import LLMSelfKnowledge

if __name__ == "__main__":
    llm = LLM(model_name=MODEL_NAME)
    rag_dataset = load_dataset(DATASET)
    data_set = rag_dataset[DATA_TYPE]
    llm_self_knowledge = LLMSelfKnowledge(llm, data_set)
    llm_self_knowledge.main()
