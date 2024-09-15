import asyncio

import numpy as np
from datasets import load_dataset

from async_self_knowledge import LLMSelfKnowledge
from chat import AsyncLLM
from tqdm.asyncio import tqdm

from config import SEMAPHORE_MAX_CONCURRENT_REQUESTS, MODEL_NAME, DATASET, DATA_TYPE
from visual import Visualization


async def run_async_process(llm, data_set, temperatures, semaphore):
    llm_self_knowledge = LLMSelfKnowledge(llm, data_set, temperatures)

    total_items = len(data_set) * len(temperatures)
    progress_bar = tqdm(total=total_items, desc="Overall Progress")
    await llm_self_knowledge.process_dataset(semaphore, progress_bar)
    progress_bar.close()

    visualization = Visualization(output_dir=llm_self_knowledge.output_dir)
    visualization.visualize_results(llm_self_knowledge.results)


if __name__ == "__main__":
    llm = AsyncLLM(model_name=MODEL_NAME)
    rag_dataset = load_dataset(DATASET)
    data_set = rag_dataset[DATA_TYPE]
    temperatures = [round(r, 2) for r in np.arange(0, 0.5, 0.1)]
    semaphore = asyncio.Semaphore(SEMAPHORE_MAX_CONCURRENT_REQUESTS)

    asyncio.run(run_async_process(llm, data_set, temperatures, semaphore))
