import os

from dotenv import load_dotenv

API_KEY = None

BASE_URL = None

MODEL_NAME = "glm-4-flash"
SEMAPHORE_MAX_CONCURRENT_REQUESTS = 200
DATASET = "neural-bridge/rag-dataset-1200"
DATA_TYPE = "test"

if API_KEY is None:
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    BASE_URL = os.getenv('BASE_URL')
