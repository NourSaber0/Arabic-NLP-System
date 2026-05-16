import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env", override=True)
load_dotenv(BASE_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

models = client.models.list()
for m in models.data:
    print(m.id)
