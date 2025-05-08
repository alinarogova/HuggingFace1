#from langchain.prompts import PromptTemplate

import dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint

dotenv.load_dotenv()

res = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    #top_k = 3,
    #top_p = 0.8,
    temperature =0.6,
    max_new_tokens = 100
)
instruction = (
    "[INST]Ти — Шерлок Голмс, відомий детектив з Бейкер-стріт. "
    "Ти надзвичайно логічний, уважний до деталей, іноді зарозумілий. "
    "Спілкуєшся мовою освіченого джентльмена з вікторіанської епохи.[/INST]"
)

chat_history = f"{instruction}\n"

print("Чат із Шерлоком Голмсом. Введіть 'вийти' для завершення.")
while True:
    user_input = input("Ви: ")
    if user_input.lower() == 'вийти':
        break

    chat_history += f"Human: {user_input}\nAI:"
    response = llm.invoke(chat_history)
    print("Шерлок:", response.strip())

    chat_history += f" {response.strip()}\n"
