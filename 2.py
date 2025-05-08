import dotenv
import os
import warnings

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate


warnings.filterwarnings('ignore') # ігнорувати warnings
dotenv.load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3',
    temperature = 0.7,
)

zero_shot_prompt = '''
[INST]Сформуй детальний план навчального курсу на тему "олімпіадні задачі з математики" для цільової аудиторії діти 10-12 років.
Врахуй рівень підготовки, вік, потреби і очікувані результати навчання.[/INST]
'''
response = llm.invoke(zero_shot_prompt)
print(response)

print("*"*40)

few_shot_prompt = '''
[INST]Сформуй детальний план навчального курсу на тему "олімпіадні задачі з математики" для цільової аудиторії діти 10-12 років.
Врахуй рівень підготовки, вік, потреби і очікувані результати навчання.[/INST]

[INST]Ось приклад плану навчального курсу:

Тема: "Основи веброзробки"
Цільова аудиторія: підлітки 14–16 років, які хочуть створювати власні сайти без попереднього досвіду.
План курсу:
1. Вступ: що таке веброзробка
2. Основи HTML: структура сторінки
3. CSS: оформлення сторінки
4. Простий проєкт — особиста вебсторінка
5. Основи JavaScript: інтерактивність
6. Практика: створення міні-сайту-портфоліо
7. Як розмістити сайт онлайн
8. Підсумковий проєкт
[/INST]
'''

response = llm.invoke(few_shot_prompt)
print(response)