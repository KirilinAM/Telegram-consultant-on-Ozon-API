import os
import pandas as pd
import ast
import openai
from openai import OpenAI 
from scipy import spatial  # вычисляет сходство векторов
import tiktoken  # для подсчета токенов

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use
EMBEDDING_MODEL = "text-embedding-ada-002"  # Модель токенизации от OpenAI



# Функция поиска
def strings_ranked_by_relatedness(
    query: str, # пользовательский запрос
    df: pd.DataFrame, # DataFrame со столбцами text и embedding (база знаний)
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), # функция схожести, косинусное расстояние
    top_n: int = 100 # выбор лучших n-результатов
) -> tuple[list[str], list[float]]: # Функция возвращает кортеж двух списков, первый содержит строки, второй - числа с плавающей запятой
    """Возвращает строки и схожести, отсортированные от большего к меньшему"""

    # Отправляем в OpenAI API пользовательский запрос для токенизации
    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    # Получен токенизированный пользовательский запрос
    query_embedding = query_embedding_response.data[0].embedding

    # Сравниваем пользовательский запрос с каждой токенизированной строкой DataFrame
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]

    # Сортируем по убыванию схожести полученный список
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)

    # Преобразовываем наш список в кортеж из списков
    strings, relatednesses = zip(*strings_and_relatednesses)

    # Возвращаем n лучших результатов
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Возвращает число токенов в строке для заданной модели"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Функция формирования запроса к chatGPT по пользовательскому вопросу и базе знаний
def query_message(
    query: str, # пользовательский запрос
    df: pd.DataFrame, # DataFrame со столбцами text и embedding (база знаний)
    model: str, # модель
    token_budget: int # ограничение на число отсылаемых токенов в модель
) -> str:
    """Возвращает сообщение для GPT с соответствующими исходными текстами, извлеченными из фрейма данных (базы знаний)."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df) # функция ранжирования базы знаний по пользовательскому запросу
    # Шаблон инструкции для chatGPT
    message = 'Use the following parts of the Ozon API documentation to answer the following question. If the answer is not found in the documentation, write "Я не смог найти ответ"'
    # Шаблон для вопроса
    question = f"\n\nQuestion: {query}"

    # Добавляем к сообщению для chatGPT релевантные строки из базы знаний, пока не выйдем за допустимое число токенов
    for string in strings:
        next_article = f'\n\nParts of the Ozon API:\n"""\n{string}\n"""'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str, # пользовательский запрос
    df: pd.DataFrame, # DataFrame со столбцами text и embedding (база знаний)
    model: str = GPT_MODEL, # модель
    token_budget: int = 4096 - 500, # ограничение на число отсылаемых токенов в модель
    print_message: bool = False, # нужно ли выводить сообщение перед отправкой
) -> str:
    """Отвечает на вопрос, используя GPT и базу знаний."""
    # Формируем сообщение к chatGPT (функция выше)
    message = query_message(query, df, model=model, token_budget=token_budget)
    # Если параметр True, то выводим сообщение
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You're answering questions about the Ozon API. Отвечай по русски."},
        {"role": "user", "content": message},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0 # гиперпараметр степени случайности при генерации текста. Влияет на то, как модель выбирает следующее слово в последовательности.
    )
    response_message = response.choices[0].message.content
    return response_message

def ask_on_ozon_api(
    query: str, # пользовательский запрос
    model: str = GPT_MODEL, # модель
    token_budget: int = 4096 - 500, # ограничение на число отсылаемых токенов в модель
    print_message: bool = False, # нужно ли выводить сообщение перед отправкой
) -> str:
    """Запрос по документации к API Ozon"""
    embeddings_path = "embeddings.csv"

    df = pd.read_csv(embeddings_path)

    # Конвертируем наши эмбединги из строк в списки
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

    return ask(query=query,df=df,model=model,token_budget=token_budget,print_message=print_message)

if __name__ == '__main__':
    print(ask_on_ozon_api('Какой метод получает инофрмацию о товарах?'))