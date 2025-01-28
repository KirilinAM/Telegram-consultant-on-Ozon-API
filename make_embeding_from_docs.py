import os
from bs4 import BeautifulSoup
from itertools import chain
import pandas as pd
from openai import OpenAI
import re 
import tiktoken  # для подсчета токенов
from collections import defaultdict

import API_KEYS

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use
EMBEDDING_MODEL = "text-embedding-ada-002"  # Модель токенизации от OpenAI

def is_tag(element):
    """Проверяет, является ли элемент тегом."""
    return element.name is not None

def is_header(element):
    """Проверяет, является ли элемент заголовком."""
    return element.name.startswith('h') and element.name[1:].isdigit()

def contains_heading_tags(element):
    """Проверяет, содержит ли элемент вложенные заголовки."""
    return bool(element.find(is_header))

def update_header_hierarchy(hierarchy, header):
    """Обновляет иерархию заголовков."""
    header_level = int(header.name[-1])  # Уровень текущего заголовка (1 для h1, 2 для h2 и т.д.)
    header_text = header.get_text(strip=True)  # Текст текущего заголовка

    # Удаляем все заголовки с уровнем >= текущего
    for level in list(hierarchy.keys()):  # Используем list для безопасного удаления
        if level >= header_level:
            del hierarchy[level]

    # Добавляем текущий заголовок (уровень и текст)
    hierarchy[header_level] = header_text

def process_element(element, hierarchy=None, sections=None) -> dict[int:str]:
    if hierarchy is None:
        hierarchy = {}  # Инициализация иерархии как словаря
    if sections is None:
        sections = defaultdict(str)  # Инициализация словаря с пустыми строками по умолчанию

    while element is not None:
        if is_tag(element):
            if is_header(element):
                update_header_hierarchy(hierarchy, element)
            else:
                if contains_heading_tags(element):
                    first_child = element.find()
                    process_element(first_child, hierarchy, sections)
                else:
                    # Создаем ключ для sections, используя значения hierarchy
                    hierarchy_key = tuple(hierarchy.values())
                    text = element.get_text(strip=True,separator='\n')
                    sections[hierarchy_key] += "\n" + text if sections[hierarchy_key] else text
        else:
            text = element.strip()
            if text:
                # Создаем ключ для sections, используя значения hierarchy
                hierarchy_key = tuple(hierarchy.values())
                sections[hierarchy_key] += "\n" + text if sections[hierarchy_key] else text
        element = element.next_sibling

    return sections

def all_section_with_text(html,preheader):
    soup = BeautifulSoup(html, "html.parser")
    hierarchy = {0:preheader}
    element = soup.find()
    sections_with_text = [[list(sections), text] for sections, text in process_element(element,hierarchy=hierarchy).items()]
    return sections_with_text

# Очистка текста секции от ссылок <ref>xyz</ref>, начальных и конечных пробелов
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    titles, text = section
    # Удаляем ссылки
    text = re.sub(r"<ref.*?</ref>", "", text)
    # Удаляем пробелы вначале и конце
    text = text.strip()
    return (titles, text)

# Отфильтруем короткие и пустые секции
def keep_section(section: tuple[list[str], str], len_trashold = 16) -> bool:
    """Возвращает значение True, если раздел должен быть сохранен, в противном случае значение False."""
    titles, text = section
    # Фильтруем по произвольной длине, можно выбрать и другое значение
    if len(text) < len_trashold:
        return False
    else:
        return True
    
# Функция подсчета токенов
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Возвращает число токенов в строке."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Функция разделения строк
def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Разделяет строку надвое с помощью разделителя (delimiter), пытаясь сбалансировать токены с каждой стороны."""

    # Делим строку на части по разделителю, по умолчанию \n - перенос строки
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # разделитель не найден
    elif len(chunks) == 2:
        return chunks  # нет необходимости искать промежуточную точку
    else:
        # Считаем токены
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        # Предварительное разделение по середине числа токенов
        best_diff = halfway
        # В цикле ищем какой из разделителей, будет ближе всего к best_diff
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        # Возвращаем левую и правую часть оптимально разделенной строки
        return [left, right]

# Функция обрезает строку до максимально разрешенного числа токенов
def truncated_string(
    string: str, # строка
    model: str, # модель
    max_tokens: int, # максимальное число разрешенных токенов
    print_warning: bool = True, # флаг вывода предупреждения
) -> str:
    """Обрезка строки до максимально разрешенного числа токенов."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    # Обрезаем строку и декодируем обратно
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Предупреждение: Строка обрезана с {len(encoded_string)} токенов до {max_tokens} токенов.")
    # Усеченная строка
    return truncated_string

# Функция делит секции статьи на части по максимальному числу токенов
def split_strings_from_subsection(
    subsection: tuple[list[str], str], # секции
    max_tokens: int = 1000, # максимальное число токенов
    model: str = GPT_MODEL, # модель
    max_recursion: int = 5, # максимальное число рекурсий
) -> list[str]:
    """
    Разделяет секции на список из частей секций, в каждой части не более max_tokens.
    Каждая часть представляет собой кортеж родительских заголовков [H1, H2, ...] и текста (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # Если длина соответствует допустимой, то вернет строку
    if num_tokens_in_string <= max_tokens:
        return [string]
    # если в результате рекурсия не удалось разделить строку, то просто усечем ее по числу токенов
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # иначе разделим пополам и выполним рекурсию
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]: # Пробуем использовать разделители от большего к меньшему (разрыв, абзац, точка)
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # если какая-либо половина пуста, повторяем попытку с более простым разделителем
                continue
            else:
                # применим рекурсию на каждой половине
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1, # уменьшаем максимальное число рекурсий
                    )
                    results.extend(half_strings)
                return results
    # иначе никакого разделения найдено не было, поэтому просто обрезаем строку (должно быть очень редко)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

# Функция отправки chatGPT строки для ее токенизации (вычисления эмбедингов)
def get_embedding(text, model="text-embedding-ada-002"):
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def main():
    MAX_TOKENS = 1600
    SAVE_PATH = "./embeddings.csv"

    doc_folder = 'ozon docs'
    doc_names_and_preheaders = [
        ['seller.html','Ozon Seller API']
        ,['performance.html','Ozon Performance API']
    ]

    preheaders_and_html = []
    for name, preheader in doc_names_and_preheaders:
        path = os.path.join(doc_folder,name)
        with open(path, "r", encoding="utf-8") as file:
            html = file.read()
        preheaders_and_html.append([preheader,html])

    sections = []
    for preheader, html in preheaders_and_html:
        sections += all_section_with_text(html,preheader)
    
    sections = [clean_section(ws) for ws in sections]
    sections = [ws for ws in sections if keep_section(ws)]

    strings = []
    for section in sections:
        strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS, max_recursion=10))
    sections = strings

    df = pd.DataFrame({"text": sections[:10]})
    # df = pd.DataFrame({"text": sections})

    # df['embedding'] = df.text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

    df.to_csv(SAVE_PATH, index=False)

if __name__ == "__main__":
    main()