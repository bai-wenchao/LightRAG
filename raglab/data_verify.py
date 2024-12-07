#! /usr/bin/env python3

import os
from pathlib import Path
from openai import OpenAI

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent


def openai_complete_if_cache(
    model="gpt-4o", 
    prompt=None, 
    system_prompt=None, 
    history_messages=[],
    api_key=None,
    base_url=None,
    **kwargs
) -> str:
    print("Start the query.")
    print(f"api_key: {api_key}")
    print(f"base_url: {base_url}")
    openai_client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = openai_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    print("Complete the query.")
    return response.choices[0].message.content


def load_newsdata(file_path: str) -> str:
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def generate_qa(input_path:str, output_path:str) -> None:
    newsdata = load_newsdata(file_path=input_path)

    prompt = f"""
    Given the following news data:

    {newsdata}

    Please identify 3 potential users who would engage with this news. 
    For each user, list 3 tasks they would perform with this news. 
    Then, for each (user, task) combination, 
    generate 5 questions that require a high-level understanding of the entire news data.
    Finally, provide the answer for each questions.
    
    ===
    NOTE: All questions and answers should be highly corelated with the provided news.
    In other word, they cannot be too subjective and can only found in the provided news data.
    ===

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Answer 1:
            - Question 2:
            - Answer 2:
            - Question 3:
            - Answer 3:
        - Task 2: [task description]
        - Task 3: [task description]
    ...
    - User 3: [user description]
        ...
    """

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    result = openai_complete_if_cache(model="qwen-max", 
                                      api_key=api_key,
                                      base_url=base_url,
                                      prompt=prompt)

    with open(output_path, 'w') as file:
        file.write(result)


def generate_qawok(input_path: str, output_path: str) -> None:

    with open(input_path, 'r') as file:
        qa_data = file.read()

    qaa = []
    for line in qa_data.splitlines():
        if "Question" not in line:
            continue

        qaa_item = {}
        qaa_item['id'] = int(line[19])
        qaa_item['q'] = line[22:]
        qaa_item['awok'] = ""

        qaa.append(qaa_item)

    with open(output_path, 'w') as file:
        for qaa_item in qaa:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

            retries, max_retries = 0, 3
            while retries < max_retries:
                try:
                    qaa_item['awok'] = openai_complete_if_cache(model="qwen-max", 
                                                        api_key=api_key,
                                                        base_url=base_url,
                                                        prompt=qaa_item['q'])
                    break
                except Exception as e:
                    retries += 1
                    print(f"Retry {retries}, {e}")

            file.write(f"Question {qaa_item['id']}: {qaa_item['q']}\n")
            file.write(f"Answer without knowledge {qaa_item['id']}: {qaa_item['awok']}\n")
    


if __name__ == "__main__":
    CATEGORY = "politics"
    TITLE = "health_insurance_GOP"

    input_file_path = os.path.join(
        PROJECT_ROOT, 
        f"datasets/newsdata/{CATEGORY}/{TITLE}.txt")
    
    output_file_path = os.path.join(
        PROJECT_ROOT, 
        f"datasets/newsdata/{CATEGORY}/QA-{TITLE}.txt")
    
    input_qa_path = os.path.join(
        PROJECT_ROOT, 
        f"datasets/newsdata/{CATEGORY}/QA-{TITLE}.txt")
    output_qa_path = os.path.join(
        PROJECT_ROOT,
        f"datasets/newsdata/{CATEGORY}/QAwoK-{TITLE}.txt")
    
    # generate_qa(input_path=input_file_path, output_path=output_file_path)
    generate_qawok(input_path=input_qa_path, output_path=output_qa_path)
