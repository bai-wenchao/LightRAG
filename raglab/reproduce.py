#! /usr/bin/env python3

import os
import json
import glob
import time

from transformers import GPT2Tokenizer
from openai import OpenAI


def extract_unique_contexts(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(input_directory, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files.")

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_unique_contexts.json"
        output_path = os.path.join(output_directory, output_filename)

        unique_contexts_dict = {}

        print(f"Processing file: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get("context")
                        if context and context not in unique_contexts_dict:
                            unique_contexts_dict[context] = None
                    except json.JSONDecodeError as e:
                        print(
                            f"JSON decoding error in file {filename} at line {line_number}: {e}"
                        )
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        print(
            f"There are {len(unique_contexts_list)} unique `context` entries in the file {filename}."
        )

        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(unique_contexts_list, outfile,
                          ensure_ascii=False, indent=4)
            print(
                f"Unique `context` entries have been saved to: {output_filename}")
        except Exception as e:
            print(
                f"An error occurred while saving to the file {output_filename}: {e}")

    print("All files have been processed.")


def get_context_summary(context, tot_tokens=2000):
    model_path = os.getenv("MODELSCOPE_HOME") + "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000: 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens): 1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary


def openai_complete_if_cache(
    model, api_key_env, llm_base_url, prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_client = OpenAI(
        api_key=os.getenv(api_key_env),
        base_url=llm_base_url,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    print("Sending request via API.")
    
    response = openai_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    print("Received response from API.")

    return response.choices[0].message.content


def generate_questions_from_context_summary(dataset_dir, ctx_cls, llm_model_name, api_key_env, llm_base_url):
    if not os.path.exists(f"{dataset_dir}questions/"):
        os.makedirs(f"{dataset_dir}questions/")

    for cls in ctx_cls:
        with open(f"{dataset_dir}unique_contexts/{cls}_unique_contexts.json", mode="r") as f:
            unique_contexts = json.load(f)

        summaries = [get_context_summary(context)
                     for context in unique_contexts]
        print(f"Generated {len(summaries)} summaries for {cls}.")

        total_description = "\n\n".join(summaries)

        prompt = f"""
        Given the following description of a dataset:

        {total_description}

        Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

        Output the results in the following structure:
        - User 1: [user description]
            - Task 1: [task description]
                - Question 1:
                - Question 2:
                - Question 3:
                - Question 4:
                - Question 5:
            - Task 2: [task description]
                ...
            - Task 5: [task description]
        - User 2: [user description]
            ...
        - User 5: [user description]
            ...
        """

        retry, max_retry = 0, 5

        while retry < max_retry:
            try:
                print(f"Generating questions for {cls}. {retry}-th try.")
                result = openai_complete_if_cache(
                    model=llm_model_name, api_key_env=api_key_env, llm_base_url=llm_base_url, prompt=prompt)
                break
            except Exception as e:
                retry += 1
                print(
                    f"Failed to generate questions for {cls}. Retrying ({retry}/{max_retry}). Error: {e}")
                time.sleep(10)

        file_path = f"{dataset_dir}questions/{cls}_questions.txt"
        with open(file_path, "w") as file:
            file.write(result)

        print(f"{cls}_questions written to {file_path}")
