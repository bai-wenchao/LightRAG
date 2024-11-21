#! /usr/bin/env python3

import json
import time

from lightrag import QueryParam

from raglab import ConfigManager, RAGFactory


class RAGManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.rag_factory = RAGFactory(self.config_manager)
        self.rag_inst = self.rag_factory.rag_inst

    def insert_text(self) -> None:
        with open(self.config_manager.doc_file, 'r', encoding='utf-8') as file:
            self.rag_inst.insert(file.read())

    def insert_text(self, file_path) -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            unique_contexts = json.load(file)
        
        retries = 0
        max_retries = self.config_manager.text_insersion_max_retries
        while retries < max_retries:
            try:
                self.rag_inst.insert(unique_contexts)
                break
            except Exception as e:
                retries += 1
                print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
                time.sleep(10)
        if retries == max_retries:
            print("Insertion failed after exceeding the maximum number of retries")
    

    def exact_query(self, prompt: str, mode: str) -> str:
        result = self.rag_inst.query(prompt, param=QueryParam(mode=mode))
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")

        return result

    def query(self, prompt: str) -> dict:
        res_dict = {}
        for mode in self.config_manager.query_mode:
            print(f"*** {mode} search ***")
            res_dict[mode] = self.exact_query(prompt, mode)
        return res_dict

    def first_item_query(self) -> dict:
        prompt = self.config_manager.get_first_default_prompt()
        res_dict = {}
        for mode in self.config_manager.query_mode:
            print(f"*** {mode} search ***")
            res_dict[mode] = self.exact_query(prompt, mode)
        return res_dict
