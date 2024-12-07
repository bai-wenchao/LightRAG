#! /usr/bin/env python3

import os
import time

import json
import logging

from lightrag import QueryParam
from lightrag.utils import logger, set_logger

from raglab import ConfigManager, RAGFactory


class RAGManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.rag_factory = RAGFactory(self.config_manager)
        self.rag_inst = self.rag_factory.rag_inst

    def __post__init__(self):
        log_file = os.path.join(
            self.config_manager.working_dir, 'exec_progress.log')
        set_logger(log_file)
        logger.setLevel(logging.DEBUG)
        logger.info("RAGManagerLogger initialized to {log_file}.")

    def insert_text(self) -> None:
        with open(self.config_manager.doc_file, 'r', encoding='utf-8') as file:
            self.rag_inst.insert(file.read())

    def insert_text(self, file_path) -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            unique_contexts = json.load(file)

        start_ctx = 8

        for ctx_idx, ctx in enumerate(unique_contexts):
            if ctx_idx < start_ctx:
                continue
            retries = 0
            max_retries = self.config_manager.text_insersion_max_retries
            while retries < max_retries:
                logger.info(
                    f"[INSERT-DOC] Inserting the {ctx_idx} / {len(unique_contexts)} ctx object. " \
                    f"{retries}/{max_retries}-th try.")
                try:
                    self.rag_inst.insert(ctx)
                    break
                except Exception as e:
                    retries += 1
                    logger.info(
                        f"[FAIL-DOC] Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
                    time.sleep(10)
            if retries == max_retries:
                logger.info(
                    "[RETRY-DOC] Insertion failed after exceeding the maximum number of retries")
            logger.info(
                f"Complete the {ctx_idx} / {len(unique_contexts)} ctx object.")

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
