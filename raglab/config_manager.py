#! /usr/bin/env python3

import os

import yaml


class ConfigManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.dataset_home = config['dataset_home']
        self.dataset_dir = config['dataset_dir']

        self.working_dir = self.dataset_home + \
            self.dataset_dir + config['working_dir']
        self.doc_file = self.dataset_home + \
            self.dataset_dir + config['doc_file']
        self.prompts_file = self.dataset_home + \
            self.dataset_dir + config['prompts_file']

        self.llm_model_type = config['llm_model_type']
        self.embedding_model_type = config['embedding_model_type']

        self.llm_model_name = config['llm_model_name']
        self.embedding_model_name = config['embedding_model_name']

        self.api_key_env = config['api_key_env']
        self.llm_base_url = config['llm_base_url']
        self.host = config['host']
        self.num_ctx = config['num_ctx']

        self.llm_model_max_async = config['llm_model_max_async']
        self.llm_model_max_token_size = config['llm_model_max_token_size']
        self.embedding_dim = config['embedding_dim']
        self.embedding_max_token_size = config['embedding_max_token_size']

        self.query_mode = config['query_mode']

        with open(self.prompts_file, 'r') as f:
            self.prompts = yaml.safe_load(f)

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def show_config(self):
        print("*************** Config ***************")
        print(f"working_dir: {self.working_dir}")
        print(f"doc_file: {self.doc_file}")
        print(f"prompt_file: {self.prompts_file}")

        print(f"llm_model_type: {self.llm_model_type}")
        print(f"embedding_model_type: {self.embedding_model_type}")

        print(f"llm_model_name: {self.llm_model_name}")
        print(f"embedding_model: {self.embedding_model_name}")

        print(f"api_key_env: {self.api_key_env}")
        print(f"llm_base_url: {self.llm_base_url}")
        print(f"host: {self.host}")
        print(f"num_ctx: {self.num_ctx}")

        print(f"llm_model_max_async: {self.llm_model_max_async}")
        print(f"llm_model_max_token_size: {self.llm_model_max_token_size}")
        print(f"embedding_dim: {self.embedding_dim}")
        print(f"embedding_max_token_size: {self.embedding_max_token_size}")

        print(f"query_mode: {self.query_mode}")
        print("*************** End ***************")

    def get_first_default_prompt(self) -> str:
        return self.prompts['default'][0]
