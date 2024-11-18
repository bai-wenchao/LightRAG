#! /usr/bin/env python3

import os
import yaml
from lightrag import LightRAG
from lightrag.llm import openai_complete_if_cache, ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc


class ConfigManager:
    def __init__(self, config_path: str):
        """
        1. Parse the config file.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        dataset_home = config['dataset_home']
        dataset_dir = config['dataset_dir']
        self.working_dir = dataset_home + dataset_dir + config['working_dir']
        self.doc_file = dataset_home + dataset_dir + config['doc_file']

        self.llm_model_type = config['llm_model_type']
        self.embedding_model_type = config['embedding_model_type']

        self.llm_model_name = config['llm_model_name']
        self.embedding_model = config['embedding_model']

        self.api_key_env = config['api_key_env']
        self.llm_base_url = config['llm_base_url']
        self.host = config['host']
        self.num_ctx = config['num_ctx']

        self.llm_model_max_async = config['llm_model_max_async']
        self.llm_model_max_token_size = config['llm_model_max_token_size']
        self.embedding_dim = config['embedding_dim']
        self.embedding_max_token_size = config['embedding_max_token_size']

        self.prompts_file = dataset_home + dataset_dir + config['prompts_file']
        with open(self.prompts_file, 'r') as f:
            self.prompts = yaml.safe_load(f)

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        """
        2. Construct the LLM model function.
        """
        if self.llm_model_type == 'openai-like':
            async def llm_model_func(
                prompt, system_prompt=None, history_messages=[], **kwargs
            ) -> str:
                return await openai_complete_if_cache(
                    model=self.llm_model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=os.getenv(self.api_key_env),
                    base_url=self.llm_base_url,
                    **kwargs
                )
            self.llm_model_func = llm_model_func
        elif self.llm_model_type == 'ollama':
            self.llm_model_func = ollama_model_complete

        """
        3. Construct the LightRAG instance.
        """
        if self.llm_model_type == 'openai-like':
            self.rag_inst = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                llm_model_name=self.llm_model_name,
                llm_model_max_async=self.llm_model_max_async,
                llm_model_max_token_size=self.llm_model_max_token_size,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=self.embedding_max_token_size,
                    func=lambda texts: ollama_embedding(
                        texts, embed_model=self.embedding_model, host=self.host
                    ),
                ),
            )
        elif self.llm_model_type == 'ollama':
            self.rag_inst = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                llm_model_name=self.llm_model_name,
                llm_model_max_async=self.llm_model_max_async,
                llm_model_max_token_size=self.llm_model_max_token_size,
                llm_model_kwargs={
                    "host": self.host, "options": {"num_ctx": self.num_ctx}},
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=self.embedding_max_token_size,
                    func=lambda texts: ollama_embedding(
                        texts, embed_model=self.embedding_model, host=self.host
                    ),
                ),
            )

    def get_first_default_prompt(self) -> str:
        return self.prompts['default'][0]
    
    def show_config(self):
        print("*** Config ***")
        print(f"working_dir: {self.working_dir}")
        print(f"doc_file: {self.doc_file}")
        print(f"prompt_file: {self.prompts_file}")
        print(f"llm_model_type: {self.llm_model_type}")
        print(f"embedding_model_type: {self.embedding_model_type}")
        print(f"llm_model_name: {self.llm_model_name}")
        print(f"embedding_model: {self.embedding_model}")
        print(f"api_key_env: {self.api_key_env}")
        print(f"llm_base_url: {self.llm_base_url}")
        print(f"host: {self.host}")
        print(f"num_ctx: {self.num_ctx}")
        print(f"llm_model_max_async: {self.llm_model_max_async}")
        print(f"llm_model_max_token_size: {self.llm_model_max_token_size}")
        print(f"embedding_dim: {self.embedding_dim}")
        print(f"embedding_max_token_size: {self.embedding_max_token_size}")
        print("*** End ***")
