#! /usr/bin/env python3

import os

from lightrag import LightRAG
from lightrag.llm import openai_complete_if_cache, ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

from raglab import ConfigManager


class RAGFactory:
    def __init__(self, config_manager: ConfigManager):
        if config_manager.llm_model_type == "ollama":
            # LLM model function.
            self.llm_model_func = ollama_model_complete

            # LightRAG isntance.
            self.rag_inst = LightRAG(
                working_dir=config_manager.working_dir,
                llm_model_func=self.llm_model_func,
                llm_model_name=config_manager.llm_model_name,
                llm_model_max_async=config_manager.llm_model_max_async,
                llm_model_max_token_size=config_manager.llm_model_max_token_size,
                llm_model_kwargs={
                    "host": config_manager.host, "options": {"num_ctx": config_manager.num_ctx}},
                embedding_func=EmbeddingFunc(
                    embedding_dim=config_manager.embedding_dim,
                    max_token_size=config_manager.embedding_max_token_size,
                    func=lambda texts: ollama_embedding(
                        texts, embed_model=config_manager.embedding_model_name, host=config_manager.host
                    ),
                ),
            )
            print("[Factory] Built an ollama RAG instance.")
            print(f"[Factory] working dir: {config_manager.working_dir}")

        elif config_manager.llm_model_type == "openai-like":
            # LLM model function.
            async def llm_model_func(
                prompt, system_prompt=None, history_messages=[], **kwargs
            ) -> str:
                return await openai_complete_if_cache(
                    model=config_manager.llm_model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=os.getenv(config_manager.api_key_env),
                    base_url=config_manager.llm_base_url,
                    **kwargs
                )
            self.llm_model_func = llm_model_func

            # LightRAG instance.
            self.rag_inst = LightRAG(
                working_dir=config_manager.working_dir,
                llm_model_func=self.llm_model_func,
                llm_model_name=config_manager.llm_model_name,
                llm_model_max_async=config_manager.llm_model_max_async,
                llm_model_max_token_size=config_manager.llm_model_max_token_size,
                embedding_func=EmbeddingFunc(
                    embedding_dim=config_manager.embedding_dim,
                    max_token_size=config_manager.embedding_max_token_size,
                    func=lambda texts: ollama_embedding(
                        texts, embed_model=config_manager.embedding_model_name, host=config_manager.host
                    ),
                ),
            )
            print("[Factory] Built an openai-like RAG instance.")
            print(f"[Factory] working dir: {config_manager.working_dir}")
