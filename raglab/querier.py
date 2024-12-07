#! /usr/bin/env python3

import logging

from lightrag import QueryParam

from lightrag.utils import logger, set_logger
from raglab import ConfigManager, RAGManager


if __name__ == "__main__":

    set_logger("query.log")
    logging.basicConfig(level=logging.INFO)

    config_manager = ConfigManager("config/reproduce.yaml")

    rag_manager = RAGManager(config_manager=config_manager)

    query_mode = [
        'direct',
        # 'naive',
        # 'local',
        # 'global',
        'hybrid',
    ]

    query = f"""
    Introduce the person Roger Zauner.
    You can output at most 256 words.
    """

    for mode in query_mode:
        logger.info(f"Query mode: {mode}")
        logger.info(f"Query: {query}")
        sys_prompt, result = rag_manager.rag_inst.query(query, param=QueryParam(mode=mode))
        logger.info(f"System prompt: {sys_prompt}")
        logger.info(f"Result: {result}")
