#! /usr/bin/env python3

import os
from pathlib import Path
from raglab import reproduce, ConfigManager, RAGManager

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent


if __name__ == "__main__":
    print("=== LAUNCHED KG CONSTRUCTOR ===")

    config_name = "newsdata"
    config_path = os.path.join(PROJECT_ROOT, f"raglab/config/{config_name}.yaml")

    config_manager = ConfigManager(config_path=config_path)
    print("Initialized config maneger.")

    rag_manager = RAGManager(config_manager=config_manager)
    print("Initialized RAG manager.")

    dataset = "newsdata/politics"
    file_name = "health_insurance_GOP.txt"
    insert_file_path = os.path.join(PROJECT_ROOT, f"datasets/{dataset}/{file_name}")
    print("---> Starting inserting article...")
    rag_manager.insert_article(file_path=insert_file_path)
    print("Completed article insersion")
