#! /usr/bin/env python3

from raglab import reproduce, ConfigManager, RAGManager


# 0. Extract unique contexts.
def step_0(config_manager: ConfigManager) -> None:
    print("[STEP-0] Extract unique contexts.")
    print(f"[STEP-0] from {config_manager.dataset_dir}.")
    print(f"[STEP-0] to {config_manager.context_dir}.")
    reproduce.extract_unique_contexts(
        config_manager.dataset_dir, config_manager.context_dir)


# 1. Insert text.
def step_1(
        config_manager: ConfigManager,
        rag_manager: RAGManager) -> None:
    print("[STEP-1]: Insert text.")
    context_file_path = \
        f"{config_manager.context_dir}{config_manager.data_class}_unique_contexts.json"
    print(f"[STEP-1]: context_file_path: {context_file_path}")
    rag_manager.insert_text(context_file_path)


if __name__ == '__main__':
    print("=== LAUNCHED PIPELINE. ===")

    config_manager = ConfigManager("config/reproduce.yaml")
    print("Initialized config manager.")

    rag_manager = RAGManager(config_manager=config_manager)
    print("Initialized RAG manager.")

    workflow = [
        # {
        #     'func': step_0,  # Extract unique contexts.
        #     'args': [config_manager]
        # },
        {
            'func': step_1,  # Insert text.
            'args': [config_manager, rag_manager]
        },
    ]

    for step in workflow:
        func = step['func']
        args = step['args']

        func(*args)
