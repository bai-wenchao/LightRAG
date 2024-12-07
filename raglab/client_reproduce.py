#! /usr/bin/env python3

from raglab import reproduce, ConfigManager, RAGManager, GraphVis


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


# 2. Generate questions.
def step_2(config_manager: ConfigManager, ctx_cls: list[str]) -> None:
    print("[STEP-2] Generate questions.")
    dataset_dir = config_manager.dataset_dir
    llm_model_name = config_manager.llm_model_name
    api_key_env = config_manager.api_key_env
    llm_base_url = config_manager.llm_base_url
    reproduce.generate_questions_from_context_summary(
        dataset_dir, ctx_cls, llm_model_name, api_key_env, llm_base_url)


def visualize(config_manager: ConfigManager) -> None:
    print("[VISUALIZE] Visualizing.")
    vis = GraphVis(config_manager)
    vis.convert()


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
        # {
        #     'func': step_1,  # Insert text.
        #     'args': [config_manager, rag_manager]
        # },
        # {
        #     'func': step_2,  # Insert text.
        #     'args': [config_manager, ['cs']]
        # },
        {
            'func': visualize,  # Visualize.
            'args': [config_manager]
        }
    ]

    for step in workflow:
        func = step['func']
        args = step['args']

        func(*args)
