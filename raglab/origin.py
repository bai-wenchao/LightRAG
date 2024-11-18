import os

import yaml

from lightrag import QueryParam

from raglab import ConfigManager

if __name__ == "__main__":
    config_manager = ConfigManager("config/origin.yaml")

    rag = config_manager.rag_inst

    with open(config_manager.doc_file, 'r', encoding='utf-8') as file:
        rag.insert(file.read())

    prompt = config_manager.get_first_default_prompt()

    # Perform naive search
    print("*** Naive Search ***")
    print(
        rag.query(prompt, param=QueryParam(mode="naive"))
    )

    # # Perform local search
    # print("*** Local Search ***")
    # print(
    #     rag.query(prompt, param=QueryParam(mode="local"))
    # )

    # # Perform global search
    # print("*** Global Search ***")
    # print(
    #     rag.query(prompt, param=QueryParam(mode="global"))
    # )

    # # Perform hybrid search
    # print("*** Hybrid Search ***")
    # print(
    #     rag.query(prompt, param=QueryParam(mode="hybrid"))
    # )