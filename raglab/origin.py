#! /usr/bin/env python3

from raglab import ConfigManager, GraphVis, RAGManager

if __name__ == "__main__":
    config_manager = ConfigManager("config/origin.yaml")

    rag_manager = RAGManager(config_manager=config_manager)
    graph_vis = GraphVis(config_manager=config_manager)
    
    rag_manager.doc2kg()
    rag_manager.first_item_query()

    graph_vis.convert()
