from raglab import RAGManager

if __name__ == "__main__":
    rag_manager = RAGManager("config/origin.yaml")
    rag_manager.doc2kg()
    rag_manager.first_item_query()
