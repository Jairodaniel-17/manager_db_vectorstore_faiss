from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-code",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={
                "device": "cuda",
            },
        )

    def get_embeddings(self):
        return self.embeddings
