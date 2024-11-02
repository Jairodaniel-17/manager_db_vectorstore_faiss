from langchain_huggingface import HuggingFaceEmbeddings
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self.__initialized = True
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-code",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={
                "device": device,
            },
        )

    def get_embeddings(self):
        return self.embeddings
