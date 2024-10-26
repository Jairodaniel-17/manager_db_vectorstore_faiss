import os
from typing import List
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
)


class DocumentProcessor:
    def __init__(self, path: str):
        self.path = path

    def files_to_texts(self) -> list:
        loaders = []

        if any(fname.endswith(".pdf") for fname in os.listdir(self.path)):
            pdf_loader = DirectoryLoader(
                path=self.path, glob="*.pdf", loader_cls=PyMuPDFLoader
            )
            loaders.append(pdf_loader)
        if any(fname.endswith(".txt") for fname in os.listdir(self.path)):
            text_loader = DirectoryLoader(
                path=self.path,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            loaders.append(text_loader)
        if any(fname.endswith(".docx") for fname in os.listdir(self.path)):
            docx_loader = DirectoryLoader(
                path=self.path, glob="*.docx", loader_cls=Docx2txtLoader
            )
            loaders.append(docx_loader)
        if any(fname.endswith(".doc") for fname in os.listdir(self.path)):
            doc_loader = DirectoryLoader(
                path=self.path, glob="*.doc", loader_cls=Docx2txtLoader
            )
            loaders.append(doc_loader)

        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        return documents
