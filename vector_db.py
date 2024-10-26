import os
from typing import Optional, List
import shutil
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from document_processor import DocumentProcessor


class VectorStoreManager:
    def __init__(self, path: str, name: str, embeddings: Embeddings):
        """
        Descripción: Clase para gestionar el vectorstore, incluyendo la creación, eliminación y búsqueda de documentos similares.

        Parámetros:
        - path: str - ruta del directorio que contiene los documentos (usualmente es "database" que es el directorio donde se almacenan las bases de datos).
        - name: str - nombre del vectorstore. (usualmente es el nombre de la base de datos que contiene los documentos).
        - embeddings: Embeddings - modelo de embeddings para el vectorstore.

        """
        self.path = path
        self.name = name
        self.embeddings = embeddings
        self.vectorstore = None

    def create_vectorstore(self) -> bool:
        documents = DocumentProcessor(self.path).files_to_texts()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(
            documents=texts, embedding=self.embeddings
        )
        base_de_datos_dir = os.path.join("database", self.name)
        self.vectorstore.save_local(folder_path=base_de_datos_dir)
        return True

    def delete_vectorstore(self) -> bool:
        try:
            shutil.rmtree(f"database/{self.name}")
        except FileNotFoundError:
            return False
        return True

    def search_similarity(self, query: str, fuente: Optional[str] = None) -> str:
        """
        Modo de uso:
        debe ingresar la query y la fuente (opcional) para buscar documentos similares en el vectorstore.

        Nota: debe estar definido el vectorstore para poder realizar la búsqueda.

        Parámetros:
        query: str - texto de la query.
        fuente: str - fuente de los documentos a buscar.

        Retorna:
        str - documentos similares.
        """
        if not self.vectorstore:
            self.vectorstore = self.load_vectorstore()

        if fuente:
            filtro = {"source": fuente}
            retriever = self.vectorstore.similarity_search(
                query=query, k=3, filter=filtro
            )
        else:
            retriever = self.vectorstore.similarity_search(query=query, k=20)
        return str(retriever)

    def list_sources(self) -> List[str]:
        if not self.vectorstore:
            self.vectorstore = self.load_vectorstore()

        docstore_dict = self.vectorstore.docstore._dict
        source_metadata = {}
        for doc_id, document in docstore_dict.items():
            source = document.metadata.get("source", None)
            source_metadata[doc_id] = source

        return list(set(source_metadata.values()))

    def extract_texts_by_source(self, source: str) -> List[str]:
        if not self.vectorstore:
            self.vectorstore = self.load_vectorstore()

        docstore_dict = self.vectorstore.docstore._dict
        texts = []
        for document in docstore_dict.values():
            source_doc = document.metadata.get("source", None)
            if source_doc == source:
                texts.append(document.page_content)
        return texts

    def save_text_to_file_temp(self, source: str) -> bool:
        try:
            texts = self.extract_texts_by_source(source)
            carpeta = "temp"
            if os.path.exists(carpeta):
                shutil.rmtree(carpeta)
            os.makedirs(carpeta)
            target_source_safe = source.replace("\\", "_").replace("/", "_")
            file_path = os.path.join(carpeta, target_source_safe + ".txt")

            with open(file_path, "w", encoding="utf-8") as file:
                for text in texts:
                    file.write(text)
                    file.write("\n")
            return True
        except Exception:
            return False

    def load_vectorstore(self) -> FAISS:
        return FAISS.load_local(
            folder_path=os.path.join("database", self.name),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def add_files_vectorstore(self) -> Optional[FAISS]:
        temp_folder = "docs"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
            return None

        documents = DocumentProcessor(temp_folder).files_to_texts()
        if not documents:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_documents(documents)
        self.vectorstore = self.load_vectorstore()
        self.vectorstore.add_documents(documents=texts)
        self.vectorstore.save_local(folder_path=os.path.join("database", self.name))
        return self.vectorstore
