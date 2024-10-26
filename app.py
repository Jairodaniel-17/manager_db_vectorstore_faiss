from fastapi import FastAPI, Form, HTTPException, Depends, File, UploadFile, Body
from typing import Optional, List
from pydantic import BaseModel
import urllib
from embeddings import EmbeddingManager
from vector_db import VectorStoreManager
import os
import shutil

app = FastAPI()

# Crear una sola instancia de EmbeddingManager
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.get_embeddings()
path_docs = "docs"  # Directorio temporal para almacenar los archivos subidos
path_db = "database"  # Directorio para almacenar el vectorstore


# Autenticación básica para los endpoints
def basic_auth(username: str = Form(...), password: str = Form(...)):
    if username != "admin" or password != "admin":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return f"{username}:{password}"


# Modelo Pydantic para el cuerpo de la solicitud
class CreateVectorStoreRequest(BaseModel):
    name: str


# Descripción: Crea el vectorstore a partir de los documentos subidos.
@app.post("/vectorstore/create", tags=["VectorStore"])
async def create_vectorstore(
    create_request: CreateVectorStoreRequest = Depends(),  # Usar el modelo como dependencia
    files: List[UploadFile] = File(...),
):
    """Create a vectorstore from the uploaded documents."""
    try:
        # Crear un directorio temporal para almacenar los archivos subidos
        if os.path.exists(path_docs):
            shutil.rmtree(path_docs)  # Limpiar el directorio si ya existe
        os.makedirs(path_docs)

        for file in files:
            file_path = os.path.join(path_docs, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

        # Crear el vectorstore
        manager = VectorStoreManager(
            path=path_docs, name=create_request.name, embeddings=embeddings
        )
        if manager.create_vectorstore():
            # Limpiar el directorio temporal después de crear el vectorstore
            shutil.rmtree(path_docs)
            return {"message": "Vectorstore created successfully."}

        # Limpiar el directorio temporal en caso de fallo
        shutil.rmtree(path_docs)
        return {"message": "Failed to create vectorstore."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchSimilarityRequest(BaseModel):
    name_database: str
    query: str
    fuente: Optional[str] = None


#
@app.get("/vectorstore/search", tags=["Similarity Search"])
async def search_similarity(search_request: SearchSimilarityRequest = Depends()):
    """Search for similar documents in the vectorstore."""
    try:
        manager = VectorStoreManager(
            path=path_db,
            name=search_request.name_database,
            embeddings=embeddings,
        )

        # convertir %20 a espacios y demás caracteres especiales con urllib.parse.unquote
        search_request.query = str(urllib.parse.unquote(search_request.query))
        result = manager.search_similarity(
            query=search_request.query, fuente=search_request.fuente
        )
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ListSourcesRequest(BaseModel):
    nombre_db_vectorial: str


# Descripción: Lista todas las fuentes disponibles en el vectorstore.
@app.get("/vectorstore/sources", tags=["Sources"])
async def list_sources(list_request: ListSourcesRequest = Depends()):
    try:
        manager = VectorStoreManager(
            path=path_db, name=list_request.nombre_db_vectorial, embeddings=embeddings
        )
        sources = manager.list_sources()
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# class TextsBySourceRequest(BaseModel):
#     nombre_db_vectorial: str
#     fuente: str


# @app.get("/vectorstore/texts_by_source", tags=["Texts by Source"])
# async def extract_texts_by_source(texts_request: TextsBySourceRequest = Depends()):
#     """Extrae textos de una fuente específica del vectorstore.
#     Se requiere el nombre de la base de datos vectorial y la fuente de los documentos.
#     """
#     try:
#         manager = VectorStoreManager(
#             path=path_db, name=texts_request.nombre_db_vectorial, embeddings=embeddings
#         )
#         texts = manager.extract_texts_by_source(source=texts_request.fuente)
#         return {"texts": texts}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


class SaveTempRequest(BaseModel):
    nombre_db_vectorial: str
    fuente: str


@app.post("/vectorstore/save_temp", tags=["Save Temp"])
async def save_text_to_file_temp(save_temp: SaveTempRequest = Depends()):
    # Descripción: Guarda en un archivo temporal el texto de una fuente específica.
    try:
        manager = VectorStoreManager(
            path=path_db, name=save_temp.nombre_db_vectorial, embeddings=embeddings
        )
        saved = manager.save_text_to_file_temp(source=save_temp.fuente)
        if saved:
            return {"message": "Text saved to file successfully."}
        else:
            return {"message": "No text found to save."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AddFilesRequest(BaseModel):
    nombre_db_vectorial: str


@app.post("/vectorstore/add_files", tags=["Add Files"])
async def add_files_vectorstore(
    add_files_request: AddFilesRequest = Depends(), files: List[UploadFile] = File(...)
):
    try:
        # Crear un directorio temporal para almacenar los archivos subidos
        if os.path.exists(path_docs):
            shutil.rmtree(path_docs)  # Limpiar el directorio si ya existe
        os.makedirs(path_docs)

        for file in files:
            file_path = os.path.join(path_docs, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

        # Agregar documentos al vectorstore
        manager = VectorStoreManager(
            path=path_docs,
            name=add_files_request.nombre_db_vectorial,
            embeddings=embeddings,
        )
        if manager.add_files_vectorstore():
            # Limpiar el directorio temporal después de agregar los documentos
            shutil.rmtree(path_docs)
            return {"message": "Files added to vectorstore successfully."}

        # Limpiar el directorio temporal en caso de fallo
        shutil.rmtree(path_docs)
        return {"message": "Failed to add files to vectorstore."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Descripción: Elimina el vectorstore completo y sus datos.
class DeleteVectorStoreRequest(BaseModel):
    nombre_db_vectorial: str


@app.delete("/vectorstore/delete", tags=["Delete VectorStore"])
async def delete_vectorstore(delete_request: DeleteVectorStoreRequest = Depends()):
    """Delete the vectorstore and its data."""
    try:
        manager = VectorStoreManager(
            path=path_db, name=delete_request.nombre_db_vectorial, embeddings=embeddings
        )
        if manager.delete_vectorstore():
            return {"message": "Vectorstore deleted successfully."}
        return {"message": "Failed to delete vectorstore."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
