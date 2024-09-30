from .base import BaseVectorDB
import chromadb

from chromadb.config import Settings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from chromadb.utils import embedding_functions

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

import uuid


class ChromaDB(BaseVectorDB):
    def __init__(self, mount_dir) -> None:
        super().__init__()
        self.mount_dir = mount_dir
        self.build_database()
        
        # self.client = chromadb.PersistentClient(path=self.mount_dir)
        # collection_name = os.path.splitext(self.mount_dir)[0]
        # self.collection = self.client.create_collection(name=collection_name)
        # script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../{client_name}")
        # self.client = chromadb.PersistentClient(path=script_dir)
        # self.embedding_function = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", truncate_dim=384)

    # add collection
    def build_database(self):
        # for subdir, _, files in os.walk(self.mount_dir):
            
        for subdir, _, files in os.walk(self.mount_dir):
            # Use subfolder as the PersistentClient database directory
            client_settings = Settings(persist_directory=subdir)
            client = chromadb.PersistentClient(client_settings)

            for file in files:
                file_path = os.path.join(subdir, file)
                collection_name = os.path.splitext(file)[0]
                if collection_name == ".DS_Store":
                    continue
                self.add_or_update_file_in_collection(
                    client, collection_name, file_path
                )

    def add_or_update_file_in_collection(self, client, collection_name, file_path):
        """
        Adds or updates the file's content in the specified collection.
        - client: PersistentClient object
        - collection_name: Name of the collection (filename without extension)
        - file_path: Path to the file to be added or updated
        """
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        content = " ".join([doc.text for doc in documents])

        collection = client.get_or_create_collection(name=collection_name)

        existing_docs = collection.get(ids=[file_path])

        if existing_docs["ids"]:
            doc_id = existing_docs["ids"]
            collection.update(
                documents=[content],
                ids=[doc_id],
                metadatas=[
                    {"file_path": file_path, "file_name": collection_name}
                ]
            )
        else:
            doc_id = str(uuid.uuid4())
            collection.add(
                documents=[content], ids=[doc_id], 
                metadatas=[
                    {"file_path": file_path, "file_name": collection_name}
                ]
            )

    def delete_file_from_collection(self, client, collection_name, file_path):
        """
        Removes the file's document from the specified collection.
        - client: PersistentClient object
        - collection_name: Name of the collection (filename without extension)
        - file_path: Path to the file that was deleted
        """
        collection = client.get_or_create_collection(name=collection_name)

        existing_docs = collection.get(ids=[file_path])

        if existing_docs["ids"]:
            # print(f"Deleting document for file: {file_path}")
            collection.delete(ids=[file_path])
        else:
            print(f"No document found for deleted file: {file_path}")
