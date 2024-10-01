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
        # self.build_database()

        mounted_fs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mounted_fs"
        )

        self.client = chromadb.PersistentClient(mounted_fs_dir)
        # print(self.mount_dir)
        collection_name = os.path.basename(self.mount_dir)

        # print(collection_name)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # if not os.path.exists(os.path.join(mounted_fs_dir, "chroma.sqlite3")):
        # self.build_database()

    # add collection
    def build_database(self):
        # for subdir, _, files in os.walk(self.mount_dir):
        for subdir, _, files in os.walk(self.mount_dir):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_name = os.path.splitext(file)[0]
                # print(file_path, file_name)
                if file_name == ".DS_Store":
                    continue
                self.add_or_update_file_in_collection(file_path, file_name)

    def add_or_update_file_in_collection(self, file_path, file_name):
        """
        Adds or updates the file's content in the specified collection.
        - client: PersistentClient object
        - collection_name: Name of the collection (filename without extension)
        - file_path: Path to the file to be added or updated
        """
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        content = " ".join([doc.text for doc in documents])

        # print(content[:500])

        # collection = client.get_or_create_collection(name=collection_name)

        existing_docs = self.collection.get(ids=[file_name])

        if existing_docs["ids"]:
            doc_id = existing_docs["ids"]
            self.collection.update(
                documents=[content],
                ids=doc_id,
                metadatas=[{"file_path": file_path, "file_name": file_name}],
            )
        else:
            # doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[content],
                ids=[file_name],
                metadatas=[{"file_path": file_path, "file_name": file_name}],
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

    def retrieve(self, name, k, keywords):
        results = self.collection.query(query_texts=[keywords], n_results=int(k))
        return results
        # print([doc[:500] for doc in results["documents"][0]])
        # print(results["metadatas"])
