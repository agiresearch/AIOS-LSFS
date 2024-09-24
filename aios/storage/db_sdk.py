from .base import BaseStorage
from pathlib import PurePosixPath
import json
import os
from llama_index.core import PromptTemplate
import chromadb
from chromadb.api.types import Metadata
import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Document
from llama_index.core.retrievers import VectorIndexRetriever
import uuid
import redis
import logging

# from db_storage import DBStorage
from .db_storage import DBStorage
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, KEYWORD
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
from whoosh.query import Phrase
import shutil


class Data_Op(DBStorage):
    def __init__(self, retri_dic, redis_client):
        # pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
        # self.redis_client = redis.Redis(connection_pool=pool)
        self.redis_client = redis_client
        # self.redis_client.select(0)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.retri_dic = retri_dic
        super().__init__(self.redis_client, self.embed_model, self.retri_dic)

    def create(self, db_path, db_name, doc, metaname=None):
        # create chroma database by single doc or file
        if not os.path.exists(doc):
            super().create_or_get_file(db_path, db_name, doc)
        else:
            if os.path.isdir(doc):
                for root, dirs, filenames in os.walk(doc):
                    # print(filenames)
                    for filename in filenames:
                        if filename == ".DS_Store":
                            continue
                        name, _ = os.path.splitext(filename)
                        docu = os.path.join(root, filename)
                        # print(name, docu)
                        super().create_or_get_file(
                            db_path=db_path, 
                            db_name=db_name, 
                            metaname=name, 
                            doc=docu
                        )
            else:
                name = os.path.basename(doc)
                name = os.path.splitext(name)[0]
                if metaname is None:
                    super().create_or_get_file(db_path, db_name, name, doc)
                else:
                    super().create_or_get_file(db_path, db_name, metaname, doc)

        # return super().create_or_get_collection(db_path,db_name)

    def insert(self, db_path, db_name, doc, metaname):

        return super().add_in_db(db_path, db_name, doc, metaname)

    def retrieve(self, db_path, db_name, query, type="meaning", loc=None):

        if type == "full_text":
            return super().full_text_retrieve(db_path, db_name, query, loc=loc)
        elif type == "meaning":
            return super().sym_retrieve(db_path, db_name, query, loc=loc)
        else:
            raise ValueError("retrieve type error")

    def update(self, db_path, db_name, doc, obj=None):

        return super().change_db(db_path, db_name, doc, obj)

    def delete(self, db_path, db_name, metaname):

        return super().del_(db_path, db_name, metaname)

    def group_keywords(self, db_path, query, new_name, db_name=None, con=None):
        ans, ans_name, metadatas = self.keyword_retrieve(
            db_path, query, db_name=db_name, con=con, group=True
        )
        for i in range(len(ans_name)):
            self.create_or_get_file(
                db_path=db_path,
                db_name=new_name,
                metaname=ans_name[i],
                doc=ans[i],
                metadata=metadatas[i],
            )

        return db_path, new_name

    def group_semantic(self, db_path, query, new_name, db_name=None, top_k=None):
        # group by with key word and build a new dababase
        ans, ans_name, metadatas = self.semantic_retrieve(
            db_path, query, top_k=top_k, db_name=db_name
        )
        for i in range(len(ans_name)):
            self.create_or_get_file(
                db_path=db_path,
                db_name=new_name,
                metaname=ans_name[i],
                doc=ans[i],
                metadata=metadatas[i],
            )
        return db_path, new_name

    def integrated_retrieve(
        self, db_path, kquery, squery, top_k, new_name, db_name=None, con=None
    ):
        db_path, new_name = self.group_keywords(
            db_path, kquery, new_name, db_name=db_name, con=con
        )
        ans, ans_name = self.semantic_retrieve(
            db_path, squery, top_k=top_k, db_name=db_name
        )

        return ans, ans_name

    def join(self, db_path, db_name1, metaname1, metaname2, db_name2=None, new=True):
        # add 2 to 1
        if new:
            if db_name2 is None:
                obj_path = os.path.join(db_path, db_name1)
                chroma_client = chromadb.PersistentClient(path=obj_path)
                chroma_collection1 = chroma_client.get_collection(metaname1)
                chroma_collection2 = chroma_client.get_collection(metaname2)

                doc1 = chroma_collection1.get()["documents"]
                doc1 = "".join(doc1)
                doc2 = chroma_collection2.get()["documents"]
                doc2 = "".join(doc2)
                name1 = chroma_collection1.name
                name2 = chroma_collection2.name
                metadata1 = chroma_collection1.get()["metadatas"][0]
                metadata2 = chroma_collection2.get()["metadatas"][0]
                name = name1 + "_" + name2
                doc = doc1 + doc2
                merged_metadata = {}
                for key in set(metadata1) | set(metadata2):
                    values = []
                    if key in metadata1:
                        values.append(metadata1[key])
                    if key in metadata2:
                        values.append(metadata2[key])
                    merged_metadata[key] = values if len(values) > 1 else values[0]
                self.create_or_get_file(
                    db_path, db_name1, db_name=name, doc=doc, metadata=merged_metadata
                )
            else:
                obj_path1 = os.path.join(db_path, db_name1)
                obj_path2 = os.path.join(db_path, db_name2)
                chroma_client1 = chromadb.PersistentClient(path=obj_path1)
                chroma_client2 = chromadb.PersistentClient(path=obj_path2)
                chroma_collection1 = chroma_client1.get_collection(metaname1)
                chroma_collection2 = chroma_client2.get_collection(metaname2)

                doc1 = chroma_collection1.get()["documents"]
                doc1 = "".join(doc1)
                doc2 = chroma_collection2.get()["documents"]
                doc2 = "".join(doc2)
                name1 = chroma_collection1.name
                name2 = chroma_collection2.name
                metadata1 = chroma_collection1.get()["metadatas"][0]
                metadata2 = chroma_collection2.get()["metadatas"][0]
                name = name1 + "_" + name2
                doc = doc1 + doc2
                merged_metadata = {}
                for key in set(metadata1) | set(metadata2):
                    values = []
                    if key in metadata1:
                        values.append(metadata1[key])
                    if key in metadata2:
                        values.append(metadata2[key])
                    merged_metadata[key] = values if len(values) > 1 else values[0]
                self.create_or_get_file(
                    db_path, db_name1, db_name=name, doc=doc, metadata=merged_metadata
                )
        else:
            if db_name2 is None:
                obj_path = os.path.join(db_path, db_name1)
                chroma_client = chromadb.PersistentClient(path=obj_path)
                chroma_collection2 = chroma_client.get_collection(metaname2)

                doc = chroma_collection2.get()["documents"]

                super().add_in_db(db_path, db_name1, doc, metaname1)
                super().del_(db_path, db_name1, metaname2)
            else:
                obj_path1 = os.path.join(db_path, db_name1)
                obj_path2 = os.path.join(db_path, db_name2)
                chroma_client1 = chromadb.PersistentClient(path=obj_path1)
                chroma_client2 = chromadb.PersistentClient(path=obj_path2)
                chroma_collection2 = chroma_client2.get_collection(metaname2)
                doc = chroma_collection2.get()["documents"]
                super().add_in_db(db_path, db_name1, doc, metaname1)
                super().del_(db_path, db_name2, metaname2)

    def group_retrieve(self, db_path, query, db_name=None):
        # retrieve query in content related to keywords, like where in sql

        doc, name = self.from_some_key_full(db_path, query, db_name=db_name)
        documents = [
            Document(id=str(uuid.uuid4()), text=doc_content) for doc_content in doc
        ]
        index = VectorStoreIndex.from_documents(documents)
        retriever = VectorIndexRetriever(index, similarity_top_k=2)
        result = retriever.retrieve(query)
        ans = result[0].get_content()
        ans = "".join(ans)
        return ans, name

    def lock_file(self, db_path, db_name, metaname):
        collection = self.create_or_get_file(db_path, db_name, metaname)
        ids = collection.id
        metadata = collection.get()["metadatas"][0]
        new_data = {"state": 1}
        update_data = {**metadata, **new_data}
        collection.update(ids=[ids], metadatas=[update_data])

    def unlock_file(self, db_path, db_name, metaname):
        collection = self.create_or_get_file(db_path, db_name, metaname)
        ids = collection.id
        metadata = collection.get()["metadatas"][0]
        update_data = metadata.copy()
        if update_data["state"]:
            update_data["state"] = 0
        collection.update(ids=[ids], metadatas=[update_data])

    def get_collection(self, db_path, db_name, metaname=None):

        collection = self.create_or_get_file(db_path, db_name, metaname=metaname)
        # print(collection.get()['documents'])
        return collection

        # res = collection.get(ids=['188a8336-eab0-455b-b78e-28e011e34b19'])
        # print(res)
        # print(collection)
