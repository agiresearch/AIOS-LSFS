# TODO: Not implemented
# Storing to databases has not been implemented yet
from .base import BaseStorage
from pathlib import PurePosixPath
import json
import shutil
import os
from llama_index.core import PromptTemplate
import chromadb
from chromadb.api.types import Metadata
import numpy as np
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Document
from llama_index.core.retrievers import VectorIndexRetriever
import redis
import uuid
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser


class DBStorage:
    # def __init__(self, agent_name, ragdata_path,db_path):
    def __init__(self, redis_client, retri_dic):
        # self.agent_name = agent_name
        # self.client = chromadb.Client()
        # # self.collection = self._create_or_get_collection()
        # self.data_path = ragdata_path
        # self.db_path = db_path
        # self.llm = llm
        # self.embed_model = embed_model
        self.redis_client = redis_client
        self.retri_dic = retri_dic

    def create_or_get_file(
        self, db_path, db_name, metaname=None, doc=None, metadata=None
    ):
        path = os.path.join(db_path, db_name)
        if metaname is None:
            chroma_client = chromadb.PersistentClient(path=path)
            return chroma_client

        if doc is None:
            chroma_client = chromadb.PersistentClient(path=path)
            chroma_collection = chroma_client.get_collection(metaname)
            # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # index = VectorStoreIndex(storage_context=storage_context, embed_model=self.embed_model)
            return chroma_collection

        else:
            chroma_client = chromadb.PersistentClient(path=path)
            chroma_collection = chroma_client.get_or_create_collection(metaname)
            # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embed_model)
            if os.path.exists(doc):
                id = []
                documents = SimpleDirectoryReader(input_files=[doc]).load_data()
                # for doc in documents:
                #     print(doc.text)
                merged_content = " ".join(doc.text for doc in documents)
                document = merged_content
            else:
                doc_id = str(uuid.uuid4())
                document = [doc]
                document = Document(id=doc_id, text=document)
                
            embedding = self.embed_model._embed(document)
            id = str(uuid.uuid4())
            chroma_collection.add(
                ids=id, embeddings=embedding, documents=document, metadatas=metadata
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            # storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )
            # index = VectorStoreIndex.from_documents(documents,storage_context=storage_context, embed_model=self.embed_model)
            path = os.path.join(path, metaname)
            index.storage_context.persist(persist_dir=path)

        # return chroma_collection

    def add_in_db(self, db_path, db_name, doc, metaname):
        add_path = os.path.join(db_path, db_name, metaname)
        if not os.path.exists(add_path):
            index = self.create_or_get_file(db_path, db_name, metaname, doc)
        else:
            chroma_client = chromadb.PersistentClient(path=add_path)
            chroma_collection = chroma_client.get_or_create_collection(metaname)
            if (
                "state" in chroma_collection.metadata[0]
                and chroma_collection.metadata[0]["state"] == 1
            ):
                raise Exception(
                    "The file is lock, you can not change it until you unlock it"
                )
            id = str(uuid.uuid4())
            if os.path.exists(doc):
                documents = SimpleDirectoryReader(input_files=[doc]).load_data()
                merged_content = " ".join(doc.text for doc in documents)
                document = merged_content
            else:
                document = [doc]

            embedding = self.embed_model._embed(document)
            chroma_collection.add(ids=[id], embeddings=embedding)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )
        index.storage_context.persist(persist_dir=add_path)

    def del_(self, db_path, db_name=None, metaname=None, text=None):
        if metaname is None and text is None:
            raise ValueError("Must have one of metaname and text as arguments")
        if db_name is not None:
            del_path = os.path.join(db_path, db_name)
            if not os.path.exists(del_path):
                raise FileNotFoundError("delete path is not exist")
            else:
                if metaname:
                    del_path = os.path.join(del_path, metaname)
                    chroma_client = chromadb.PersistentClient(path=del_path)
                    chroma_collection = chroma_client.get_collection(metaname)
                    if (
                        "state" in chroma_collection.metadata[0]
                        and chroma_collection.metadata[0]["state"] == 1
                    ):
                        raise Exception(
                            "The file is lock, you can not change it until you unlock it"
                        )
                    chroma_collection = chroma_collection.delete()
                else:
                    chroma_client = chromadb.PersistentClient(path=del_path)
                    collections = chroma_client.list_collections()
                    for collection in collections:
                        if (
                            "state" in collection.metadata[0]
                            and collection.metadata[0]["state"] == 1
                        ):
                            print(
                                "{} is lock, you can not change it until you unlock it".format(
                                    collection.name
                                )
                            )
                            continue
                        req = {"$contains": text}
                        collection = collection.get(where_document=req)
                        ids = collection["ids"]
                        if ids is None:
                            continue
                        else:
                            path = collection.get()["metadatas"][0]["file_path"]
                            jud = input(
                                f"\n {path} will be deleted, you will confirm it. Please input yes or no:"
                            )
                            if jud.lower() == "yes":
                                chroma_collection = collection.delete(ids=ids)
                            else:
                                continue
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                index = VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=self.embed_model
                )
                index.storage_context.persist(persist_dir=del_path)
                if chroma_client.count_collections() == 0:
                    os.rmdir(del_path)
        else:
            del_path = db_path
            if not os.path.exists(del_path):
                raise FileNotFoundError("delete path is not exist")
            else:
                path = []
                for root, dirs, files in os.walk(del_path):
                    for dirname in dirs:
                        tar_path = os.join(del_path, dirname)
                        if metaname:
                            chroma_client = chromadb.PersistentClient(path=tar_path)
                            collections = chroma_client.list_collections()
                            for collection in collections:
                                if (
                                    "state" in chroma_collection.metadata[0]
                                    and chroma_collection.metadata[0]["state"] == 1
                                ):
                                    raise Exception(
                                        "The file is lock, you can not change it until you unlock it"
                                    )
                                if metaname == collection.name:
                                    path.append(
                                        collection.get()["metadatas"][0]["file_path"]
                                    )
                        else:
                            chroma_client = chromadb.PersistentClient(path=tar_path)
                            collections = chroma_client.list_collections()
                            for collection in collections:
                                if (
                                    "state" in collection.metadata[0]
                                    and collection.metadata[0]["state"] == 1
                                ):
                                    print(
                                        "{} is lock, you can not change it until you unlock it".format(
                                            collection.name
                                        )
                                    )
                                    continue
                                req = {"$contains": text}
                                collection = collection.get(where_document=req)
                                ids = collection["ids"]
                                if ids is None:
                                    continue
                                else:
                                    path = collection.get()["metadatas"][0]["file_path"]
                                    jud = input(
                                        f"\n {path} will be deleted, you will confirm it. Please input yes or no:"
                                    )
                                    if jud.lower() == "yes":
                                        chroma_collection = collection.delete(ids=ids)
                                    else:
                                        continue
                            if chroma_client.count_collections() == 0:
                                os.rmdir(tar_path)
                if metaname:
                    if len(path) == 0:
                        raise Exception(
                            "There is no such file, please check your file name"
                        )
                    elif len(path) == 1:
                        tar_path = path[0]
                    else:
                        del_path = input(
                            f"\n All eligible paths are as follows{path}, please choose which one you want to delete:"
                        )
                        tar_path, _ = os.path.split(tar_path)
                    chroma_client = chromadb.PersistentClient(path=tar_path)
                    chroma_collection = chroma_client.get_collection(metaname)
                    chroma_collection = chroma_collection.delete()
                    if chroma_client.count_collections() == 0:
                        os.rmdir(tar_path)

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )
            index.storage_context.persist(persist_dir=del_path)

    def change_db(self, db_path, db_name, doc, metaname):
        change_path = os.path.join(db_path, db_name)
        if not os.path.exists(change_path):
            raise FileNotFoundError("change path is not exist")
        else:
            chroma_client = chromadb.PersistentClient(path=change_path)
            chroma_collection = chroma_client.get_collection(metaname)
            if (
                "state" in chroma_collection.metadata[0]
                and chroma_collection.metadata[0]["state"] == 1
            ):
                raise Exception(
                    "The file is lock, you can not change it until you unlock it"
                )
            ids = chroma_collection.id
            if os.path.exists(doc):
                # documents = SimpleDirectoryReader(doc).load_data()
                documents = SimpleDirectoryReader(input_files=[doc]).load_data()
                merged_content = " ".join(doc.text for doc in documents)
                document = merged_content
            else:
                # document = Document(id=id, content=doc)
                document = [doc]
            embedding = self.embed_model._embed(document)
            chroma_collection.update(ids=[str(ids)], embeddings=embedding)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=self.embed_model
            )
            index.storage_context.persist(persist_dir=change_path)

        return chroma_collection

    def keyword_retrieve(self, db_path, query, db_name=None, con=None, group=False):
        logging.basicConfig(
            filename="results.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        # return the content with same keywords
        ans = []
        rk = {}
        name_ans = []
        metadatas = []
        if db_name is not None:
            search_path = os.path.join(db_path, db_name)
            chroma_client = chromadb.PersistentClient(path=search_path)
            # collection = chroma_client.get_collection('promptbench')
            # print(collection.get()['documents'])
            # print(ssd)
            collections = chroma_client.list_collections()
            for collection in collections:
                # if collection.name != 'os_copilot':
                #     continue
                # chroma_collection = chroma_client.get_collection(name)
                doc = collection.get()["documents"]
                metadata = collection.get()["metadatas"][0]
                docu = " ".join(doc)
                text = docu.replace("\n", " ")
                if "References" in text:
                    parts = text.split("References")
                    text = parts[0]
                # print(text)
                # print(ssd)
                schema = Schema(content=TEXT(stored=True))
                index_dir = collection.name
                if not os.path.exists(index_dir):
                    os.mkdir(index_dir)
                index = create_in(index_dir, schema)
                writer = index.writer()
                writer.add_document(content=text)
                writer.commit()
                with index.searcher() as searcher:
                    # reader = searcher.reader()
                    # # 获取索引中的所有文档 ID
                    # doc_ids = list(reader.all_doc_ids())
                    # for doc_id in doc_ids:
                    #     # 获取文档的内容
                    #     doc = reader.stored_fields(doc_id)
                    #     print(f"Document {doc_id}: {doc}")
                    # print(ssd)
                    if isinstance(query, list):
                        for i, condition in enumerate(query):
                            tmp_que = QueryParser("content", index.schema).parse(
                                f"*condition*"
                            )
                            if i == 0:
                                que = tmp_que
                            else:
                                if con == "and":
                                    que = que & tmp_que
                                elif con == "or":
                                    que = que | tmp_que
                    else:
                        # query = f"*{query}*"
                        que = QueryParser("content", index.schema).parse(f"*{query}*")
                    results = searcher.search(que)
                    # print(results)
                    # print(ssd)
                    # for result in results:
                    #     print(11111)
                    #     print(result)
                    # print(ssd)
                    if len(results) > 0:
                        ans.append(doc)
                        name_ans.append(collection.name)
                        metadatas.append(metadata)

                shutil.rmtree(index_dir)
        else:
            for root, dirs, files in os.walk(db_path):
                for dir in dirs:
                    search_path = os.path.join(db_path, dir)
                    print(search_path)
                    chroma_client = chromadb.PersistentClient(path=search_path)
                    collections = chroma_client.list_collections()
                    for collection in collections:
                        # chroma_collection = chroma_client.get_collection(name)
                        doc = collection.get()["documents"]
                        metadata = collection.get()["metadatas"][0]
                        doc = " ".join(doc)
                        key = "References"
                        schema = Schema(content=TEXT(stored=True))
                        index_dir = collection.name
                        if not os.path.exists(index_dir):
                            os.mkdir(index_dir)
                        index = create_in(index_dir, schema)

                        writer = index.writer()
                        writer.add_document(content=doc)
                        writer.commit()
                        with index.searcher() as searcher:
                            if isinstance(query, list):
                                que = None
                                for condition in query:
                                    tmp_que = QueryParser(
                                        "content", index.schema
                                    ).parse(condition)
                                    if con == "and":
                                        que = que & tmp_que
                                    elif con == "or":
                                        que = que | tmp_que
                            else:
                                que = QueryParser("content", index.schema).parse(query)
                            results = searcher.search(que)
                            if len(results) > 0:
                                ans.append(doc)
                                name_ans.append(collection.name)
                                metadatas.append(metadata)
                        shutil.rmtree(index_dir)
        if group:
            return ans, name_ans, metadatas
        else:
            return ans, name_ans

    def semantic_retrieve(self, db_path, query, top_k, db_name=None, group=False):
        logging.basicConfig(
            filename="results.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        # return the content with same keywords
        ans = []
        rk = {}
        name_ans = []
        
        if db_name is not None:
            search_path = os.path.join(db_path, db_name)
            chroma_client = chromadb.PersistentClient(path=search_path)
            collections = chroma_client.list_collections()
            
            for collection in collections:
                # print(name)
                # chroma_collection = chroma_client.get_collection(name)
                vector_store = ChromaVectorStore(chroma_collection=collection)
                index = VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=self.embed_model
                )
                retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
                result = retriever.retrieve(query)
                rk[collection.name] = result[0].score
            # print(rk)

        else:
            for root, dirs, files in os.walk(db_path):
                # print(dirs)
                for dir in dirs:
                    search_path = os.path.join(db_path, dir)
                    # print(search_path)
                    chroma_client = chromadb.PersistentClient(path=search_path)
                    # print(chroma_client)
                    collections = chroma_client.list_collections()
                    print(collections)
                    
                    for collection in collections:
                        print(collection)
                        vector_store = ChromaVectorStore(chroma_collection=collection)
                        index = VectorStoreIndex.from_vector_store(
                            vector_store, embed_model=self.embed_model
                        )
                        retriever = VectorIndexRetriever(
                            index=index, similarity_top_k=2
                        )
                        result = retriever.retrieve(query)
                        rk[collection.name] = result[0].score
                    # print(rk)

        sorted_dict_desc = dict(
            sorted(rk.items(), key=lambda item: item[1], reverse=True)
        )
        keys_iterator = iter(sorted_dict_desc)
        metadatas = []
        for i in range(top_k):
            key = next(keys_iterator)
            collection = chroma_client.get_collection(key)
            doc = collection.get()["documents"]
            doc = "".join(doc)
            ans.append(doc)
            name_ans.append(collection.name)
            metadatas.append(collection.get()["metadatas"][0])

        if group:
            return ans, name_ans, metadatas
        else:
            return ans, name_ans

    def check(self):
        inter_path = os.path.join(self.db_path, self.agent_name, "inter_data")
        chroma_client = chromadb.PersistentClient(path=inter_path)
        collection = chroma_client.get_collection("inter_data")
        do = collection.get()
