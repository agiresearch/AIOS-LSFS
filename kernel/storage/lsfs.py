# from aios.storage.db.redis import Redis

from aios.storage.vectordb.chromadb import ChromaDB

from watchdog.observers import Observer

from watchdog.events import FileSystemEventHandler

import os

from pyopenagi.utils.chat_template import Query

from pyopenagi.utils.chat_template import Response


class LSFSSupervisor(FileSystemEventHandler):
    def __init__(self, mount_dir) -> None:
        self.mount_dir = mount_dir
        # self.db = Redis()
        # self.vector_db = ChromaDB(mount_dir=mount_dir)
        self.is_start = False

    def terminate(self):
        self.is_start = False

    def on_modified(self, event):
        if not event.is_directory:
            self.update_file_in_database(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            # print(f"File created: {event.src_path}")
            self.update_file_in_database(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            # print(f"File deleted: {event.src_path}")
            self.delete_file_in_database(event.src_path)

    def update_file_in_database(self, file_path):
        """
        Update the file in ChromaDB (add or modify).
        """
        # subdir = os.path.dirname(file_path)
        collection_name = os.path.splitext(os.path.basename(file_path))[0]

        # Initialize PersistentClient for this subfolder
        self.vector_db.add_or_update_file_in_collection(
            collection_name=collection_name, file_path=file_path
        )

    def delete_file_in_database(self, file_path):
        """
        Delete the file's document from ChromaDB.
        """
        # subdir = os.path.dirname(file_path)
        collection_name = os.path.splitext(os.path.basename(file_path))[0]

        # Delete the file content from the collection
        self.vector_db.delete_file_from_collection(
            collection_name=collection_name, file_path=file_path
        )


class LSFSParser:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.api_call_format = [
            {
                "type": "function",
                "function": {
                    "name": "create_file",
                    "description": "create a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the file",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "create a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the directory",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "change_summary",
                    "description": "change file or change directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src": {
                                "type": "string",
                                "description": "source name of the file or directory",
                            }
                        },
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "target name of the file or directory",
                            }
                        },
                        "required": ["src", "target"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_summary",
                    "description": "retrieve files and summary the content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the file",
                            },
                            "k": {
                                "type": "string",
                                "default": "3",
                                "description": "top k files to be retrieved",
                            },
                            "keywords": {
                                "type": "string",
                                "description": "keywords used to describe how to locate the files",
                            },
                        },
                        "required": ["k", "keywords"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rollback",
                    "description": "rollback a file to a specific version",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the file",
                            },
                            "n": {
                                "type": "string",
                                "default": "1",
                                "description": "the number of versions to rollback",
                            },
                            "time": {
                                "type": "string",
                                "description": "the specific time of a file version",
                            },
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "link",
                    "description": "generate a link for a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the file",
                            }
                        },
                        "required": [],
                    },
                },
            },
        ]
        self.system_instruction = " ".join(
            [
                "You are a good parser to parse natural prompts into executable API calls. ",
                "Given a query, identify the functions to call. ",
            ]
        )

    def parse(self, agent_request):

        task_input = agent_request.query.messages[1]["content"]
        step_instruction = agent_request.query.messages[-1]["content"]

        messages = [
            {"role": "system", "content": self.system_instruction},
            {
                "role": "user",
                "content": "The task is: " + task_input + step_instruction,
            },
        ]
        query = Query(
            messages=messages, tools=self.api_call_format, action_type="message_llm"
        )
        agent_request.query = query

        response = self.llm.address_request(agent_request)

        api_calls = response.tool_calls

        return api_calls


class LSFS:
    def __init__(self, mount_dir) -> None:
        self.mount_dir = mount_dir
        self.vector_db = ChromaDB(mount_dir=mount_dir)
        # self.event_handler = LSFSSupervisor(mount_dir=mount_dir)
        # self.observer = Observer()
        self.is_start = False
        self.api_map = {
            "create_file": self.create_file,
            "create_directory": self.create_directory,
            "change_summary": self.change_summary,
            "retrieve_summary": self.retrieve_summary,
            "rollback": self.rollback,
            "link": self.link,
        }
        # self.lsfs_parser = LSFSParser()

    def start(self):
        self.is_start = True
        # self.observer.schedule(self.event_handler, self.mount_dir, recursive=True)
        # self.observer.start()
        # while self.is_start:
        #     pass
        # self.observer.join()

    def execute_calls(self, api_calls):
        api_call = api_calls[0]
        # for api_call in api_calls:
        name, params = api_call["name"], api_call["parameters"]
        return self.api_map[name](params)

    def terminate(self):
        self.is_start = False
        # self.observer.stop()

    def create_file(self, params):
        pass

    def create_directory(self, params):
        pass

    def retrieve_summary(self, params):
        # print(params)
        name = params["name"] if "name" in params else None
        k = params["k"] if "k" in params else None
        keywords = params["keywords"] if "keywords" in params else None
        if k and keywords:
            results = self.vector_db.retrieve(name, k, keywords)
            chosen_results = self.choose_result(results)
            return chosen_results

    def choose_result(self, results):
        result_list = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for i, r in enumerate(zip(documents, metadatas)):
            result_list.append(
                f"{i+1}. "
                + r[1]["file_path"]
                + "\n[Part of the file content is]\n "
                + r[0][:500]
            )
        result_list.append(
            "Choose the file number which is correctly retrived based on your query: "
        )
        file_numbers = int(input("\n\n".join(result_list)))
        return Response(response_message=result_list[file_numbers - 1])

    def change_summary(self, params):
        pass

    def rollback(self, params):
        pass

    def link(self, params):
        pass
