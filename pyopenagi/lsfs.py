from aios.storage.db.redis import Redis

from aios.storage.vectordb.chromadb import ChromaDB

from watchdog.observers import Observer

from watchdog.events import FileSystemEventHandler

from agents.base_agent import BaseAgent

import os


class LSFSHandler(FileSystemEventHandler):
    def __init__(self, mount_dir) -> None:
        self.mount_dir = mount_dir
        self.db = Redis()
        self.vector_db = ChromaDB(mount_dir=mount_dir)
        self.is_start = False

    def terminate(self):
        self.is_start = False

    def on_modified(self, event):
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
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


class LSFS(BaseAgent):
    def __init__(self, mount_dir) -> None:
        self.mount_dir = mount_dir
        self.event_handler = LSFSHandler(mount_dir=mount_dir)
        self.observer = Observer()
        self.is_start = False

    def start(self):
        self.is_start = True
        self.observer.schedule(self.event_handler, self.mount_dir, recursive=True)
        self.observer.start()
        while self.is_start:
            pass
        self.observer.join()

    class LSFSParser:
        def __init__(self) -> None:
            pass

    def terminate(self):
        self.is_start = False
        self.observer.stop()

    def semantic_retrieve(self):
        pass

    def keyword_retrieve(self):
        pass

    def rollback(self):
        pass
