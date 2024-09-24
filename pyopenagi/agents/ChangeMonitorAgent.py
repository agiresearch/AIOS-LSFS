from pyopenagi.agents.base_agent import BaseAgent
import time
from pyopenagi.agents.agent_process import AgentProcess
from pyopenagi.utils.chat_template import Query
from aios.storage.db_sdk import Data_Op
import threading
import argparse
from pyopenagi.utils.filereader import update_file

from concurrent.futures import as_completed
import re
import json
import os
import logging
import subprocess
import redis
import threading
import fcntl

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="/Users/manchester/Documents/rag/AIOS/change_record.log",
)


class ChangeAgent(BaseAgent):
    def __init__(
        self,
        agent_name,
        task_input,
        data_path,
        use_llm,
        retric_dic,
        redis,
        agent_process_factory,
        log_mode,
        raw_datapath=None,
        monitor_path=None,
        sub_name=None,
    ):
        BaseAgent.__init__(
            self, agent_name, task_input, agent_process_factory, log_mode
        )
        self.data_path = data_path
        self.raw_datapath = raw_datapath
        self.use_llm = use_llm
        self.sub_name = sub_name
        self.monitor_path = monitor_path
        self.file_mod_times = {}
        self.active = False
        self.retric_dic = retric_dic
        self.tools = None
        self.redis_client = redis
        self.database = Data_Op(retric_dic, self.redis_client)
        self.lock = threading.Lock()

    def scan_files(self):
        for root, _, files in os.walk(self.monitor_path):
            for file in files:
                filepath = os.path.join(root, file)
                self.file_mod_times[filepath] = os.path.getmtime(filepath)

    def monitor_files(self):
        while self.active:
            for filepath in list(self.file_mod_times.keys()):
                if os.path.exists(filepath):
                    current_mod_time = os.path.getmtime(filepath)
                    if current_mod_time != self.file_mod_times[filepath]:
                        with self.lock:
                            print(f"File modified: {filepath}")
                            self.file_mod_times[filepath] = current_mod_time
                            file_name = os.path.basename(filepath)
                            path_parts = filepath.split(os.sep)
                            folder_name = path_parts[-2]
                            file_name, _ = os.path.splitext(file_name)
                            self.monitor_run(folder_name, file_name, filepath)
                else:
                    print(f"File deleted: {filepath}")
                    del self.file_mod_times[filepath]

    def start_monitoring(self):
        self.active = True
        self.scan_files()
        monitor_thread = threading.Thread(target=self.monitor_files)
        monitor_thread.start()
        return monitor_thread

    def build_system_instruction(self, i):
        prefix = "".join(["".join(self.config["description"][i])])
        self.messages.append({"role": "system", "content": prefix})

    def automatic_workflow(self):
        return super().automatic_workflow()

    def manual_workflow(self):
        pass

    def stop_monitoring(self):
        self.active = False

    def lock_file(self, file):
        fcntl.flock(file, fcntl.LOCK_EX)

    def unlock_file(self, file):
        fcntl.flock(file, fcntl.LOCK_UN)

    def run(self):
        # self.redis_client.select(1)
        # keys = self.redis_client.keys('*')
        # for key in keys:
        #     print(self.redis_client.llen(key))
        # values = self.redis_client.lrange(key, 0, -1)  # 获取列表中的所有元素
        # print(values)

        self.build_system_instruction(0)

        if self.use_llm is False:
            task_input = "The task you need to solve is: " + self.task_input
        else:
            task_input = (
                "The task you need to solve is: "
                + self.task_input
                + " than summarize the difference between two files"
            )

        i = 0
        # with open('/Users/manchester/Documents/rag/AIOS/test/cs_example.txt', 'r') as file:

        workflow = self.config["workflow"][0]
        prompt = f"\nAt current step, you need to {workflow}, the sentence is {self.task_input}. Here is the example, if the input is  Please change content in '/Users/manchester/Documents/rag/rag_source/physics/quantum.txt' to old_quan in physics\
                    Your output should be'/Users/manchester/Documents/rag/rag_source/physics/quantum.txt, old_quan, physics'. You need to output like format without other words"
        self.messages.append({"role": "user", "content": prompt})
        response, start_times, end_times, waiting_times, turnaround_times = (
            self.get_response(query=Query(messages=self.messages, tools=None))
        )
        response_message = response.response_message
        result = response_message.split(",")
        alter_data, metaname, sub_name = result[0], result[1], result[2]

        jud = input(
            "You will use content in {alter_data} change {metaname}. You should confirm it. Please input yes or no:"
        )
        if jud.lower() == "no":
            return

        self.logger.log(f"{task_input}\n", level="info")
        path = os.path.join(self.data_path, sub_name, metaname)
        if not os.path.exists(path):
            if self.raw_datapath is None:
                raise Exception("Database does's not exist and raw_date is empty")
            self.database.create(
                self.data_path, sub_name, self.raw_datapath, metaname=metaname
            )

        self.monitor_run(sub_name, metaname, alter_data)
        if not self.active:
            self.logger.log("monitor thread begins\n", level="info")
            monitor_thread = self.start_monitoring()
        while True:
            command = input("Enter 'stop' to stop monitoring: ").strip().lower()
            if command == "stop":
                self.stop_monitoring()
                monitor_thread.join()
                break

    def version(self, text_name, text_date, text_before, text_path, db_path, sub_name):
        self.redis_client.select(1)
        file_info = {
            "file_name": text_name,
            "last_modified_date": text_date,
            "content": text_before,
            "text_path": text_path,
            "db_path": db_path,
            "sub_name": sub_name,
        }
        redis_key = file_info["file_name"]
        file_info_json = json.dumps(file_info)

        if self.redis_client.llen(redis_key) > 40 and self.redis_client.exists(
            redis_key
        ):
            loc = self.redis_client.lindex(redis_key, -1)
            self.redis_client.ltrim(loc, 0, 9)

        self.redis_client.rpush(redis_key, file_info_json)

    def monitor_run(self, sub_name, metaname, alter_data):

        with self.lock:
            sfm_file = os.path.join(self.data_path, sub_name, metaname)

            collection_fore = self.database.get_collection(
                self.data_path, sub_name, metaname=metaname
            )
            text_before = collection_fore.get()["documents"]

            text_path = collection_fore.get()["metadatas"][0]["file_path"]
            path_parts = text_path.split(os.sep)
            text_name = os.sep.join(path_parts[-2:])
            text_date = collection_fore.get()["metadatas"][0]["last_modified_date"]

            self.version(
                text_name, text_date, text_before, text_path, self.data_path, sub_name
            )

            self.lock_file(sfm_file)
            self.lock_file(text_path)

            collection_lat = self.database.update(
                self.data_path, sub_name, alter_data, obj=metaname
            )
            text_end = collection_lat.get()["documents"]
            update_file(text_path, text_end)

            self.unlock_file(sfm_file)
            self.unlock_file(text_path)

            print("version remember !!!!!!!!!!")

            self.messages = []
            self.build_system_instruction(1)

            request_waiting_times = []
            request_turnaround_times = []

            rounds = 0
            result = []
            # response_message = ''

            workflow = self.config["workflow"][1]
            # for i, step in enumerate(workflow):
            prompt = f"\nAt current step, you need to {workflow}, the content before the update is <context>{text_before}</context>, the content after the update is <context>{text_end}</context>"
            self.messages.append({"role": "user", "content": prompt})
            tool_use = None
            response, start_times, end_times, waiting_times, turnaround_times = (
                self.get_response(query=Query(messages=self.messages, tools=tool_use))
            )

            response_message = response.response_message

            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)

            # if i == 0:
            self.set_start_time(start_times[0])

            # tool_calls = response.tool_calls

            self.messages.append({"role": "user", "content": response_message})

            # if i == len(workflow) - 1:
            final_result = self.messages[-1]
            self.logger.log(f"{response_message}\n", level="info")
            rounds += 1
            self.set_status("done")
            self.set_end_time(time=time.time())

            return {
                "agent_name": self.agent_name,
                "result": final_result,
                "rounds": rounds,
                "agent_waiting_time": self.start_time - self.created_time,
                "agent_turnaround_time": self.end_time - self.created_time,
                "request_waiting_times": request_waiting_times,
                "request_turnaround_times": request_turnaround_times,
            }

    def parse_result(self, prompt):
        return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NarrativeAgent")
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    "Please search  "
    # args = parser.parse_args()
    # agent = FileAgent(args.agent_name, args.task_input)
    # agent.run()
