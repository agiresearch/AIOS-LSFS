from pyopenagi.agents.base_agent import BaseAgent

import time

from aios.hooks.request import AgentProcess

from pyopenagi.utils.chat_template import Query
from aios_base.storage.db_sdk import Data_Op
import threading
import argparse

from concurrent.futures import as_completed
import re
import json
import os
import logging
import subprocess
import redis
from pyopenagi.utils.filereader import update_file

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="/Users/manchester/Documents/rag/AIOS/change_record.log",
)


class RollbackAgent(BaseAgent):

    # The format of rollback time must be YYYY-MM-DD
    def __init__(
        self,
        agent_name,
        task_input,
        retric_dic,
        redis,
        agent_process_factory,
        log_mode,
        use_llm=None,
        data_path=None,
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

    def build_system_instruction(self):
        prefix = "".join(["".join(self.config["description"])])
        self.messages.append({"role": "system", "content": prefix})

    def automatic_workflow(self):
        return super().automatic_workflow()

    def manual_workflow(self):
        pass

    def stop_monitoring(self):
        self.active = False

    def match(self, text, flag=False):
        result = text.split(",")
        name, version = result[0], result[1]
        if flag:
            return name, version
        else:
            return name, int(version)

        return name, date

    def rollback(self, name, date, flag=False):
        self.redis_client.select(1)
        key_with_name = []
        cursor = "0"
        while cursor != 0:
            cursor, keys = self.redis_client.scan(
                cursor=cursor, match="*{}*".format(name), count=100
            )
            key_with_name.extend([key for key in keys])
        if len(key_with_name) > 1:
            print(
                f"\nThere is more than one term containing {name}, the list of them is {key_with_name}"
            )
            db_name = input("You should choose one of them:")
        elif len(key_with_name) == 0:
            raise Exception("No such file")
        else:
            db_name = key_with_name[0]

        ll = self.redis_client.llen(db_name)
        files = None
        if flag:
            for i in range(ll):
                file_info_json = self.redis_client.lindex(db_name, i)
                file_info = json.loads(file_info_json)
                if file_info["last_modified_date"] == date:
                    files = file_info
        else:
            for i in range(ll):
                if i + 1 == date:
                    file_info_json = self.redis_client.lindex(db_name, i)
                    file_info = json.loads(file_info_json)
                    files = file_info
        if files is None:
            raise ValueError(
                f"\nThe file didn't change in {date} or it has been overdued"
            )

        document = files["content"]
        db_path = files["db_path"]
        sub_name = files["sub_name"]
        text_path = files["text_path"]

        update_file(text_path, document)

        collection_fore = self.database.get_collection(db_path, sub_name, metaname=name)
        text_before = collection_fore.get()["documents"]
        text_date = collection_fore.get()["metadatas"][0]["last_modified_date"]
        self.version(name, text_date, text_before, text_path, db_path)
        path = os.path.join(db_path, sub_name, name)
        jud = input(
            f"\n The target file is {path}, you need to confirm it. Please input yes or no:"
        )
        if jud.lower() == "no":
            print("rollback failed")
            return
        print("rollback success")

    def run(self):
        self.set_start_time(time=time.time())
        self.redis_client.select(1)
        self.build_system_instruction()

        task_input = "The task you need to solve is: " + self.task_input

        self.logger.log(f"{task_input}\n", level="info")

        request_waiting_times = []
        request_turnaround_times = []
        rounds = 0
        # with open('/Users/manchester/Documents/rag/AIOS/test/rollback.txt', 'r') as file:
        #     for line in file:
        #         if '-' in line:
        #             workflow = self.config['workflow'][0]
        #         else:
        #             workflow = self.config['workflow'][1]
        #         prompt = f"\nAt current step, you need to {workflow}.The sentence is {line}. Here is the example, if you input Please rollback file named quantum to the version in 2024-01-03, the file name is quantum and the time is 2024-01-03, so you should output like\
        #        \'quantum, 2024-01-03\'. if you input Please rollback file named quantum 5 versions, the file name is quantum and the rollback version number is 5, so you should output \'quantum, 5\'. You need to output like format without other words"
        #         self.messages.append(
        #                     {"role": "user",
        #                     "content": prompt})
        #         tool_use = None
        #         response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
        #         query = Query(
        #                     messages = self.messages,
        #                     tools = tool_use
        #                     ))
        #         response_message = response.response_message
        #         logging.info(response_message)
        #         self.messages = self.messages[:1]
        # print(ssd)

        if "-" in self.task_input:
            workflow = self.config["workflow"][0]
        else:
            workflow = self.config["workflow"][1]
        prompt = f"\nAt current step, you need to {workflow} {self.task_input}. Here is the example, if you input Please rollback file named quantum to the version in 2024-01-03, you should output like\
            'quantum, 2024-01-03'. if you input Please rollback file named quantum 5 versions, you should output 'quantum, 5'. You need to output like format without other words"
        self.messages.append({"role": "user", "content": prompt})
        tool_use = None
        response, start_times, end_times, waiting_times, turnaround_times = (
            self.get_response(query=Query(messages=self.messages, tools=tool_use))
        )

        response_message = response.response_message
        request_waiting_times.extend(waiting_times)
        request_turnaround_times.extend(turnaround_times)

        # if i == 0:
        #     self.set_start_time(start_times[0])

        # tool_calls = response.tool_calls

        self.messages.append({"role": "user", "content": response_message})
        final_result = self.messages[-1]
        self.logger.log(f"{response_message}\n", level="info")
        rounds += 1
        self.set_status("done")

        if "-" in self.task_input:
            name, date = self.match(response_message, flag=True)
            jud = input(
                "The rollback file name is {name}, you need to confirm it. Please input yes or no:"
            )
            if jud.lower() == "no":
                return
            self.rollback(name, date, flag=True)
        else:
            name, date = self.match(response_message)
            self.rollback(name, date)

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

    def version(self, text_name, text_date, text_before, text_path, db_path):
        self.redis_client.select(1)

        file_info = {
            "file_name": text_name,
            "last_modified_date": text_date,
            "content": text_before,
            "text_path": text_path,
            "db_path": db_path,
        }
        redis_key = file_info["file_name"]
        file_info_json = json.dumps(file_info)

        if self.redis_client.llen(redis_key) > 5 and self.redis_client.exists(
            redis_key
        ):
            loc = self.redis_client.lindex(redis_key, -1)
            self.redis_client.ltrim(loc, 0, 9)

        self.redis_client.rpush(redis_key, file_info_json)

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
