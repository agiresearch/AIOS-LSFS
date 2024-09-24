
from pyopenagi.agents.base_agent import BaseAgent

import time

from pyopenagi.agents.agent_process import (
    AgentProcess
)

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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename='/Users/manchester/Documents/rag/AIOS/change_record.log')
class TranslationAgent(BaseAgent):
    def __init__(self,
                 agent_name,
                 data_path,
                 task_input,
                 retric_dic,
                 redis,
                 agent_process_factory,
                 log_mode,
                 use_llm = None,
                 raw_datapath = None,
                 sub_name=None,
                 monitor_path = None,
        ):
        BaseAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.data_path = data_path
        self.raw_datapath = raw_datapath
        self.use_llm = use_llm
        self.sub_name = sub_name
        self.file_mod_times = {}
        self.active = False
        self.retric_dic = retric_dic
        self.tools = None
        self.redis_client = redis
        self.database = Data_Op(retric_dic,self.redis_client)

    def build_system_instruction(self):
        prefix = "".join(
            [
                "".join(self.config["description"])
            ]
        )        
        self.messages.append(
                {"role": "system", "content": prefix }
            )

    
    def automatic_workflow(self):
        return super().automatic_workflow()
    
    def manual_workflow(self):
        pass
    
    def stop_monitoring(self):
        self.active = False
    
    def match(self,text):
        result = text.split(",")
        name, language = result[0], result[1]
        
        return name, language
    def select_doc(self,name):
        if os.path.exists(name):
            parts = name.split(os.sep)
            parts = parts[-2:]
            sub_name, metaname = parts[0], parts[1]
            collection = self.database.get_collection(self.data_path,sub_name,metaname)
            doc = collection.get()['documents']
            return doc
        else:
            ans_list = []
            for sub_name in os.listdir(self.data_path):
                if sub_name == '.DS_Store':
                    continue
                client = self.database.get_collection(self.data_path,sub_name)
                collections = client.list_collections()
                for collection in collections:
                    if name == collection.name:
                        ans_list.append(collection.get()['metadatas'][0]['file_path'])
            
            if len(ans_list) > 1:
                print(f"\nThere is more than one term containing {name}, the list of them is {ans_list}")
                db_name = input("You should choose one of them:")
            elif len(ans_list) == 0:
                raise Exception("No such file")
            else:
                db_name = ans_list[0]


            parts = db_name.split(os.sep)
            parts = parts[-2:]
            sub_name, metaname = parts[0], parts[1]
            metaname = os.path.splitext(metaname)[0]
            collection = self.database.get_collection(self.data_path,'physics',metaname)
            doc = collection.get()['documents']
            return doc

    def run(self):
        self.build_system_instruction()

        task_input = "The task you need to solve is: " + self.task_input

        self.logger.log(f"{task_input}\n", level="info")

        request_waiting_times = []
        request_turnaround_times = []
        rounds = 0
        workflow = self.config['workflow']
        name , language, doc = None, None, None
        for i, step in enumerate(workflow):
            if i == 0:
                prompt = f"\nAt current step, you need to {workflow[i]} {self.task_input}"
            else:
                self.messages = self.messages[:1]
                prompt = f"\nAt current step, you need to {workflow[i]} {language}, <context>{doc}</context>"

            self.messages.append(
                    {"role": "user", 
                    "content": prompt}
                )
            tool_use = None
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
            query = Query(
                    messages = self.messages,
                    tools = tool_use
                    )
            )
            response_message = response.response_message
            if i == 0:
                name, language = self.match(response_message)
                doc = self.select_doc(name)

            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)

            if i == 0:
                self.set_start_time(start_times[0])
                
            self.messages.append({
                    "role": "user",
                    "content": response_message
                })

            if i == len(workflow) - 1:
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
    parser = argparse.ArgumentParser(description='Run NarrativeAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    "Please search  "
    # args = parser.parse_args()
    # agent = FileAgent(args.agent_name, args.task_input)
    # agent.run()
