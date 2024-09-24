
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
from pyopenagi.tools.google.google_link import GoogleLink

import numpy as np

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename='/Users/manchester/Documents/rag/AIOS/change_record.log')
class LinkAgent(BaseAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 data_path,
                 retric_dic,
                 redis,
                 agent_process_factory,
                 log_mode,
                 use_llm = None,
                 raw_datapath = None,
                 monitor_path = None,
                 sub_name=None,
        ):
        BaseAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.data_path = data_path
        self.raw_datapath = raw_datapath
        self.use_llm = use_llm
        self.sub_name = sub_name
        self.monitor_path = monitor_path
        self.file_mod_times = {}
        self.active = False
        self.retric_dic =retric_dic,
        self.tools = None
        # self.link = GoogleLink()
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
        name, date = result[0], result[1]
        if date == 'None':
            return name,None

        pattern = re.compile(r'(?:(\d+)\s*days?)?\s*(?:(\d+)\s*weeks?)?\s*(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*minutes?)?\s*(?:(\d+)\s*seconds?)?')
        match = pattern.search(date)
    
        if match:
            days = int(match.group(1)) if match.group(1) else 0
            weeks = int(match.group(2)) if match.group(2) else 0
            hours = int(match.group(3)) if match.group(3) else 0
            minutes = int(match.group(4)) if match.group(4) else 0
            seconds = int(match.group(5)) if match.group(5) else 0

        return name, (days,weeks,hours,minutes,seconds)
    def search_path(self,name):
        if os.path.exists(name):
            return name
        else:
            ans_list = []
            name = os.path.splitext(name)[0]
            for sub_name in os.listdir(self.data_path):
                if sub_name == '.DS_Store':
                    continue
                client = self.database.get_collection(self.data_path,sub_name)
                collections = client.list_collections()
                for collection in collections:
                    fname = os.path.splitext(collection.name)[0]
                    if name == fname:
                        ans_list.append(collection.get()['metadatas'][0]['file_path'])
            
            if len(ans_list) > 1:
                print(f"\nThere is more than one term containing {name}, the list of them is {ans_list}")
                db_name = input("You should choose one of them:")
            elif len(ans_list) == 0:
                raise Exception("No such file")
            else:
                db_name = ans_list[0]
        return db_name

    def run(self):
        # self.redis_client.select(1)
        self.build_system_instruction()

        # task_input = "The task you need to solve is writing a codes to generate a link for: " + self.task_input

        # self.logger.log(f"{task_input}\n", level="info")

        request_waiting_times = []
        request_turnaround_times = []
        rounds = 0
        workflow = self.config['workflow']
        for i, step in enumerate(workflow):
            # with open('/Users/manchester/Documents/rag/AIOS/test/link.txt', 'r') as file:
            #     for line in file:
            #         prompt = f"\nAt current step, you should to {workflow}. The sentence is {line}. Here is the example, if you input Please generate a validity period of 5 days and 3 hours for file named aios. You should \
            #                 output in this format \'aios, 5 days 3 hours\'.  You need to output like format without other extra words"
            #         self.messages.append({"role": "user", "content": prompt})
            #         tool_use = None
            #         response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
            #         query = Query(
            #                 messages = self.messages,
            #                 tools = tool_use
            #                 )
            #         )
            #         response_message = response.response_message
            #         print(response_message)
            #         logging.info(response_message)
            #         self.messages = self.messages[:1]
            # print(ssd)
            # prompt = f"\nAt current step, you need to {workflow} {self.task_input}. Here is the example, if you input Please generate a validity period of 5 days and 3 hours for aios. You need \
            #     to output \'aios, 5 days 3 hours\'. If you input Please generate links for aios, there is not period of validity, you should output \'aios, None\'. You need to output like format without other words"
            prompt = f"\nAt current step,{workflow}, {self.task_input}"
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
            request_waiting_times.extend(waiting_times)
            request_turnaround_times.extend(turnaround_times)

            if i == 0:
                self.set_start_time(start_times[0])

                # tool_calls = response.tool_calls
                
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
        
        # name, date = self.match(response_message)
        # path = self.search_path(name)
        # link = self.link.generate_shareable_link(path,date)
        # self.logger.log(f"{link}\n", level="info")

        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": rounds,
            "agent_waiting_time": self.start_time - self.created_time,
            "agent_turnaround_time": self.end_time - self.created_time,
            "request_waiting_times": request_waiting_times,
            "request_turnaround_times": request_turnaround_times,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NarrativeAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    "Please search  "
    # args = parser.parse_args()
    # agent = FileAgent(args.agent_name, args.task_input)
    # agent.run()
