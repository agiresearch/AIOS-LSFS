
from pyopenagi.agents.base_agent import BaseAgent

import time

from pyopenagi.agents.agent_process import (
    AgentProcess
)

from pyopenagi.utils.chat_template import Query
from aios_base.storage.db_sdk import Data_Op

import argparse

from concurrent.futures import as_completed
import re
import json
import os
import logging
import subprocess
import ast

import numpy as np
logging.basicConfig(filename='ans.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class RetrieveSummary(BaseAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 data_path,
                 use_llm,
                 retric_dic,
                 redis_client,
                 agent_process_factory,
                 log_mode,
                 sub_name=None,
                 raw_datapath = None,
                 monitor_path = None
        ):
        BaseAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.data_path = data_path
        self.raw_datapath = raw_datapath
        self.use_llm = use_llm
        self.sub_name = sub_name
        self.retric_dic = retric_dic
        self.redis = redis_client
        self.database = Data_Op(retric_dic,self.redis)
        # self.tool_list = {
        #     "database_method": Data_Op()
        # }
        # self.workflow = self.config["workflow"]
        self.tools = None

    def match(self,input,mode):
        if mode == 'add':
            pattern = r"Please add (\w+) to (\w+) of database (\w+)"
            match = re.search(pattern, input)
            if match:
                alter_data, metaname, sub_name = match.groups(1),match.groups(2),match.groups(3)
            else:
                raise ValueError('Please input order in correct format')
            return alter_data, metaname, sub_name
        elif mode == 'delete':
            pattern1 = r"Please delete (\w+) of database (\w+)"
            pattern2 = r"Please delete the content related to (.+?) from (\w+)"
            match1 = re.search(pattern1, input)
            match2 = re.search(pattern2, input)
            if match1:
                metaname, sub_name = match1.groups(1),match1.groups(2)
                return metaname, sub_name, 'delete_collection'
            elif match2:
                text, sub_name = match2.groups(1),match2.groups(2)
                return text, sub_name, 'delete_correlation'
            else:
                raise ValueError('Please input order in correct format')
        elif mode == 'alter':
            pattern = r"Please change (\w+) to (\w+) of database (\w+)"
            match = re.search(pattern, input)
            if match:
                alter_data, metaname, sub_name = match.groups(1),match.groups(2),match.groups(3)
            else:
                raise ValueError('Please input order in correct format')
            return alter_data, metaname, sub_name
        elif mode == 'retrieve':
            pattern = r"Please retrieve the content about (.+?) in (\w+) of database (\w+) by (\w+)"
            match = re.search(pattern, input)
            if match:
                query, metaname, sub_name, method =  match.groups(1),match.groups(2),match.groups(3),match.groups(4)
                return query, metaname, sub_name, method
            else:
               raise ValueError('Please input order in correct format')
        elif mode == 'join':
            pattern = r"Please join (\w+) of database (\w+) to (\w+) of database (\w+)"
            match = re.search(pattern,input)
            if match:
                metaname2,sub_name2,metaname1,subname1 = match.groups(1),match.groups(2),match.groups(3),match.groups(4)
                return  metaname2,sub_name2,metaname1,subname1
            else:
                raise ValueError('Please input order in correct format')
        elif mode == 'contains':
            pattern = r"Please search paper contains (.+?) from (\w+)"
            match = re.search(pattern,input)
            if match:
                query = match.group(1)
                sub_name = match.group(2)
                return query,sub_name
            else:
                raise ValueError('Please input order in correct format')
        elif mode == 'about':
            pattern = r"Please search for papers about (.+?) from (\w+) in top (\w+) rank"
            match = re.match(pattern,input)
            if match:
                query, sub_name, top = match.groups(1)[0],match.groups(1)[1],match.groups(1)[2]
                return int(top), query, sub_name
            else:
                raise ValueError('Please input order in correct format')

    def pre_rag(self):
        if not os.path.exists(self.data_path):
            if self.raw_datapath is None:
                raise ValueError('Database is not existing must create it but no raw data')
            else:
                sub_name = os.path.basename(self.raw_datapath)
                self.database.create(self.data_path,self.sub_name,self.raw_datapath)
        else:
            if re.match(r"Please add", self.task_input):
                alter_data, metaname, sub_name = self.match(self.task_input,'add')
                self.database.insert(self.data_path,sub_name,alter_data,metaname)
            elif re.match(r"Please delete", self.task_input):
                metaname, sub_name, type = self.match(self.task_input,'delete')
                if type == 'delete_collection':
                    self.database.delete(self.data_path,sub_name,metaname=metaname)
                else:
                     self.database.delete(self.data_path,sub_name,text=metaname)
            elif re.match(r"Please alter", self.task_input):
                alter_data, metaname, sub_name = self.match(self.task_input,'alter')
                self.database.update(self.data_path,sub_name,alter_data,metaname)
            elif re.match(r"Please retrieve", self.task_input):
                query, metaname, sub_name, method = self.match(self.task_input,'retrieve')
                self.database.retrieve(self.data_path,sub_name,query,method,metaname)
            elif re.match(r"Please join", self.task_input):
                metaname2,sub_name2,metaname1,sub_name1 = self.match(self.task_input,'join')
                self.database.join(self.data_path,sub_name1,metaname1,metaname2,sub_name2)
            elif re.match(r"Please search paper contains",self.task_input):
                query,sub_name = self.match(self.task_input,'contains')
                if 'and' in query:
                    que = []
                    slices = query.split("and")
                    for i, s in enumerate(slices):
                        que.append(s)
                    ans, name = self.database.from_some_key_full(self.data_path,que,sub_name,con='and')
                elif 'or' in query:
                    que = []
                    slices = query.split("or")
                    for i, s in enumerate(slices):
                        que.append(s)
                    ans, name = self.database.from_some_key_full(self.data_path,que,sub_name,con='or')
                else:
                    ans, name = self.database.from_some_key_full(self.data_path,query,sub_name)
                return ans,name
            elif re.match(r"Please search for papers about", self.task_input):
                top, query,sub_name = self.match(self.task_input,'about')
                ans, name = self.database.from_some_key_sy(self.data_path,query,top,sub_name)
                return ans,name
            else:
                raise ValueError('Input Format Error!')
        
        return None,None

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

                
    def run(self):
        self.set_start_time(time=time.time())
        # self.build_system_instruction()
        # task_input, gt = self.task_input.split(";")
        # self.task_input = task_input
        # gt = [item.strip() for item in gt.split(',') if item.strip()]


        if self.use_llm is False:
            task_input = "The task you need to solve is: " + self.task_input
        else:
            task_input = "The task you need to solve is: " + self.task_input + ' then summarize it'

        self.logger.log(f"{task_input}\n", level="info")

        # ans, name = self.pre_rag()
        # print(name)
        # logging.info(name)
        # if len(gt) > 0:
        #     dele = []
        #     for i in range(len(name)):
        #         if name[i] not in gt:
        #             dele.append(i)
        #     for index in sorted(dele, reverse=True):
        #         del ans[index]
        #         del name[index]
        #     print(name)
        # logging.info(name)
        # change = input("please check the ans and delete the wrong ans:")
        # if change != 'no':
        #     change_list = change.split(',')
        #     for i in range(len(change_list)):
        #         index = name.index(change_list[i])
        #         ans.remove(index)

        # if ans is None and name is None:
        #     self.logger.log(f"{task_input} has been finished\n", level="info")
        #     return 

        # if self.use_llm is False:
        #     self.logger.log(f"{name} include relevant content, the content is {ans} \n", level="info")
        #     return {
        #         "paper name":name,
        #         "result": ans
        #     }

        request_waiting_times = []
        request_turnaround_times = []

        rounds = 0
        result = []
        # response_message = ''
        workflow = self.config['workflow']
        final_result = []
        for i, step in enumerate(workflow):
            # for j in range(len(ans)):
            #     text = ans[j]
            #     keywords = 'References'
            #     if keywords in text[0]:
            #         parts = text[0].split(keywords)
            #         text = parts[0]
            count = 7
            with open('/Users/manchester/Documents/rag/AIOS/test/file10.txt', 'r') as file:
                for line in file:  
                    count +=1
                    start = time.time()
                    print(line)
                    # collection = self.database.get_collection(self.data_path,'ragbase','CoT_no_prompt')  
                    client = self.database.get_collection(self.data_path,'ragbase')  
                    collections = client.list_collections()
                    if count <= 6:
                        for j,collection in enumerate(collections):
                            print(j)
                            text = collection.get()['documents']
                            print(collection.name)
                            keywords = 'References'
                            if keywords in text:
                                parts = text.split(keywords)
                                text = parts[0]
                            prompt = f"\nAt current step, you need to find {line}.If yes, summarize the paper, if no you don't need to respond. The paper is {text}. "
                            # prompt = f"\nDon't respond"
                            self.messages.append({"role": "user", 
                                                "content": prompt})
                            tool_use = None
                            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                            query = Query(messages = self.messages,
                                            tools = tool_use))
                            response_message = response.response_message
                            request_waiting_times.extend(waiting_times)
                            request_turnaround_times.extend(turnaround_times)                  
                            # self.messages.append({
                            #         "role": "user",
                            #         "content": response_message
                            #     })
                            self.messages = self.messages[:1]
                            logging.info(response_message)
                            self.logger.log(f"{response_message}\n", level="info")
                            if i == len(workflow) - 1:
                                final_result.append(response_message)
                    else:
                        tmp = ''
                        tool_use = None
                        prompt = f"\nIn the next step, you need to accept and remember the paper, but do not generate any outputs. Until you are told to output something"
                        self.messages.append(
                                            {"role": "user", 
                                            "content": prompt}
                                            )
                        response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                            query = Query(
                                        messages = self.messages,
                                        tools = tool_use
                                        ))
                        response_message = response.response_message
                        self.logger.log(f"{response_message}\n", level="info")
                        for j,collection in enumerate(collections):
                            print(j)
                            text = collection.get()['documents']
                            prompt = f"\nThe paper is {text}"
                            self.messages.append(
                                                {"role": "user", 
                                                "content": prompt}
                                            )
                            tool_use = None
                            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                            query = Query(
                                        messages = self.messages,
                                        tools = tool_use
                                        ))
                            response_message = response.response_message
                            self.logger.log(f"{response_message}\n", level="info")
                            # self.messages = self.messages[:1]
                            
                            if (j + 1) % 5 == 0 and j != 0:
                                # self.messages = []
                                prompt = f"\n Now you can to output the answer. You need to find 2 papers which most relate to LLM Uncertainty from previous record and summary them respectively"
                                self.messages.append(
                                                    {"role": "user", 
                                                    "content": prompt}
                                                    )
                                response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                                query = Query(
                                                messages = self.messages,
                                                tools = tool_use
                                                ))
                                response_message = response.response_message
                                self.logger.log(f"{response_message}\n", level="info")
                                tmp +=response_message
                                self.messages = self.messages[:1]
                        self.messages = []
                        if count == 7:
                            prompt = f"\n Now you can to output the answer. you need to choose from {tmp} to find the top two papers that is most relevant to prompt engineering"
                        elif count == 8:
                            prompt = f"\n Now you can to output the answer. you need to choose from {tmp} to find the top two papers that is most relevant to LLM Uncertainty"
                        elif count == 9:
                            prompt = f"\n Now you can to output the answer. you need to choose from {tmp} to find the top three papers that is most relevant to Large Language Model"
                        elif count == 10:
                            prompt = f"\n Now you can to output the answer. you need to choose from {tmp} to find the top two papers that is most relevant to Adversarial attack"
                        else:
                            break
                        self.messages.append(
                                    {"role": "user", 
                                     "content": prompt})
                        response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                        query = Query(
                                        messages = self.messages,
                                        tools = tool_use
                                        ))
                        response_message = response.response_message
                        request_waiting_times.extend(waiting_times)
                        request_turnaround_times.extend(turnaround_times)       
                        self.messages.append({
                                    "role": "user",
                                    "content": response_message
                                })
                        if i == len(workflow) - 1:
                                final_result.append(self.messages[-1])
                        self.messages = []
                        logging.info(response_message)
                        self.logger.log(f"{response_message}\n", level="info")

                    point = time.time()
                    logging.info(f'one turn time:{point - start}')

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
