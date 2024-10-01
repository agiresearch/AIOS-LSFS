from pyopenagi.agents.base_agent import BaseAgent

import time

from aios.hooks.request import AgentProcess

from pyopenagi.utils.chat_template import Query
from aios.storage.db_sdk import Data_Op

import argparse

from concurrent.futures import as_completed
import re
import json
import os
import logging
import subprocess
import ast

import numpy as np


# logging.basicConfig(filename='ans.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class RetrieveSummary(BaseAgent):
    def __init__(
        self,
        agent_name,
        task_input,
        data_path,
        use_llm,
        retric_dic,
        redis_client,
        agent_process_factory,
        log_mode,
        sub_name=None,
        raw_datapath=None,
        monitor_path=None,
    ):
        BaseAgent.__init__(
            self, agent_name, task_input, agent_process_factory, log_mode
        )
        self.data_path = data_path
        self.raw_datapath = raw_datapath
        self.use_llm = use_llm
        self.sub_name = sub_name
        self.retric_dic = retric_dic
        self.redis = redis_client
        self.database = Data_Op(retric_dic, self.redis)
        # self.tool_list = {
        #     "database_method": Data_Op()
        # }
        # self.workflow = self.config["workflow"]
        self.tools = None

    def pre_rag(self, task_input):
        s_messages = []
        if (
            "top" in task_input
            or "most" in task_input
            or "highest" in task_input
            or "strongest" in task_input
        ):
            s_messages.append(
                {
                    "role": "system",
                    "content": "You are good at extracting messages from sentence",
                }
            )
            workflow = "You should extract the number of papers retrieved, query, directory from it directly and separate them by commas. If some of them doensn't contain in the sentence, you need to output None\
                        Here is the example: if the input is Locate the 3 papers showing the highest correlation with reinforcement learning in LLM_training. The search number is 3, the query is reinforcement learning and the directory is LLM_training\
                        you need to output like this format'3, reinforcement learning, LLM_training' directly. if the input is Search for the top 2 most relevant papers on the security of large models.The search number is 2, the query is the security of large models. The sentence does not have directory. You need to output like this format'2, the security of large models, None' directly\
                        The second item only needs to include the target keyword and does not need to include the condition.You need to output answer like format in example without other words"
            prompt = f"\nNow {workflow}, the sentence is {task_input}"
            s_messages.append({"role": "user", "content": prompt})
            tool_use = None
            response, start_times, end_times, waiting_times, turnaround_times = (
                self.get_response(query=Query(messages=s_messages, tools=tool_use))
            )
            response_message = response.response_message
            print(f"Response message: {response_message}")
            # logging.info(response_message)
            result = response_message.split(",")
            top_k, query, sub_name = result[0], result[1], None if result[2].strip() == "None" else result[2]
            if top_k.isdigit():
                top_k = int(top_k)
            else:
                raise ValueError("you need input Arabic numerals in the top files")
            ans, name = self.database.semantic_retrieve(
                self.data_path, query, top_k, db_name=sub_name
            )
        else:
            s_messages.append(
                {
                    "role": "system",
                    "content": "You are good at extracting messages from sentence",
                }
            )
            workflow = "You should extract the search key, directory from it directly and separate them by commas.As for search key, you should only output core words such as name or school and so on. If some of them doensn't contain in the sentence, you need to output None. \
                        Here is the example: if the input is Please search for papers whose authors include Amy. The search key is Amy and there is no directory, you should output 'Amy, None' directly. If the input is Please search llm_directory for articles with authors from Peking University and Rutgers University, which contains two school. Thus,\
                        the search key contains Peking University and Rutgers University and the directoy is llm_directory, you should output 'Peking University and Rutgers University, llm_directory' directly. You need to output answer like format in example without other words."
            prompt = f"\nNow {workflow}, the sentence is {task_input}"
            s_messages.append({"role": "user", "content": prompt})
            tool_use = None
            response, start_times, end_times, waiting_times, turnaround_times = (
                self.get_response(query=Query(messages=s_messages, tools=tool_use))
            )
            response_message = response.response_message
            print(response_message)
            logging.info(response_message)
            result = response_message.split(",")
            query = result[0]
            sub_name = result[1]
            que = []
            print(query)
            if "and" in query:
                que_tmp = query.split("and")
                for i in range(len(que_tmp)):
                    que.append(que_tmp[i])
                ans, name = self.database.keyword_retrieve(
                    self.data_path, que, db_name=sub_name, con="and"
                )
            elif "or" in query:
                que_tmp = query.split("and")
                for i in range(len(que_tmp)):
                    que.append(que_tmp[i])
                ans, name = self.database.keyword_retrieve(
                    self.data_path, que, db_name=sub_name, con="or"
                )
        # return result
        return ans, name

    def build_system_instruction(self):
        prefix = "".join(["".join(self.config["description"])])
        self.messages.append({"role": "system", "content": prefix})

    def automatic_workflow(self):
        return super().automatic_workflow()

    def manual_workflow(self):
        pass

    def run(self):
        self.set_start_time(time=time.time())
        # task_input, gt = self.task_input.split(";")
        # self.task_input = task_input
        # gt = [item.strip() for item in gt.split(',') if item.strip()]

        if self.use_llm is False:
            task_input = "The task you need to solve is: " + self.task_input
        else:
            task_input = (
                "The task you need to solve is: "
                + self.task_input
                + " then summarize it"
            )

        self.logger.log(f"{task_input}\n", level="info")

        # with open('/Users/manchester/Documents/rag/AIOS/test/rs_example.txt', 'r') as file:
        #     for line in file:
        #         name = self.pre_rag(line)
        #         print(name)
        # print(ssd)

        ans, name = self.pre_rag(self.task_input)
        print(name)
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
        change = input("please check the ans and delete the wrong ans:")

        if change.lower() != "no":
            change_list = change.split(",")
            print(change_list)
            for i in range(len(change_list)):
                index = name.index(change_list[i])
                name.pop(index)
                ans.pop(index)

        if ans is None and name is None:
            self.logger.log(f"{task_input} has been finished\n", level="info")
            return

        if self.use_llm is False:
            self.logger.log(
                f"{name} include relevant content, the content is {ans} \n",
                level="info",
            )
            return {"paper name": name, "result": ans}

        request_waiting_times = []
        request_turnaround_times = []
        self.build_system_instruction()
        rounds = 0
        result = []
        # response_message = ''
        workflow = self.config["workflow"]
        final_result = []
        for i, step in enumerate(workflow):
            for j in range(len(ans)):
                text = ans[j]
                keywords = "References"
                if keywords in text[0]:
                    parts = text[0].split(keywords)
                    text = parts[0]
                elif "REFERENCES" in text[0]:
                    parts = text[0].split("REFERENCES")
                    text = parts[0]
                prompt = f"\nAt current step, you need to {workflow},<context>{text}</context>"
                self.messages.append({"role": "user", "content": prompt})
                tool_use = None
                response, start_times, end_times, waiting_times, turnaround_times = (
                    self.get_response(
                        query=Query(messages=self.messages, tools=tool_use)
                    )
                )
                # self.messages = [self.messages[0]]

                # if k == len(content) - 1:
                response_message = response.response_message

                request_waiting_times.extend(waiting_times)
                request_turnaround_times.extend(turnaround_times)

                # if i == 0:
                #     self.set_start_time(start_times[0])

                # tool_calls = response.tool_calls

                self.messages.append({"role": "user", "content": response_message})

                if i == len(workflow) - 1:
                    final_result.append(self.messages[-1])
                self.messages = self.messages[:1]
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
