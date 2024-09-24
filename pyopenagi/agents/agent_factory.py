from threading import Lock
from pympler import asizeof
from .interact import Interactor
from ..manager.manager import AgentManager
import os
import random
import threading
import importlib

class AgentFactory:
    def __init__(
        self,
        #  agent_process_queue,
        agent_process_factory,
        agent_log_mode,
    ):
        # self.max_aid = 256
        # self.llm = llm
        # self.aid_pool = [i for i in range(self.max_aid)]
        # heapq.heapify(self.aid_pool)
        # self.agent_process_queue = agent_process_queue
        self.agent_process_factory = agent_process_factory

        self.current_agents = {}

        self.current_agents_lock = Lock()

        # self.terminate_signal = Event()

        self.agent_log_mode = agent_log_mode

        self.manager = AgentManager("http://localhost:3000")

    def snake_to_camel(self, snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    def list_agents(self):
        agent_list = self.manager.list_available_agents()

        for agent in agent_list:
            print(agent)

    # def load_agent_instance(self, compressed_name: str):
    #     name_split = compressed_name.split("/")
    #     agent_class = self.manager.load_agent(*name_split)

    #     return agent_class
    def load_agent_instance(self, agent_name):
        author, name = agent_name.split("/")
        module_name = ".".join(["pyopenagi", "agents", author, name, "agent"])
        class_name = self.snake_to_camel(name)

        agent_module = importlib.import_module(module_name)

        # dynamically loads the class
        agent_class = getattr(agent_module, class_name)
        
        return agent_class

    # def activate_agent(self, agent_name, task_input,retric_dic,redis,data_path=None, use_llm=None,raw_datapath = None,monitor_path = None):
    def activate_agent(self, agent_name, task_input):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        # downloads the agent if its not installed already
        # interactor = Interactor()

        # if not os.path.exists(os.path.join(script_dir, agent_name)):
        #     interactor.download_agent(agent_name)

        # if not interactor.check_reqs_installed(agent_name):
        #     interactor.install_agent_reqs(agent_name)

        # agent_name = "/".join(self.manager.download_agent(*agent_name.split("/")))

        # we instantiate the agent directly from the class
        agent_class = self.load_agent_instance(agent_name)
        # folder, name = agent_name.split("/")

        agent = agent_class(
            agent_name=agent_name,
            task_input=task_input,
            agent_process_factory=self.agent_process_factory,
            log_mode=self.agent_log_mode,
        )

        # use a lock to make sure only one agent can read the values at a time
        # if not self.terminate_signal.is_set():
        # with self.current_agents_lock:
        #     self.current_agents[aid] = agent

        return agent

    def run_agent(self, agent_name, task_input):
        agent = self.activate_agent(agent_name=agent_name, task_input=task_input)
        aid = threading.get_native_id()
        # print(f"Agent ID: {aid}")
        agent.set_aid(aid)
        # print(task_input)
        output = agent.run()
        # self.deactivate_agent(agent.get_aid())
        return output

    def run_retrieve(
        self,
        agent_name,
        task_input,
        retric_dic,
        redis,
        data_path,
        use_llm,
        raw_datapath=None,
        monitor_path=None,
    ):
        agent = self.activate_agent(
            agent_name=agent_name,
            task_input=task_input,
            retric_dic=retric_dic,
            redis=redis,
            data_path=data_path,
            use_llm=use_llm,
            raw_datapath=raw_datapath,
            monitor_path=monitor_path,
        )
        output = agent.run()
        # self.deactivate_agent(agent.get_aid())
        return output

    def run_retrieve(
        self,
        agent_name,
        task_input,
        retric_dic,
        redis,
        data_path,
        use_llm,
        raw_datapath=None,
        monitor_path=None,
    ):
        agent = self.activate_agent(
            agent_name=agent_name,
            task_input=task_input,
            retric_dic=retric_dic,
            redis=redis,
            data_path=data_path,
            use_llm=use_llm,
            raw_datapath=raw_datapath,
            monitor_path=monitor_path,
        )
        output = agent.run()
        self.deactivate_agent(agent.get_aid())
        return output

    def print_agent(self):
        headers = ["Agent ID", "Agent Name", "Created Time", "Status", "Memory Usage"]
        data = []
        for id, agent in self.current_agents.items():
            agent_name = agent.agent_name
            created_time = agent.created_time
            status = agent.status
            memory_usage = f"{asizeof.asizeof(agent)} bytes"
            data.append([id, agent_name, created_time, status, memory_usage])
        self.print(headers=headers, data=data)

    def print(self, headers, data):
        # align output
        column_widths = [
            max(len(str(row[i])) for row in [headers] + data)
            for i in range(len(headers))
        ]
        print("+" + "-" * (sum(column_widths) + len(headers) * 3 - 3) + "+")
        print(self.format_row(headers, column_widths))
        print("=" * (sum(column_widths) + len(headers) * 3 - 1))
        for i, row in enumerate(data):
            print(self.format_row(row, column_widths))
            if i < len(data):
                print("-" * (sum(column_widths) + len(headers) * 3 - 1))
        print("+" + "-" * (sum(column_widths) + len(headers) * 3 - 3) + "+")

    def format_row(self, row, widths, align="<"):
        row_str = " | ".join(
            f"{str(item):{align}{widths[i]}}" for i, item in enumerate(row)
        )
        return row_str

    def deactivate_agent(self, aid):
        self.current_agents.pop(aid)
        # heapq.heappush(self.aid_pool, aid)