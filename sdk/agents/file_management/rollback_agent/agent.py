from ...RollbackAgent import RollbackAgent

class Roll_backAgent(RollbackAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 retric_dic,
                 redis,
                 agent_process_factory,
                 log_mode:str,
                 use_llm = None,
                 data_path = None,
                 sub_name=None,
                 raw_datapath = None,
                 monitor_path = None
        ):
        RollbackAgent.__init__(self,agent_name,task_input,retric_dic,redis,agent_process_factory,log_mode, data_path, use_llm, raw_datapath=raw_datapath,monitor_path=None)
        self.workflow_mode = "automatic"

        def manaul_workflow(self):
            pass
        
        def run(self):
            return super().run()