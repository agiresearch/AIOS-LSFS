from ...ChangeMonitorAgent import ChangeAgent

class ChangeMonitorAgent(ChangeAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 data_path,
                 use_llm,
                 retric_dic,
                 redis,
                 agent_process_factory,
                 log_mode:str,
                 sub_name=None,
                 raw_datapath = None,
                 monitor_path = None
        ):
        ChangeAgent.__init__(self,agent_name,task_input,data_path,use_llm,retric_dic,redis,agent_process_factory,log_mode,raw_datapath=raw_datapath,monitor_path=monitor_path)
        self.workflow_mode = "automatic"

        def manaul_workflow(self):
            pass
        
        def run(self):
            return super().run()