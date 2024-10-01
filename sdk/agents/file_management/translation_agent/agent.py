from ...TranslationAgent import TranslationAgent

class TransltionAgent(TranslationAgent):
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
        TranslationAgent.__init__(self,agent_name,task_input,retric_dic,redis,agent_process_factory,log_mode, data_path, use_llm, raw_datapath=raw_datapath,monitor_path=None)
        self.workflow_mode = "automatic"

        def manaul_workflow(self):
            pass
        
        def run(self):
            return super().run()