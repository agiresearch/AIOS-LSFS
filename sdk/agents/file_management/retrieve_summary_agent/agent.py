from ...RetrieveSummaryAgent import RetrieveSummary
import os
import redis
class RetrieveSummaryAgent(RetrieveSummary):
    def __init__(
        self,
        agent_name,
        task_input,
        agent_process_factory,
        # data_path,
        # use_llm,
        # retric_dic,
        # redis,
        log_mode: str,
        # sub_name=None,
        # raw_datapath=None,
        # monitor_path=None,
    ):
        use_llm = False
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../test/ragbase")
        
        RetrieveSummary.__init__(
            self,
            agent_name,
            task_input,
            data_path,
            use_llm,
            # retric_dic,
            # redis_client,
            agent_process_factory,
            log_mode,
        )
        self.workflow_mode = "automatic"

        def manaul_workflow(self):
            pass

        def run(self):
            return super().run()
