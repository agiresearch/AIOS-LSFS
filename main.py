# This is a main script that tests the functionality of specific agents.
# It requires no user input.

from aios.utils.utils import (
    parse_global_args,
)
import os
import warnings

from aios.hooks.llm import useFactory, useKernel, useFIFOScheduler

from aios.utils.utils import delete_directories

from aios.storage.lsfs import LSFS

from dotenv import load_dotenv

def clean_cache(root_directory):
    targets = {
        ".ipynb_checkpoints",
        "__pycache__",
        ".pytest_cache",
        "context_restoration",
    }
    delete_directories(root_directory, targets)


def main():
    # parse arguments and set configuration for this run accordingly
    main_id = os.getpid()
    print(f"Main ID is: {main_id}")
    warnings.filterwarnings("ignore")
    parser = parse_global_args()
    args = parser.parse_args()

    llm_name = args.llm_name
    max_gpu_memory = args.max_gpu_memory
    eval_device = args.eval_device
    max_new_tokens = args.max_new_tokens
    scheduler_log_mode = args.scheduler_log_mode
    agent_log_mode = args.agent_log_mode
    llm_kernel_log_mode = args.llm_kernel_log_mode
    use_backend = args.use_backend
    load_dotenv()

    llm = useKernel(
        llm_name=llm_name,
        max_gpu_memory=max_gpu_memory,
        eval_device=eval_device,
        max_new_tokens=max_new_tokens,
        log_mode=llm_kernel_log_mode,
        use_backend=use_backend
    )
    
    lsfs = LSFS(
        mount_dir=os.path.join(os.path.abspath(os.path.abspath(__file__)), "lsfs-test")
    )

    # run agents concurrently for maximum efficiency using a scheduler

    startScheduler, stopScheduler = useFIFOScheduler(
        llm=llm,
        lsfs=lsfs,
        log_mode=scheduler_log_mode,
        get_queue_message=None
    )

    submitAgent, awaitAgentExecution = useFactory(
        log_mode=agent_log_mode,
        max_workers=64
    )

    startScheduler()

    # register your agents and submit agent tasks
    agent_id = submitAgent(
        agent_name="example/academic_agent",
        task_input="tell me what is the training method used in the prollm paper"
    )

    """
    submitAgent(
        agent_name="om-raheja/transcribe_agent",
        task_input="listen to my yap for 5 seconds and write a response to it"
    )
    """
    
    """
    submitAgent(
        agent_name="example/cocktail_mixlogist",
        task_input="Create a cocktail for a summer garden party. Guests enjoy refreshing, citrusy flavors. Available ingredients include vodka, gin, lime, lemon, mint, and various fruit juices."
    )
    """
    
    """
    submitAgent(
        agent_name="example/cook_therapist",
        task_input="Develop a low-carb, keto-friendly dinner that is flavorful and satisfying."
    )
    """
    
    # agent_tasks = [
    #     ["example/academic_agent", "Tell me what is the prollm paper mainly about"]
    #     # ["example/cocktail_mixlogist", "Create a cocktail for a summer garden party. Guests enjoy refreshing, citrusy flavors. Available ingredients include vodka, gin, lime, lemon, mint, and various fruit juices."]
    # ]
    
    # i = 0
    # time = 0
    # with open('/Users/manchester/Documents/rag/AIOS/test/files10_summary.txt', 'r') as file:
    #     for line in file:
    #         i +=1
    #         retrieve_summaryagent = agent_thread_pool.submit(
    #                         agent_factory.run_retrieve,
    #                         "file_management/retrieve_summary_agent",
    #                         line,
    #                         retric_dic,
    #                         redis_client,
    #                         '/Users/manchester/Documents/data/rag_test20',
    #                         True)
    #         agent_tasks = [retrieve_summaryagent]
    #         for r in as_completed(agent_tasks):
    #                     _res = r.result()
    #                     time += _res['agent_turnaround_time']
    #                     logging.info(_res['agent_turnaround_time'])

    # time = 0

    # retrieve_summaryagent = agent_thread_pool.submit(
    #             agent_factory.run_retrieve,
    #             "file_management/retrieve_summary_agent",
    #             "Please search for the 2 papers with the highest correlation with large model uncertainty.",
    #             retric_dic,
    #             redis_client,
    #             '/Users/manchester/Documents/data/rag_test1',
    #             True)
    # agent_tasks = [retrieve_summaryagent]
    # for r in as_completed(agent_tasks):
    #     _res = r.result()
    

    # change_monitoragent = agent_thread_pool.submit(
    #     agent_factory.run_retrieve,
    #    "file_management/change_monitor_agent",
    #     "Please change content in '/Users/manchester/Documents/rag/rag_source/physics/quantum.txt' to old_quan in physics",
    #     retric_dic,
    #     redis_client,
    #     '/Users/manchester/Documents/data/rag_database',
    #     True,
    #     '/Users/manchester/Documents/rag/rag_source/change_data/quantum.txt',
    #     '/Users/manchester/Documents/rag/rag_source/physics'
    # )
    # agent_tasks = [change_monitoragent]
    # for r in as_completed(agent_tasks):
    #     _res = r.result()



    # translation_agent = agent_thread_pool.submit(
    #     agent_factory.run_retrieve,
    #    "file_management/translation_agent",
    #     "Please translate file named quantum to Chinese",
    #     retric_dic,
    #     redis_client,
    #     '/Users/manchester/Documents/data/rag_database',
    #     True,
    # )
    # agent_tasks = [translation_agent]
    # for r in as_completed(agent_tasks):
    #         _res = r.result()

    # rollback_agent = agent_thread_pool.submit(
    #     agent_factory.run_retrieve,
    #    "file_management/rollback_agent",
    #     # "Please rollback file named quantum to the version in 2024-01-03",
    #     "Please rollback file named quantum 5 versions",
    #     retric_dic,
    #     redis_client,
    #     '/Users/manchester/Documents/data/rag_database',
    #     True,
    # )
    # agent_tasks = [rollback_agent]
    # for r in as_completed(agent_tasks):
    #     _res = r.result()


    # agent_id = submitAgent(
    #     agent_name="file_management/retrieve_summary_agent",
    #     task_input="Please search for the 2 papers with the highest correlation with large model uncertainty."
    # )
    #     agent_factory.run_retrieve,
    #    "file_management/link_agent",
    #     "/Users/manchester/Documents/rag/rag_source/rag_paper/AIOS.pdf ",
    #     retric_dic,
    #     redis_client,
    #     '/Users/manchester/Documents/data/rag_database',
    #     True,
    # )
    awaitAgentExecution(agent_id)
    
    # agent_ids = []
    # for agent_name, task_input in agent_tasks:
    #     agent_id = submitAgent(
    #         agent_name=agent_name,
    #         task_input=task_input
    #     )
    #     agent_ids.append(agent_id)
    
    # for agent_id in agent_ids:
    #     awaitAgentExecution(agent_id)

    stopScheduler()

    clean_cache(root_directory="./")


if __name__ == "__main__":
    main()
