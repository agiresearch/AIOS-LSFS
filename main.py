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
        mount_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsfs-test")
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
    
    agent_tasks = [
        ["file_management/retrieve_agent", "Help me find 3 files whose authors are from Rutgers University"]
    ]
    
    for (name, task) in agent_tasks:
        agent_id = submitAgent(
            agent_name=name,
            task_input=task
        )
        awaitAgentExecution(agent_id)

    stopScheduler()

    clean_cache(root_directory="./")


if __name__ == "__main__":
    main()
