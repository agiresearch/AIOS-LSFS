from ...react_agent import ReactAgent

from ...base_agent import BaseAgent

import os
import time

from aios.hooks.request import send_request

from pyopenagi.utils.chat_template import Query

import json


class AcademicAgent(BaseAgent):
    def __init__(self, agent_name, task_input, log_mode: str):
        ReactAgent.__init__(self, agent_name, task_input, log_mode)
        self.workflow_mode = "manual"
        # self.workflow_mode = "automatic"

    def check_path(self, tool_calls):
        script_path = os.path.abspath(__file__)
        save_dir = os.path.join(
            os.path.dirname(script_path), "output"
        )  # modify the customized output path for saving outputs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for tool_call in tool_calls:
            try:
                for k in tool_call["parameters"]:
                    if "path" in k:
                        path = tool_call["parameters"][k]
                        if not path.startswith(save_dir):
                            tool_call["parameters"][k] = os.path.join(
                                save_dir, os.path.basename(path)
                            )
            except Exception:
                continue
        return tool_calls

    def manual_workflow(self):
        workflow = [
            {
                "action_type": "message_llm",
                "action": "Search for relevant papers",
                "tool_use": ["arxiv"],
            },
            {
                "action_type": "message_llm",
                "action": "Provide responses based on the user's query",
                "tool_use": [],
            }
        ]
        return workflow

    def call_tools(self, tool_calls):
        # self.logger.log(f"***** It starts to call external tools *****\n", level="info")
        success = True
        actions = []
        observations = []

        # print(tool_calls)
        for tool_call in tool_calls:
            # print(tool_call)
            function_name = tool_call["name"]
            function_to_call = self.tool_list[function_name]
            function_params = tool_call["parameters"]

            try:
                function_response = function_to_call.run(function_params)
                actions.append(f"I will call the {function_name} with the params as {function_params}")
                observations.append(f"The output of calling the {function_name} tool is: {function_response}")

            except Exception:
                actions.append("I fail to call any tools.")
                observations.append(f"The tool parameter {function_params} is invalid.")
                success = False

        return actions, observations, success
    
    def run(self):
        super().run()
        self.build_system_instruction()

        task_input = self.task_input

        self.messages.append({"role": "user", "content": task_input})
        self.logger.log(f"{task_input}\n", level="info")

        workflow = None

        if self.workflow_mode == "automatic":
            workflow = self.automatic_workflow()
        else:
            assert self.workflow_mode == "manual"
            workflow = self.manual_workflow()

        self.messages = self.messages[:1]  # clear long context

        self.messages.append(
            {
                "role": "user",
                "content": f"[Thinking]: The workflow generated for the problem is {json.dumps(workflow)}. Follow the workflow to solve the problem step by step. ",
            }
        )

        # if workflow:
        #     self.logger.log(f"Generated workflow is: {workflow}\n", level="info")
        # else:
        #     self.logger.log(
        #         "Fail to generate a valid workflow. Invalid JSON?\n", level="info"
        #     )
        
        try:
            if workflow:
                final_result = ""

                for i, step in enumerate(workflow):
                    action_type = step["action_type"]
                    action = step["action"]
                    tool_use = step["tool_use"]

                    prompt = f"At step {i + 1}, you need to: {action}. "
                    self.messages.append({"role": "user", "content": prompt})
                    
                    if tool_use:
                        selected_tools = self.pre_select_tools(tool_use)

                    else:
                        selected_tools = None

                    (
                        response,
                        start_times,
                        end_times,
                        waiting_times,
                        turnaround_times,
                    ) = send_request(
                        agent_name=self.agent_name,
                        query=Query(
                            messages=self.messages,
                            tools=selected_tools,
                            action_type=action_type,
                        )
                    )

                    if self.rounds == 0:
                        self.set_start_time(start_times[0])

                    # execute action
                    response_message = response.response_message

                    tool_calls = response.tool_calls

                    self.request_waiting_times.extend(waiting_times)
                    self.request_turnaround_times.extend(turnaround_times)

                    if tool_calls:
                        for _ in range(self.plan_max_fail_times):
                            tool_calls = self.check_path(tool_calls)
                            actions, observations, success = self.call_tools(
                                tool_calls=tool_calls
                            )

                            action_messages = "[Action]: " + ";".join(actions)
                            observation_messages = "[Observation]: " + ";".join(
                                observations
                            )

                            self.messages.append(
                                {
                                    "role": "assistant",
                                    "content": action_messages
                                    + ". "
                                    + observation_messages,
                                }
                            )
                            if success:
                                break
                    else:
                        thinkings = response_message
                        self.messages.append(
                            {"role": "assistant", "content": thinkings}
                        )

                    if i == len(workflow) - 1:
                        final_result = self.messages[-1]

                    step_result = self.messages[-1]["content"]
                    self.logger.log(f"At step {i + 1}, {step_result}\n", level="info")

                    self.rounds += 1

                self.set_status("done")
                self.set_end_time(time=time.time())

                return {
                    "agent_name": self.agent_name,
                    "result": final_result,
                    "rounds": self.rounds,
                    "agent_waiting_time": self.start_time - self.created_time,
                    "agent_turnaround_time": self.end_time - self.created_time,
                    "request_waiting_times": self.request_waiting_times,
                    "request_turnaround_times": self.request_turnaround_times,
                }

            else:
                return {
                    "agent_name": self.agent_name,
                    "result": "Failed to generate a valid workflow in the given times.",
                    "rounds": self.rounds,
                    "agent_waiting_time": None,
                    "agent_turnaround_time": None,
                    "request_waiting_times": self.request_waiting_times,
                    "request_turnaround_times": self.request_turnaround_times,
                }
        except Exception as e:
            print(e)
            return {}
