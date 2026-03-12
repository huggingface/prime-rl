import asyncio
import os
import subprocess
import sys
import tempfile
from typing import Any, cast

from datasets import load_dataset
import verifiers as vf


def load_environment(max_turns: int = 5, **kwargs) -> vf.Environment:
    # 1. Load dataset
    dataset = load_dataset("google-research-datasets/mbpp", split="train+test")

    def format_example(ex):
        prompt = ex.get("text", "")
        ex["prompt"] = (
            f"You are an expert Python programmer.\n\n{prompt}\n\nPlease write a python code to solve the problem and use the execute_python_code tool to test it."
        )
        return ex

    dataset = dataset.map(format_example)

    # 2. Define tools
    async def execute_python_code(code: str, state: vf.State) -> str:
        """Execute python code to test against the hidden test cases. Provide the complete python code.

        args:
            code (str): The complete python code to execute.
        """
        test_list = state.get("input", {}).get("test_list", [])

        full_code = code + "\n\n" + "\n".join(test_list)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            # We use asyncio.create_subprocess_exec to avoid blocking the event loop
            proc = await asyncio.create_subprocess_exec(
                sys.executable, temp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                if proc.returncode == 0:
                    state["tests_passed"] = True
                    state["final_env_response"] = "Tests passed."
                    return "Tests passed."
                else:
                    feedback = stderr.decode("utf-8")[-2000:]
                    return f"Execution failed with error:\n{feedback}\nPlease fix the code and try again."
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return "Execution timeout."
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 3. Define stateful env class that injects state
    class MBPPEnv(vf.StatefulToolEnv):
        def update_tool_args(
            self,
            tool_name: str,
            tool_args: dict,
            messages: vf.Messages,
            state: vf.State,
            **kwargs,
        ) -> dict:
            updated_args = dict(tool_args)
            if tool_name == "execute_python_code":
                updated_args["state"] = state
            return updated_args

    # 4. Define rubric
    rubric = vf.Rubric()

    async def tests_passed_reward(state: vf.State) -> float:
        return 1.0 if state.get("tests_passed") else 0.0

    rubric.add_reward_func(tests_passed_reward, weight=1.0)

    # 5. Instantiate env
    vf_env = MBPPEnv(
        dataset=dataset,
        rubric=rubric,
        tools=[],  # We add tools later to use args_to_skip
        max_turns=max_turns,
        **kwargs,
    )

    vf_env.add_tool(execute_python_code, args_to_skip=["state"])

    return vf_env
