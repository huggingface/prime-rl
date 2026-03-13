import asyncio
import os
import subprocess
import sys
import tempfile

import verifiers as vf
from datasets import load_dataset


def load_environment(max_turns: int = 5, **kwargs) -> vf.Environment:
    # 1. Load dataset
    dataset = load_dataset("nvidia/Nemotron-RL-coding-competitive_coding", split="train")

    # Filter to rows that have valid unit tests
    dataset = dataset.filter(
        lambda ex: (
            isinstance(ex.get("verifier_metadata"), dict)
            and isinstance(ex["verifier_metadata"].get("unit_tests"), dict)
            and len(ex["verifier_metadata"]["unit_tests"].get("inputs", [])) > 0
            and len(ex["verifier_metadata"]["unit_tests"]["inputs"])
            == len(ex["verifier_metadata"]["unit_tests"].get("outputs", []))
        )
    )

    def format_example(ex):
        prompt = ex["responses_create_params"]["input"][0]["content"]
        parts = prompt.split("```python\n# Your code here\n```")
        if len(parts) > 1:
            prompt = parts[-1].strip()
        else:
            prompt = prompt.strip()

        ex["prompt"] = (
            f"You are an expert Python programmer.\n\n{prompt}\n\nPlease write a python code to solve the problem and use the execute_python_code tool to test it."
        )
        # Store unit_tests in "info" so State auto-forwards it
        ex["info"] = ex["verifier_metadata"]["unit_tests"]
        return ex

    dataset = dataset.map(format_example)

    # 2. Define tools
    async def execute_python_code(code: str, state: vf.State) -> str:
        """Execute python code to test against the hidden test cases. Provide the complete python code.

        args:
            code (str): The complete python code to execute.
        """
        unit_tests = state["info"]
        inputs = unit_tests["inputs"]
        outputs = unit_tests["outputs"]

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            for idx, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, temp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=test_input.encode("utf-8")), timeout=3.0
                    )

                    if proc.returncode != 0:
                        feedback = stderr.decode("utf-8")[-2000:]
                        return f"Execution failed on test case {idx + 1} with error:\n{feedback}\nPlease fix the code and try again."

                    def normalize_output(out: str) -> str:
                        return "\n".join(line.rstrip() for line in out.replace("\r\n", "\n").split("\n")).strip()

                    actual_output = normalize_output(stdout.decode("utf-8"))
                    expected_output_clean = normalize_output(expected_output)

                    if actual_output != expected_output_clean:
                        return f"Test case {idx + 1} failed.\nInput:\n{test_input}\nExpected Output:\n{expected_output_clean}\nActual Output:\n{actual_output}\nPlease fix the code and try again."

                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return f"Execution timeout on test case {idx + 1}."

            # All tests passed
            state["tests_passed"] = True
            state["final_env_response"] = "Tests passed."
            return "Tests passed."
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 3. Define stateful env class that injects state
    class NemotronEnv(vf.StatefulToolEnv):
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
    vf_env = NemotronEnv(
        dataset=dataset,
        rubric=rubric,
        tools=[],  # We add tools later to use args_to_skip
        max_turns=max_turns,
        **kwargs,
    )

    vf_env.add_tool(execute_python_code, args_to_skip=["state"])

    return vf_env
