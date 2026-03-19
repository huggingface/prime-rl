"""
Immediate EOS sanity-check environment.

The model is rewarded for producing shorter completions.
Reward = -len(completion_text). The optimal policy is to emit
as little text as possible (ideally EOS immediately).

This mirrors the TRL async_grpo_immediate_eos.py test:
within a handful of steps, average completion length should
drop and reward should climb toward 0.
"""

import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_dict(
        {
            "question": [
                "Hello",
                "Test prompt",
                "What is 1+1?",
                "Say something",
                "Hi there",
            ]
            * 20,
        }
    )

    def negative_length_reward(completion, **kwargs) -> float:
        """Reward = -len(completion_text). Optimal: produce empty/minimal output."""
        text = ""
        for msg in completion:
            if msg.get("role") == "assistant":
                text += msg.get("content", "") or ""
        return float(-len(text))

    rubric = vf.Rubric(funcs=[negative_length_reward], weights=[1.0])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="Respond to the user.",
        rubric=rubric,
        **kwargs,
    )
