# MIT License

# Copyright (c) 2025 Nick Doiron

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Source dataset: https://huggingface.co/datasets/monsoon-nlp/genetic-counselor-multiple-choice
Questions combined from three Quizlets.

In order to have comparable results to ours, please do not forget to run with --use-chat-template
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_fn_gc_eval_task(line, task_name: str = None):
    query_template = """Question: {question}\n
    Suggested answers:
    A. {choice_a}
    B. {choice_b}
    C. {choice_c}
    D. {choice_d}

    Final answer:"""

    query = query_template.format(
        question=line["question"],
        choice_a=line["optionA"],
        choice_b=line["optionB"],
        choice_c=line["optionC"],
        choice_d=line["optionD"],
    )

    choices = ["A", "B", "C", "D"]
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=choices.index(line["letter_answer"]),
    )


gc_eval_task = LightevalTaskConfig(
    name="genetic-counselor-multiple-choice",
    prompt_function=prompt_fn_gc_eval_task,
    suite=["community"],
    hf_repo="monsoon-nlp/genetic-counselor-multiple-choice",
    hf_subset="default",
    hf_avail_splits=["evaluation"],
    evaluation_splits=["evaluation"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[Metrics.loglikelihood_acc],
    version=0,
)


# STORE YOUR EVALS
TASKS_TABLE = [gc_eval_task]
