from typing import List, Tuple

math_proposer_prompt_template = """## Task: Generate a Math Problem
You are the Question Proposer in a curriculum-based self-play loop for mathematical reasoning.
You are provided with past examples of math questions and whether the solver succeeded on them:
{examples}

### Objectives:
- Generate a new math problem that is challenging yet solvable in under 10 reasoning steps.
- Aim for a solver success rate of approximately 50% to maintain dynamic curriculum progression.
- Cover topics such as arithmetic, algebra, geometry, combinatorics, or basic number theory.
- Avoid requiring specialized theorems or advanced jargon beyond high-school level.

### Output Format:
- Provide only the problem statement in clear, concise natural language.
- Do not include hints, numbering, or explanations.
"""

def get_math_question_proposer_prompt(examples: List[Tuple[str, bool]]) -> str:
    """
    Build a prompt for the proposer using past (question, success) pairs.
    """
    formatted = "\n".join([f"{i+1}. Q: {q} [Solved: {s}]" for i, (q, s) in enumerate(examples)])
    return math_proposer_prompt_template.format(examples=formatted)


math_solver_prompt_template = """## Task: Solve a Math Problem
You are the Math Solver in a self-play loop. Solve the provided problem step by step.

Problem:
{question}

### Instructions:
- Wrap each chain-of-thought step in <think>...</think> tags.
- After reasoning, provide the final answer in <answer>...</answer> tags.
- Show all intermediate calculations and justifications.

### Evaluation Criteria:
- Correctness: The final answer must match the ground truth.
- Clarity: Reasoning should be logically structured and easy to follow.
- Completeness: No essential steps omitted.

Example:
<think>First, I ...</think>
...
<answer>\\boxed{{42}}</answer>
"""

def get_math_solver_prompt(question: str) -> str:
    """
    Build a prompt for the solver to produce <think>...</think> and <answer>...</answer>.
    """
    return math_solver_prompt_template.format(question=question)