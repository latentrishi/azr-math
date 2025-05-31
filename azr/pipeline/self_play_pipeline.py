import torch
from typing import List, Tuple
import subprocess
import tempfile

from absolute_zero_reasoner.data_construction.buffer import RollingBuffer
from absolute_zero_reasoner.data_construction.prompts import (
    get_math_question_proposer_prompt,
    get_math_solver_prompt,
)
from absolute_zero_reasoner.rewards import solver_reward, proposer_reward
from absolute_zero_reasoner.rewards.math_utils import extract_answer, grade_answer_sympy
from absolute_zero_reasoner.trainer.ppo.reason_rl_ray_trainer import ReasonRLRayPPOTrainer


class SelfPlayPipeline:
    """
    Orchestrates one self-play episode:
      proposer → oracle → solver → verify → buffer → PPO push.
    """
    def __init__(
        self,
        model,               # LLM for proposer & solver
        oracle_model,        # strong LLM oracle
        tokenizer,
        trainer: ReasonRLRayPPOTrainer,
        buffer_size: int = 500,
        init_examples: List[Tuple[str,str,bool]] = None
    ):
        self.model = model
        self.oracle = oracle_model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.buffer = RollingBuffer(buffer_size)
        if init_examples:
            for q,a,s in init_examples:
                self.buffer.add(q, a, s)

    def run_episode(self) -> bool:
        # 1) Propose
        recent = list(self.buffer.get_all())[-5:]
        ctx = [(q, s) for q, _, s in recent]
        prompt = get_math_question_proposer_prompt(ctx)
        q_tok = self.tokenizer(prompt, return_tensors="pt")
        q_out = self.model.generate(**q_tok)
        question = self.tokenizer.decode(q_out[0], skip_special_tokens=True)

        # 2) Oracle: generate Python code and execute for ground-truth
        code_prompt = f"## Task: Write Python code to solve the following math problem\nProblem: {question}\nPlease provide a Python script that prints only the final answer as plain text."
        code_tok = self.tokenizer(code_prompt, return_tensors="pt")
        c_out = self.oracle.generate(**code_tok)
        code = self.tokenizer.decode(c_out[0], skip_special_tokens=True)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmpf:
            tmpf.write(code)
            tmpf.flush()
            try:
                result = subprocess.run(["python3", tmpf.name], capture_output=True, text=True, timeout=30)
                gt = result.stdout.strip()
            except Exception:
                gt = None

        # 3) Model solves
        solver_tok = self.tokenizer(
            get_math_solver_prompt(question), return_tensors="pt"
        )
        s_out = self.model.generate(**solver_tok)
        solver_sol = self.tokenizer.decode(s_out[0], skip_special_tokens=True)
        pred = extract_answer(solver_sol)

        # 4) Grade & buffer
        success = grade_answer_sympy(pred, gt)
        # store full solver response
        self.buffer.add(question, solver_sol, success)

        # 5) Rewards
        r_s = solver_reward(success)
        # per-question solve rate
        hist = [s for q_i, _, s in self.buffer.get_all() if q_i == question]
        q_rate = sum(hist) / len(hist) if hist else 0.0
        r_p = proposer_reward(q_rate)

        # 6) Format datapoints & push to trainer
        # solver datapoint
        solver_dp = {
            "input_ids": solver_tok["input_ids"],
            "attention_mask": solver_tok["attention_mask"],
            "rewards": torch.tensor([r_s] * solver_tok["input_ids"].size(-1)),
        }
        self.trainer.push(solver_dp)
        # proposer datapoint
        prop_dp = {
            "input_ids": q_tok["input_ids"],
            "attention_mask": q_tok["attention_mask"],
            "rewards": torch.tensor([r_p] * q_tok["input_ids"].size(-1)),
        }
        self.trainer.push(prop_dp)
        return success

    def train(self, episodes: int, log_every: int = 10):
        for ep in range(1, episodes + 1):
            succ = self.run_episode()
            if ep % self.trainer.config.batch_size == 0:
                self.trainer.fit()
            if ep % log_every == 0:
                print(f"Ep {ep}: succ={succ}, buf_SR={self.buffer.success_rate():.2f}")