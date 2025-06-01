import re
import sympy 
from typing import Optional

class MathEvaluator:
    # Format status constants
    FORMAT_OK = "FORMAT_OK"
    FORMAT_BAD_MISSING_THINK = "FORMAT_BAD_MISSING_THINK"
    FORMAT_BAD_MISSING_ANSWER = "FORMAT_BAD_MISSING_ANSWER"
    FORMAT_BAD_MISSING_BOTH_TAGS = "FORMAT_BAD_MISSING_BOTH_TAGS"
    FORMAT_BAD_EMPTY_ANSWER_TAG = "FORMAT_BAD_EMPTY_ANSWER_TAG" # If <answer></answer> is empty

    def __init__(self):
        pass

    def _normalize_answer(self, answer_str: str) -> str:
        """Basic normalization: lowercase, remove spaces, common LaTeX artifacts."""
        if not answer_str: return ""
        answer_str = str(answer_str).lower().strip()
        # Remove \\boxed{}
        answer_str = re.sub(r'\\boxed{(.*?)}', r'\1', answer_str)
        answer_str = answer_str.replace(" ", "")
        return answer_str

    def evaluate(self, solver_answer: str, oracle_gt_answer: str, method="strict_normalized") -> bool:
        """
        Compares the solver's answer to the oracle's ground truth.
        Methods:
        - "strict_normalized": Exact match after normalization.
        - "sympy_equivalent": (Requires sympy) Checks for mathematical equivalence.
        """
        if solver_answer is None or oracle_gt_answer is None:
            return False

        norm_solver_ans = self._normalize_answer(solver_answer)
        norm_oracle_gt = self._normalize_answer(oracle_gt_answer)

        if method == "strict_normalized":
            return norm_solver_ans == norm_oracle_gt
        elif method == "sympy_equivalent":
            try:
                # Normalize first to remove LaTeX and spaces that might interfere with sympify
                norm_solver_answer = self._normalize_answer(solver_answer)
                norm_oracle_gt_answer = self._normalize_answer(oracle_gt_answer)
                
                if norm_solver_answer == "" or norm_oracle_gt_answer == "": # Avoid sympifying empty strings
                    return norm_solver_answer == norm_oracle_gt_answer

                solver_expr = sympy.sympify(norm_solver_answer)
                oracle_expr = sympy.sympify(norm_oracle_gt_answer)
                return solver_expr.equals(oracle_expr)
            except (sympy.SympifyError, AttributeError, TypeError) as e:
                # logger.error(f"Error during sympy evaluation: {e}. Falling back to strict normalized comparison.")
                # Fallback to strict normalized if sympy fails (e.g., non-math string)
                return self._normalize_answer(solver_answer) == self._normalize_answer(oracle_gt_answer)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def extract_answer_from_solution(self, solution_text: str) -> Optional[str]:
        """
        Extracts the final answer from a solution string that is expected to be
        within <answer>\boxed{result}</answer> tags.
        Example: "blah blah <answer>\boxed{42}</answer> blah"
        """
        if not solution_text: return None
        # First, try to find content within <answer>...</answer>
        answer_tag_match = re.search(r'<answer>(.*?)</answer>', solution_text, re.DOTALL)
        if not answer_tag_match:
            # No <answer> tag found, so the required structure is missing.
            return None

        content_within_answer_tag = answer_tag_match.group(1).strip()
        
        # Now, find \\boxed{} within the content of <answer> tags
        boxed_match = re.search(r'\\boxed{(.*?)}', content_within_answer_tag, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()
        return None # Stricter: if <answer> found, \\boxed must be inside.

    def check_response_format(self, full_solution_text: str) -> str:
        """
        Checks the structural format of the solver's full response string.
        Specifically looks for non-empty <think>...</think> and <answer>...</answer> tags.
        """
        if not isinstance(full_solution_text, str):
            return self.FORMAT_BAD_MISSING_BOTH_TAGS # Or a more specific error for non-string input

        has_think_tag = bool(re.search(r"<think>(.*?)</think>", full_solution_text, re.DOTALL))
        # Check for <answer> tag and that it's not empty, e.g. <answer>  </answer>
        answer_match = re.search(r"<answer>(.*?)</answer>", full_solution_text, re.DOTALL)
        has_answer_tag = bool(answer_match)
        answer_content_is_not_empty = bool(answer_match and answer_match.group(1).strip())

        if has_think_tag and has_answer_tag and answer_content_is_not_empty:
            return self.FORMAT_OK
        elif not has_think_tag and not has_answer_tag:
            return self.FORMAT_BAD_MISSING_BOTH_TAGS
        elif not has_think_tag:
            return self.FORMAT_BAD_MISSING_THINK
        elif not has_answer_tag: # Implies think_tag is present
            return self.FORMAT_BAD_MISSING_ANSWER
        elif has_answer_tag and not answer_content_is_not_empty: # Has <answer> but it's empty
             return self.FORMAT_BAD_EMPTY_ANSWER_TAG
        else: # Should not be reached given the logic, but as a fallback
            return self.FORMAT_BAD_MISSING_BOTH_TAGS # Or some other general format error

