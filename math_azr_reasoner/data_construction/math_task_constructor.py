import random
from typing import Dict, List, Optional, Tuple

class MathTaskBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size # Currently informational
        self.question_stats: Dict[str, Dict[str, int]] = {}
        self.total_recorded_attempts: int = 0
        self.total_recorded_successes: int = 0

    def record_solver_attempt(self, question: str, success: bool):
        """Records the outcome of a solver's attempt on a given question."""
        if question not in self.question_stats:
            self.question_stats[question] = {"attempts": 0, "successes": 0}
        
        self.question_stats[question]["attempts"] += 1
        self.total_recorded_attempts += 1
        if success:
            self.question_stats[question]["successes"] += 1
            self.total_recorded_successes += 1

    def get_question_success_rate(self, question: str) -> Optional[float]:
        """
        Calculates the success rate for a specific question.
        Returns None if the question has no recorded attempts.
        """
        stats = self.question_stats.get(question)
        if stats and stats["attempts"] > 0:
            return stats["successes"] / stats["attempts"]
        return None

    def get_overall_success_rate(self) -> Optional[float]:
        """
        Calculates the overall success rate across all questions.
        Returns None if there are no recorded attempts.
        """
        if self.total_recorded_attempts > 0:
            return self.total_recorded_successes / self.total_recorded_attempts
        return None

    def sample_questions_for_proposer_context(self, n_samples: int) -> List[Tuple[str, float]]:
        """
        Samples unique questions and their success rates for the proposer's context.
        If a question has no success rate (e.g., no attempts), it defaults to 0.5 for context.
        """
        if not self.question_stats:
            return []

        available_questions = list(self.question_stats.keys())
        num_to_sample = min(n_samples, len(available_questions))
        sampled_questions = random.sample(available_questions, num_to_sample)

        context = []
        for q_text in sampled_questions:
            rate = self.get_question_success_rate(q_text)
            # Default to 0.5 if rate is None (e.g. new question, though unlikely if sampled from keys with stats)
            # or if we want to provide a neutral prior for questions with no attempts yet.
            context.append((q_text, rate if rate is not None else 0.5))
        return context

    def sample_questions_for_solver(self, n_samples: int) -> List[str]:
        """
        Samples unique question strings for the solver to attempt.
        """
        if not self.question_stats:
            return []
        
        available_questions = list(self.question_stats.keys())
        num_to_sample = min(n_samples, len(available_questions))
        return random.sample(available_questions, num_to_sample)
    
    def __len__(self) -> int:
        """Returns the number of unique questions for which stats are stored."""
        return len(self.question_stats)
