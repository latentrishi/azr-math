def compute_solver_reward(success: bool) -> float:
    """
    Computes the reward for the solver.
    Returns 1.0 if success, 0.0 otherwise.
    """
    if success:
        reward = 1.0
    else:
        reward = 0.0
    return reward

def compute_proposer_reward(question_specific_solve_rate: float) -> float:
    """
    Computes the reward for the proposer.
    Encourages generation of tasks that are learnable (not too easy, not too hard).
    Based on the solver's average success rate (question_specific_solve_rate) on the proposed task.
    r_propose = { 0 if r_solve_bar == 0 or r_solve_bar == 1, else 1 - r_solve_bar }
    """
    if question_specific_solve_rate == 0.0 or question_specific_solve_rate == 1.0:
        reward = 0.0
    else:
        reward = 1.0 - question_specific_solve_rate
    return reward
