def solver_reward(success: bool) -> float:
    return 1.0 if success else 0.0

def proposer_reward(success_rate: float) -> float:
    if success_rate == 0.0 or success_rate == 1.0:
        return 0.0
    return 1.0 - success_rate