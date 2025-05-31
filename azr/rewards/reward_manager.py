from absolute_zero_reasoner.data_construction import RollingBuffer
from absolute_zero_reasoner.rewards import solver_reward, proposer_reward

buf = RollingBuffer(max_size=100)
question = proposer(...)
success = check_answer(...)
buf.add(question, success)

r_s = solver_reward(success)
rate = buf.success_rate()
r_p = proposer_reward(rate)