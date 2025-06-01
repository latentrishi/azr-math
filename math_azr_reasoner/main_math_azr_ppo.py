import yaml
import os
from dotenv import load_dotenv

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer 

import logging 
from .data_construction.math_task_constructor import MathTaskBuffer
from .data_construction import prompts 
from .environment.oracle import OracleClient 
import json
from typing import Optional 
from .environment.code_executor import CodeExecutor
from .environment.solver import SolverClient
from .environment.proposer import ProposerClient 
from .rewards.math_reward_functions import compute_solver_reward, compute_proposer_reward
from .rewards.math_evaluator import MathEvaluator
from .trainer.ppo.math_azr_ray_trainer import MathAZRRayPPOTrainer 
from .utils.tracking_math import WandBTracker


def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    load_dotenv()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ppo_config_path = os.path.join(project_root, args.ppo_config)
    ppo_config = load_config(ppo_config_path)

    logging.basicConfig(level=ppo_config.get("logging", {}).get("level", "INFO"), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Configurations loaded.")

    if ppo_config.get("wandb", {}).get("track", False):
        tracker = WandBTracker(
            project_name=ppo_config["wandb"]["project_name"],
            run_name=ppo_config["wandb"].get("run_name"),
            config={**ppo_config} # Only ppo_config remains
        )
        logger.info(f"WandB tracking enabled for project '{ppo_config['wandb']['project_name']}'.")
    else:
        tracker = None
        logger.info("WandB tracking disabled.")

    model_path = ppo_config['actor_rollout_ref']['model']['path']
    model_device = ppo_config['actor_rollout_ref']['model'].get('device', 'cpu')
    logger.info(f"Loading model and tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(model_device)
        logger.info(f"Model {model_path} loaded to {model_device}.")
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer from {model_path}: {e}", exc_info=True)
        raise

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables. OracleClient may not work.")
    oracle_model_name = ppo_config.get('azr', {}).get('oracle_model_name')
    oracle_client = OracleClient(api_key=openai_api_key, model_name=oracle_model_name)
    logger.info(f"Oracle Client initialized with model: {oracle_client.model_name}")

    proposer_gen_params = ppo_config.get('azr', {}).get('proposer_generation_params')
    proposer_client = ProposerClient(model=model, tokenizer=tokenizer, generation_params=proposer_gen_params)
    logger.info(f"Proposer Client initialized with model: {model_path} and generation params: {proposer_gen_params}")

    solver_gen_params = ppo_config.get('azr', {}).get('solver_generation_params')
    solver_client = SolverClient(model=model, tokenizer=tokenizer, generation_params=solver_gen_params)
    logger.info(f"Solver Client initialized with model: {model_path} and generation params: {solver_gen_params}")
    
    code_executor = CodeExecutor(
        timeout_seconds=ppo_config.get("code_execution_timeout", 10),
        required_packages=['sympy'] 
    )
    logger.info(f"Code Executor initialized with timeout: {code_executor.timeout_seconds}s and required packages: {code_executor.required_packages}.")

    math_evaluator = MathEvaluator()
    logger.info("Math Evaluator initialized.")

    buffer = MathTaskBuffer(max_size=ppo_config["buffer"]["max_size"])
    logger.info(f"Math Task Buffer initialized with max size: {buffer.max_size}.")

    seed_dataset_config = ppo_config.get('azr', {}).get('seed_dataset', {})
    seed_file_path = seed_dataset_config.get('path')
    num_to_seed = seed_dataset_config.get('num_questions_to_seed')

    if seed_file_path and num_to_seed is not None and num_to_seed > 0:
        # Assuming seed_file_path is relative to project root if not absolute
        if not os.path.isabs(seed_file_path):
            seed_file_path = os.path.join(project_root, seed_file_path)
        
        if os.path.exists(seed_file_path):
            logger.info(f"Seeding buffer from {seed_file_path}, up to {num_to_seed} questions.")
            try:
                initial_attempts = seed_dataset_config.get('initial_attempts', 1)
                initial_successes = seed_dataset_config.get('initial_successes', 0)
                seed_buffer_from_file(buffer, seed_file_path, initial_attempts, initial_successes, num_to_seed)
                logger.info(f"Buffer seeding attempt from {seed_file_path} complete. Buffer size: {len(buffer)} (Targeted {num_to_seed} questions with {initial_attempts} attempts, {initial_successes} successes each). ")
            except Exception as e:
                logger.error(f"Error seeding buffer from {seed_file_path}: {e}", exc_info=True)
            # expected file format changed. For now, assuming it can take num_to_seed.
            # We'll pass num_to_seed to the function if it accepts it, or handle it in the loop inside.
            # For now, let's assume seed_buffer_from_file handles the number of questions to seed.
            # If not, the function itself needs an update or we do a partial load here.
            # Let's modify the call to seed_buffer_from_file to include num_to_seed if possible, or log a warning.
            # For now, we'll assume the function can take it or a similar mechanism.
            # This part might need further refinement based on seed_buffer_from_file's exact capabilities.
        else:
            logger.warning(f"Seed file {seed_file_path} not found. Buffer will not be seeded from file.")
    else:
        logger.info("Buffer not seeded from file (no path or num_questions_to_seed <= 0 in config).")

    # Initialize PPO Trainer
    ppo_trainer = MathAZRRayPPOTrainer(
        config=ppo_config,
        model=model, 
        tokenizer=tokenizer
    )
    logger.info("PPO Trainer initialized (placeholder).")

    # Main Self-Play Loop
    logger.info("Starting Math AZR PPO Self-Play Loop...")
    num_episodes = ppo_config.get("num_episodes", 1000)

    for episode in range(num_episodes):
        logger.info(f"--- Starting Episode {episode + 1}/{num_episodes} ---")

        # a. Proposer generates a question
        #    - Construct proposer prompt (e.g., using recent buffer stats)
        #    - Model generates question text
        proposer_context_samples = ppo_config.get('proposer_context_samples', 5)
        proposer_few_shot_tuples = buffer.sample_questions_for_proposer_context(n_samples=proposer_context_samples)
        
        few_shot_context_str = prompts.PROPOSER_FEW_SHOT_CONTEXT_HEADER
        for q_text, rate in proposer_few_shot_tuples:
            few_shot_context_str += f"Question: {q_text}\nSolve Rate: {rate:.2f}\n---\n"
        
        # This is the task-specific part of the prompt for the proposer
        proposer_task_prompt = prompts.PROPOSER_TASK_PROMPT_TEMPLATE.format(few_shot_context=few_shot_context_str)
        # Full prompt construction is handled within ProposerClient
        question_text, proposer_prompt = proposer_client.get_proposer_question(context_string=few_shot_context_str)
        if not question_text:
            logger.warning("Proposer failed to generate a question. Skipping episode.")
            # Using a fallback question to prevent crashes downstream.
            question_text = "Fallback Question: What is 1+1?"
            proposer_prompt = "Fallback Proposer Prompt (used for fallback question)" # Placeholder for prompt
            logger.warning(f"Using fallback question: {question_text}")
        logger.info(f"Proposer generated question: {question_text}")
        logger.debug(f"Proposer prompt used:\n{proposer_prompt}")

        # b. Oracle generates a ground truth solution (always code as per new spec)
        #    - If code, use CodeExecutor
        # oracle_uses_code is now always True
        gt_solution_raw = oracle_client.get_ground_truth_solution(question_text) # Assuming get_ground_truth_solution uses its configured prompts
        oracle_gt_answer = None
        if gt_solution_raw: # Oracle always provides code
            logger.debug(f"Oracle generated code:\n{gt_solution_raw}")
            executed_output, execution_error = code_executor.execute_python_code(gt_solution_raw)
            if execution_error:
                logger.warning(f"Oracle code execution failed: {execution_error}. GT will be None.")
                oracle_gt_answer = None # Explicitly set to None on error
            else:
                oracle_gt_answer = executed_output # Assuming code prints final answer
                logger.info(f"Oracle code executed. GT Answer: {oracle_gt_answer}")
        else: # Oracle failed to generate any solution (code)
            logger.error("Oracle failed to generate a ground truth solution (code)." )
            # Decide how to handle this: skip episode, use a dummy, etc.
            continue # Skip this episode if no GT
        
        if oracle_gt_answer is None:
            logger.warning(f"Could not determine Oracle GT answer for '{question_text}'. Skipping episode.")
            continue

        # c. Solver attempts to solve the question (always NL as per new spec)
        #    - Construct solver prompt
        # c. Solver attempts to solve the question
        solver_response_text, solver_prompt = solver_client.get_solver_solution(question_text)
        if not solver_response_text:
            logger.warning(f"Solver failed to generate a response for question: {question_text}. Using placeholder.")
            solver_response_text = "Placeholder solver response: Solver failed to generate."
        logger.info(f"Solver generated response: {solver_response_text}")
        if solver_prompt: # Only log if a prompt was actually constructed
            logger.debug(f"Solver prompt used:\n{solver_prompt}")

        # d. Evaluate solver's answer
        solver_answer_extracted = math_evaluator.extract_answer_from_solution(solver_response_text)
        success = math_evaluator.evaluate(solver_answer_extracted, oracle_gt_answer)
        logger.info(f"Solver extracted answer: {solver_answer_extracted}, Oracle GT: {oracle_gt_answer}, Success: {success}")

        # e. Compute rewards
        #    - Solver reward (composite: considers correctness and format)
        #    - Proposer reward based on question    's historical solve rate (from buffer)

        # Check solver response format
        solver_response_format_status = math_evaluator.check_response_format(full_solver_response)

        # Base solver reward (0 for fail, 1 for success on answer content)
        base_solver_reward = compute_solver_reward(success)

        # Composite Solver Reward Logic:
        # R(y_pi) = { r_role if passable, -0.5 if wrong but well-formatted, -1 if formatting errors }
        # r_role is base_solver_reward (0 or 1)
        if solver_response_format_status != MathEvaluator.FORMAT_OK:
            solver_reward = -1.0  # Formatting error
            logger.debug(f"Solver response format error: {solver_response_format_status}. Reward: -1.0")
        else: # Format is OK
            if success: # Correct answer and well-formatted
                solver_reward = base_solver_reward # Should be 1.0
                logger.debug(f"Solver response well-formatted and correct. Reward: {solver_reward}")
            else: # Incorrect answer but well-formatted
                solver_reward = -0.5
                logger.debug(f"Solver response well-formatted but incorrect. Reward: -0.5")

        question_solve_rate = buffer.get_question_success_rate(question_text)  # Needs implementation
        overall_solve_rate = buffer.get_overall_success_rate()  # Needs implementation
        # Use 0.5 as a neutral default if rates are None (e.g., new question/empty buffer)
        proposer_reward = compute_proposer_reward(
            question_solve_rate if question_solve_rate is not None else 0.5
        ) # overall_solve_rate removed from function signature
        logger.info(f"Final Solver reward: {solver_reward}, Proposer reward: {proposer_reward}")

        # f. Record solver attempt in buffer
        buffer.record_solver_attempt(question_text, success)
        logger.info(f"Solver attempt recorded. Unique questions in buffer: {len(buffer)}")

        # g. Push data to PPO trainer (tokenized prompts, responses, rewards)
        #    - This needs careful implementation of how to tokenize and format for PPO
        #    - Proposer data: (proposer_prompt_tokens, question_tokens, proposer_reward_per_token)
        #    - Solver data: (solver_prompt_tokens, solver_response_tokens, solver_reward_per_token)
        # ppo_trainer.push_experience(...) # Placeholder for proposer data
        # ppo_trainer.push_experience(...) # Placeholder for solver data
        logger.debug("Placeholder: Pushing experience to PPO trainer.")

        # h. PPO trainer performs an update step periodically
        if (episode + 1) % ppo_config.get("ppo_update_freq_episodes", 10) == 0:
            logger.info("Performing PPO training step...")
            # train_metrics = ppo_trainer.train_step()
            train_metrics = {"ppo_loss": 0.0} # Placeholder
            logger.info(f"PPO training step completed. Metrics: {train_metrics}")
            if tracker:
                tracker.log_metrics(train_metrics, step=episode + 1)
        
        # 8. Logging and Checkpoints
        if tracker:
            tracker.log_episode_data(
                episode_num=episode + 1,
                question=question_text,
                solver_response=full_solver_response,
                oracle_gt=oracle_gt_answer,
                success=success,
                proposer_reward=proposer_reward,
                solver_reward=solver_reward
            )
            tracker.log_metrics({
                "buffer/size": len(buffer),
                "buffer/overall_success_rate": buffer.get_overall_success_rate() or 0.0,
                "episode/proposer_reward_value": proposer_reward,
                "episode/solver_reward_value": solver_reward,
                "episode/success_bool": 1 if success else 0
            }, step=episode + 1)

        if (episode + 1) % ppo_config.get("checkpoint_freq_episodes", 100) == 0:
            checkpoint_path = os.path.join(project_root, ppo_config.get("checkpoint_dir", "checkpoints"), f"episode_{episode+1}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            # ppo_trainer.save_checkpoint(checkpoint_path)
            # model.save_pretrained(os.path.join(checkpoint_path, "model_proposer_solver")) # If saving HF model
            logger.info(f"Placeholder: Saved checkpoint at {checkpoint_path}")

    logger.info("Math AZR PPO Self-Play finished.")
    if tracker:
        tracker.finish()

def seed_buffer_from_file(buffer: MathTaskBuffer, file_path: str, initial_attempts: int = 1, initial_successes: int = 0, num_to_seed: Optional[int] = None):
    """Seeds the MathTaskBuffer with questions from a JSONL file, up to num_to_seed questions."""
    logger.info(f"Seeding buffer from {file_path}...")
    loaded_questions = 0
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if num_to_seed is not None and loaded_questions >= num_to_seed:
                    logger.info(f"Reached num_to_seed limit ({num_to_seed}). Stopping seed data loading.")
                    break
                try:
                    data = json.loads(line)
                    question_text = data.get("question") or data.get("text")
                    if question_text:
                        # Ensure the question is added with the specified initial attempts and successes.
                        # We call record_solver_attempt `initial_attempts` times.
                        # The first `initial_successes` of these will be True (correct), the rest False (incorrect).
                        for attempt_num in range(initial_attempts):
                            is_correct_for_this_attempt = (attempt_num < initial_successes)
                            buffer.record_solver_attempt(question_text, is_correct_for_this_attempt)
                        loaded_questions += 1
                    else:
                        logger.warning(f"Skipping line {i+1} in seed file, no 'question' or 'text' field: {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in seed file: {line.strip()}")

    except FileNotFoundError:
        logger.error(f"Seed tasks file not found: {file_path}. Buffer will not be seeded.")
    except Exception as e:
        logger.error(f"An error occurred while seeding the buffer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Math AZR PPO Self-Play Training.")
    parser.add_argument("--ppo_config", type=str, default="configs/math_azr_ppo_trainer.yaml", help="Path to the main PPO trainer YAML config file.")
    
    args = parser.parse_args()
    main(args)
