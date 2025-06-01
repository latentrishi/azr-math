import logging
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_azr_reasoner.data_construction.prompts import SOLVER_SYSTEM_PROMPT, SOLVER_TASK_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class SolverClient:
    def __init__(self, model: Any, tokenizer: Any, generation_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the SolverClient with a local Hugging Face model and tokenizer.

        Args:
            model: The loaded Hugging Face model (e.g., AutoModelForCausalLM).
            tokenizer: The loaded Hugging Face tokenizer (e.g., AutoTokenizer).
            generation_params: Dictionary of parameters for model.generate() 
                               (e.g., max_new_tokens, temperature, do_sample).
        """
        if model is None or tokenizer is None:
            raise ValueError("Model and Tokenizer must be provided for SolverClient.")
        
        self.model = model
        self.tokenizer = tokenizer
        self.generation_params = generation_params if generation_params else {}
        # Set default generation params if not provided
        self.generation_params.setdefault("max_new_tokens", 512)
        self.generation_params.setdefault("temperature", 0.7)
        self.generation_params.setdefault("do_sample", True)
        # Potentially add pad_token_id if not set and tokenizer has it
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            logger.info(f"Tokenizer does not have pad_token_id, setting to eos_token_id: {self.tokenizer.eos_token_id}")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            # The model's config might also need this if it's used directly by generate
            if hasattr(self.model, 'config') and self.model.config is not None:
                 self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def get_solver_solution(self, question_text: str) -> Tuple[Optional[str], Optional[str]]:
        prompt = (
            f"{SOLVER_SYSTEM_PROMPT}\n\n"
            f"{SOLVER_TASK_PROMPT_TEMPLATE.format(question_text=question_text)}"
        )

        try:
            logger.debug(f"Generating solver solution for: {question_text}")
            logger.debug(f"Using generation params: {self.generation_params}")

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Ensure pad_token_id is available for generation
            gen_kwargs = self.generation_params.copy()
            if 'pad_token_id' not in gen_kwargs and self.tokenizer.pad_token_id is not None:
                gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
            if 'eos_token_id' not in gen_kwargs and self.tokenizer.eos_token_id is not None:
                gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id

            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
            
            # Decode the output, skipping special tokens and the prompt part
            # response_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            # To only get the generated part, decode from the end of the input_ids
            prompt_input_ids_length = inputs["input_ids"].shape[1]
            generated_ids = output_sequences[0][prompt_input_ids_length:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger.debug(f"Raw generated output from Solver: {output_sequences}")
            logger.debug(f"Decoded response from Solver: {response_text}")
            return response_text.strip(), prompt
        except Exception as e:
            logger.error(f"An error occurred while generating solver solution: {e}", exc_info=True)
            return None, None
