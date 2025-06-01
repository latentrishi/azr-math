import os
import re
import logging
from typing import Optional
from openai import OpenAI, APIError
from math_azr_reasoner.data_construction.prompts import ORACLE_CODE_SYSTEM_PROMPT, ORACLE_CODE_TASK_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class OracleClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4.1"):
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key not provided and not found in OPENAI_API_KEY environment variable.")
        
        try:
            self.client = OpenAI(api_key=resolved_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
            
        self.model_name = model_name

    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extracts Python code from a markdown-formatted string."""
        match = re.search(r"```(?:python\n)?(.*?)```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("No markdown code block found in Oracle's response. Attempting to use entire response (stripped) as code.")
        stripped_response = response_text.strip()
        if any(keyword in stripped_response for keyword in ['def ', 'import ', 'print(', 'return ']):
            # A simple check, could be more robust
            if len(stripped_response.splitlines()) > 0:
                 return stripped_response
        logger.error("Could not extract code from Oracle response: %s", response_text)
        return None

    def get_ground_truth_solution(self, question: str) -> Optional[str]:
        """
        Generates a ground truth Python code solution for a given math question.
        """
        formatted_task_prompt = ORACLE_CODE_TASK_PROMPT_TEMPLATE.format(question_text=question)
        messages = [
            {"role": "system", "content": ORACLE_CODE_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_task_prompt}
        ]

        try:
            logger.debug(f"Sending request to Oracle LLM (model: {self.model_name}) for question: {question[:100]}...")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=4096
            )
            
            response_content = completion.choices[0].message.content
            if not response_content:
                logger.warning("Oracle LLM returned an empty response.")
                return None

            logger.debug(f"Oracle LLM raw response:\n{response_content}")
            extracted_code = self._extract_code_from_response(response_content)
            
            if extracted_code:
                logger.info(f"Successfully extracted code from Oracle for question: {question[:100]}...")
            else:
                logger.warning(f"Failed to extract code from Oracle's response for question: {question[:100]}...")
            
            return extracted_code

        except APIError as e:
            logger.error(f"OpenAI API error while querying Oracle: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while querying Oracle: {e}")
            return None

