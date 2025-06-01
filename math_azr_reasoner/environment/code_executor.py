import subprocess
import tempfile
import os
import shutil
import sys
import logging
import venv # For creating virtual environments
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class CodeExecutor:
    def __init__(self, 
                 timeout_seconds: int = 10, 
                 required_packages: Optional[List[str]] = None,
                 float_tolerance: float = 1e-9):
        self.timeout_seconds = timeout_seconds
        self.required_packages = required_packages or []
        self.float_tolerance = float_tolerance # For comparing float results, if needed elsewhere
        self.venv_path: Optional[str] = None
        self.python_executable: Optional[str] = None

        if os.name == 'nt':  # Windows
            self.bin_dir = 'Scripts'
        else:  # Unix-like systems
            self.bin_dir = 'bin'
        
        self._setup_venv()

    def _setup_venv(self) -> None:
        """Set up a virtual environment and install required packages."""
        try:
            # Create a temporary directory for the venv
            self.venv_path = tempfile.mkdtemp(prefix="code_executor_venv_")
            logger.info(f"Creating virtual environment in: {self.venv_path}")
            venv.create(self.venv_path, with_pip=True)
            
            self.python_executable = os.path.join(self.venv_path, self.bin_dir, 'python')

            if not os.path.exists(self.python_executable):
                # Fallback for some venv structures or Windows where python.exe might be directly in venv_path
                self.python_executable = os.path.join(self.venv_path, 'python.exe' if os.name == 'nt' else 'python')
                if not os.path.exists(self.python_executable):
                     raise FileNotFoundError(f"Python executable not found in venv: {self.venv_path}")

            if self.required_packages:
                logger.info(f"Installing packages: {self.required_packages} into {self.venv_path}")
                pip_executable = os.path.join(self.venv_path, self.bin_dir, 'pip')
                subprocess.check_call([pip_executable, 'install'] + self.required_packages)
            logger.info("Virtual environment setup complete.")
        except Exception as e:
            logger.error(f"Failed to set up virtual environment: {e}")
            self._cleanup_venv() # Clean up if setup fails
            raise # Re-raise the exception to signal failure to the caller

    def _cleanup_venv(self) -> None:
        """Clean up the virtual environment if it exists."""
        if self.venv_path and os.path.exists(self.venv_path):
            logger.info(f"Cleaning up virtual environment: {self.venv_path}")
            try:
                shutil.rmtree(self.venv_path)
                self.venv_path = None
                self.python_executable = None
            except Exception as e:
                logger.error(f"Error cleaning up venv {self.venv_path}: {e}")
        
    def __del__(self):
        """Ensure cleanup when the object is deleted."""
        self._cleanup_venv()

    def execute_python_code(self, code_string: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Executes a string of Python code in a restricted environment.
        Returns a tuple (stdout, stderr).
        stdout is None if execution fails or times out.
        stderr contains error message if any.
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp_file:
            tmp_file_name = tmp_file.name
            tmp_file.write(code_string)
        
        try:
            # Execute the script using subprocess
            if not self.python_executable or not os.path.exists(self.python_executable):
                return None, "Python executable in virtual environment is not available. Setup might have failed."
            
            process = subprocess.run(
                [self.python_executable, tmp_file_name],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False # Don't raise CalledProcessError for non-zero exit codes
            )
            
            stdout = process.stdout.strip() if process.stdout else ""
            stderr = process.stderr.strip() if process.stderr else ""

            if process.returncode != 0:
                # Include non-empty stdout for cases where code prints before erroring
                return stdout if stdout else None, f"Execution failed with return code {process.returncode}. Error: {stderr}"
            
            return stdout, None # Success
            
        except subprocess.TimeoutExpired:
            return None, "Execution timed out."
        except Exception as e:
            return None, f"An unexpected error occurred during execution: {str(e)}"
        finally:
            if os.path.exists(tmp_file_name):
                os.remove(tmp_file_name)
