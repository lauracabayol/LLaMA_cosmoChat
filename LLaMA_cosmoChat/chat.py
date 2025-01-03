# Standard library imports
import io
import json
import logging
import os
import platform
import re
import subprocess
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import torch
import huggingface_hub
import transformers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from LLaMA_cosmoChat.exceptions import QueryValidationError, SQLExtractionError

@dataclass
class ChatCosmoHub:
    """
    Initialize the ChatCosmoHub class with configurations for model loading and environment setup.

    Parameters:
    - cache_dir (str): Directory for caching model files.
    - base_model (str): Pre-trained model name from Hugging Face.
    - load_in_4bit (bool): Flag to enable 4-bit model quantization.
    - load_in_8bit (bool): Flag to enable 8-bit model quantization.
    """

    cache_dir: Optional[str] = None
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    def __post_init__(self) -> None:
        # Determine the root directory of the repository using Git
        self.root_repo = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        ).stdout.strip()

        # Initialize environment and configurations
        self._Huggingface_login()
        self._print_system_specs()
        self.tables_available = ["quaia_v1", "glade"]

        # Load schema information from a JSON file located in the repository
        with open(Path(self.root_repo) / "data" / "schema_info.json", "r") as file:
            self.schema_info = json.load(file)

        # Load the pre-trained model and tokenizer from Hugging Face
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            #load_in_8bit_fp32_cpu_offload=True,
            #offload_folder="offload_folder",
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model,
            cache_dir=self.cache_dir,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Create a text-generation pipeline using the loaded model and tokenizer
        self.pipeline_model = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )

    def _Huggingface_login(self) -> None:
        """
        Log in to Hugging Face using an authentication token from environment variables.
        """
        import logging

        logger = logging.getLogger(__name__)
        huggingface_hub.login(token=os.environ["KERNEL_HUGGINGFACE"])
        logger.info("Successfully logged in to Hugging Face")
        return None

    def _print_system_specs(self) -> None:
        """
        Print specifications of the system including CUDA devices and CPU information.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check if CUDA is available
        is_cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {is_cuda_available}")

        # Get the number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {num_cuda_devices}")

        if is_cuda_available:
            for i in range(num_cuda_devices):
                # Get CUDA device properties
                device = torch.device("cuda", i)
                logger.info(f"--- CUDA Device {i} ---")
                logger.info(f"Name: {torch.cuda.get_device_name(i)}")
                logger.info(
                    f"Compute Capability: {torch.cuda.get_device_capability(i)}"
                )
                logger.info(
                    f"Total Memory: {torch.cuda.get_device_properties(i).total_memory} bytes"
                )

        # Get CPU information
        logger.info("--- CPU Information ---")
        logger.info(f"Processor: {platform.processor()}")
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python Version: {platform.python_version()}")
        return None

    def _get_relevant_tables(self, query: str) -> List[str]:
        """
        Identify relevant tables from the schema based on the query.

        Parameters:
        - query (str): The user query to match against schema keywords.

        Returns:
        - List of relevant table schemas.
        """
        relevant_tables = []
        for table_name, schema in self.schema_info.items():
            if any(keyword in query.lower() for keyword in schema.lower().split()):
                relevant_tables.append(schema)
        return relevant_tables

    def _generate_prompt(self, query: str) -> List[Dict[str, str]]:
        """
        Generate a prompt for the language model based on the user query and schema information.

        Parameters:
        - query (str): The user query to be processed.

        Returns:
        - List of messages formatted for the model, including system and user content.
        """
        relevant_tables = self._get_relevant_tables(query)

        # Join the relevant table schemas into a single string
        table_schemas = "\n".join(relevant_tables)

        # Define the base system instruction
        system_message = f"""
        You are a specialized text-to-SQL assistant focused on astronomical data queries.
        Your task is to convert natural language questions into precise SQL queries for accessing astronomical databases.
        Given a question and table schemas, you will:
        1. Generate a SQL query that includes 'cosmohub.' prefix for all table names
        2. Select only the specific columns needed to answer the question
        3. Follow standard SQL best practices for readability and performance
        4. Handle astronomical data types and units appropriately
        
        Format your response as:
        SQL query: <your SQL query here>
        
        Important rules:
        - Never use SELECT * - always specify required columns
        - Include proper table aliases and joins when needed
        - Use appropriate aggregation functions for statistical queries
        - Apply filters (WHERE clause) to limit results when relevant
        - Format numbers and dates according to astronomical conventions
    
    
        The database schema is as follows:
        {table_schemas}
        """

        # Combine with the user query
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

        return messages

    def _clean_SQLstring(self, string: str) -> str:
        """
        Clean and format the SQL query string.

        Parameters:
        - string (str): The SQL query string to be cleaned.

        Returns:
        - Cleaned SQL query string.
        """
        string = re.sub(r"```", "", string).strip()
        string = string.replace("\n", " ")
        string = " ".join(string.split())
        # Remove trailing semicolon
        if string.endswith(";"):
            string = string[:-1].strip()
        return string

    def _extract_sql(self, content: str) -> str:
        """
        Extract SQL query  from the generated content.

        Parameters:
        - content (str): The content containing both SQL 

        Returns:
        - Tuple containing the cleaned SQL query

        Raises:
        - SQLExtractionError: If SQL query keyword is not found in content
        """
        # Define the keywords in lowercase
        sql_keyword = "sql query:"

        # Convert content to lowercase for case-insensitive matching of keywords
        content_lower = content.lower()

        # Find positions of the keywords in the lowercase content
        sql_pos = content_lower.find(sql_keyword)

        if sql_pos == -1:
            raise SQLExtractionError("Could not find SQL query in generated content")

        sql_query = content.split(content[sql_pos : sql_pos + len(sql_keyword)])[
            1
        ].strip()
        sql_part = self._clean_SQLstring(sql_query)

        return sql_query

    def query_LLaMA(self, query: str) -> Tuple[str, str]:
        """
        Query the LLaMA model with a user query to generate SQL.

        Parameters:
        - query (str): The user query to be processed.

        Returns:
        - Tuple containing the SQL query.

        Raises:
        - QueryValidationError: If the query doesn't reference any available tables
        """
        if not any(table in query for table in self.tables_available):
            raise QueryValidationError(
                f"Your query does not reference any of the relevant tables: {', '.join(self.tables_available)}"
            )

        messages = self._generate_prompt(query)

        # Prepare the prompt for the model
        prompt = self.pipeline_model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipeline_model.tokenizer.eos_token_id,
            self.pipeline_model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # Generate the output from the model
        outputs = self.pipeline_model(
            prompt,
            max_new_tokens=512,  # Increased token limit to accommodate both SQL 
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5,
            top_p=1,
        )
        content = outputs[0]["generated_text"][len(prompt) :]
        sql_part = self._clean_SQLstring(content)

        return sql_part
