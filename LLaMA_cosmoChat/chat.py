# Import essential modules for system operations, data manipulation, and machine learning
import os
import sys
import subprocess
import json
from pyhive import hive
import pandas as pd
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Import PyTorch and Hugging Face modules for model handling and text processing
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, pipeline, BitsAndBytesConfig

import wandb
import platform
from huggingface_hub import login

class ChatCosmoHub:
    def __init__(self,
                 cache_dir=None,
                 wandb_monitoring=False,
                 base_model="meta-llama/Meta-Llama-3-8B-Instruct",
                 load_in_4bit=True,
                 load_in_8bit=False):
        """
        Initialize the ChatCosmoHub class with configurations for model loading and environment setup.

        Parameters:
        - cache_dir (str): Directory for caching model files.
        - wandb_monitoring (bool): Flag to enable Weights & Biases monitoring.
        - base_model (str): Pre-trained model name from Hugging Face.
        - load_in_4bit (bool): Flag to enable 4-bit model quantization.
        - load_in_8bit (bool): Flag to enable 8-bit model quantization.
        """

        # Determine the root directory of the repository using Git
        self.root_repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], 
                                        capture_output=True, 
                                        text=True).stdout.strip()
        # Authenticate with Hugging Face
        self._Huggingface_login()
        # Print system specifications (hardware and software)
        self._print_system_specs()
        # Define available tables for queries
        self.tables_available = ['quaia_v1', 'glade']

        # Load schema information from a JSON file located in the repository
        with open(self.root_repo + '/data/schema_info.json', 'r') as file:
            self.schema_info = json.load(file)

        # Load the pre-trained model and tokenizer from Hugging Face
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Create a text-generation pipeline using the loaded model and tokenizer
        self.pipeline_model = pipeline("text-generation",
                                       model=model,
                                       tokenizer=tokenizer,
                                       model_kwargs={
                                           "torch_dtype": torch.float16,
                                           "quantization_config": {"load_in_4bit": True},
                                           "low_cpu_mem_usage": True
                                       })

    def _Huggingface_login(self):
        """
        Log in to Hugging Face using an authentication token from environment variables.
        """
        login(token=os.environ['KERNEL_HUGGINGFACE'])
    
    def _print_system_specs(self):
        """
        Print specifications of the system including CUDA devices and CPU information.
        """
        # Check if CUDA is available
        is_cuda_available = torch.cuda.is_available()
        print("CUDA Available:", is_cuda_available)
        # Get the number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()
        print("Number of CUDA devices:", num_cuda_devices)
        if is_cuda_available:
            for i in range(num_cuda_devices):
                # Get CUDA device properties
                device = torch.device('cuda', i)
                print(f"--- CUDA Device {i} ---")
                print("Name:", torch.cuda.get_device_name(i))
                print("Compute Capability:", torch.cuda.get_device_capability(i))
                print("Total Memory:", torch.cuda.get_device_properties(i).total_memory, "bytes")
        # Get CPU information
        print("--- CPU Information ---")
        print("Processor:", platform.processor())
        print("System:", platform.system(), platform.release())
        print("Python Version:", platform.python_version())    

    def _get_relevant_tables(self, query):
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

    def _generate_prompt(self, query):
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
        You are a powerful text-to-SQL and text-to-Python-code model. 
        Your job is to answer questions about a database by providing both the SQL query needed to retrieve the data and the Python script to perform the requested tasks.
        You are given a question and context regarding one or more tables. 
        You must output both the SQL query and the Python script that accomplishes the task.
        Add cosmohub. in front of the queried table in the SQL query.
        Return only the SQL query without further explanations, starting with 'SQL query:', do not start with sql, provide ONLY the query.
        Return only the script for the plot, without the python script for querying the data, starting with 'Python script:'.
        Assume that data is stored in a pandas DataFrame called 'df'
    
        The database schema is as follows:
        {table_schemas}
        """
    
        # Combine with the user query
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    
        return messages

    def _clean_SQLstring(self, string):
        """
        Clean and format the SQL query string.

        Parameters:
        - string (str): The SQL query string to be cleaned.

        Returns:
        - Cleaned SQL query string.
        """
        string = re.sub(r'```', '', string).strip()
        string = string.replace('\n', ' ')
        string = ' '.join(string.split())
        # Remove trailing semicolon
        if string.endswith(';'):
            string = string[:-1].strip()
        return string

    def _clean_Pythonstring(self, string):
        """
        Clean and format the Python script string.

        Parameters:
        - string (str): The Python script string to be cleaned.

        Returns:
        - Cleaned Python script string.
        """
        string = re.sub(r'```', '', string).strip()
        # Remove trailing semicolon
        if string.endswith(';'):
            string = string[:-1].strip()
        return string 

    def _extract_sql_and_python(self, content):
        """
        Extract SQL query and Python script from the generated content.

        Parameters:
        - content (str): The content containing both SQL and Python script.

        Returns:
        - Tuple containing the cleaned SQL query and Python script.
        """
        # Define the keywords in lowercase
        sql_keyword = "sql query:"
        python_keyword = "python script:"
        
        # Convert content to lowercase for case-insensitive matching of keywords
        content_lower = content.lower()
        
        # Find positions of the keywords in the lowercase content
        sql_pos = content_lower.find(sql_keyword)
        python_pos = content_lower.find(python_keyword)
        
        if python_pos != -1:
            # Split the content based on the Python Script keyword
            parts = content.split(content[python_pos:python_pos + len(python_keyword)])
            
            # Extract SQL query
            if sql_pos != -1:
                sql_part = parts[0].split(content[sql_pos:sql_pos + len(sql_keyword)])[1].strip()
            else:
                sql_part = None
            
            # Extract Python script
            python_part = parts[1].strip()
        else:
            # If "Python Script:" is not found, extract only the SQL query
            if sql_pos != -1:
                sql_part = content.split(content[sql_pos:sql_pos + len(sql_keyword)])[1].strip()
            else:
                sql_part = None
            python_part = None  # No Python script present

        sql_part = self._clean_SQLstring(sql_part)
        python_part = self._clean_Pythonstring(python_part)

        return sql_part, python_part

    def query_LLaMA(self, query):
        """
        Query the LLaMA model with a user query to generate SQL and Python code.

        Parameters:
        - query (str): The user query to be processed.

        Returns:
        - Tuple containing the SQL query and Python plot script.
        """
        # Check if the query contains any relevant table
        if not any(table in query for table in self.tables_available):
            raise ValueError(
                f"Error: Your query does not reference any of the relevant tables: {', '.join(self.tables_available)}"
            )

        messages = self._generate_prompt(query)
        
        # Prepare the prompt for the model
        prompt = self.pipeline_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline_model.tokenizer.eos_token_id,
            self.pipeline_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate the output from the model
        outputs = self.pipeline_model(
            prompt,
            max_new_tokens=512,  # Increased token limit to accommodate both SQL and Python code
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5,
            top_p=1,
        )
        content = outputs[0]["generated_text"][len(prompt):]
        print(content)
        sql_part, python_plot = self._extract_sql_and_python(content)
        
        return sql_part, python_plot

    def query_cosmohub(self, query):
        """
        Execute a SQL query on the Cosmohub database and return the results as a DataFrame.

        Parameters:
        - query (str): The SQL query to be executed.

        Returns:
        - pd.DataFrame: The DataFrame containing the query results.
        """
        conn = hive.connect(
            host='hsrv01.pic.es',
            port='10000',
            database='cosmohub',
            auth='KERBEROS',
            kerberos_service_name='hive',
        )
        df = pd.read_sql(query, conn)  
        
        return df

    def execute_plot_script(self, script: str, df: pd.DataFrame) -> None:
        """
        Execute a plotting script using the provided DataFrame.

        Parameters:
        - script (str): A string containing the script to execute.
        - df (pd.DataFrame): The DataFrame to be used in the script.
        
        Returns:
        - None
        """
        # Create a safe namespace for executing the script
        namespace = {'df': df, 'plt': plt}
        
        # Read the script from a string and execute it
        try:
            exec(script, globals(), namespace)
        except Exception as e:
            print(f"An error occurred while executing the script: {e}")
