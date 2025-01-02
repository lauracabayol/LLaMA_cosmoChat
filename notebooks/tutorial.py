# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: llama_py39
#     language: python
#     name: llama_py39
# ---

# %load_ext autoreload
# %autoreload 2

import sys

sys.path.append('../LLaMA_cosmoChat')

from chat import ChatCosmoHub


base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = '/data/aai/scratch/lcabayol/LLaMA_cosmoChat/cache'

import os
os.environ["KERNEL_HUGGINGFACE"]="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"

chatCH=ChatCosmoHub(cache_dir=cache_dir,
                   base_model=base_model)

example_query ="Provide the SQL query to get the Gaia G-band mean magnitude, proper motion in right ascension direction, and proper motion in declination direction for sources with proper motion in right assencion and declination greater than 0 mas/yr from table quaia_v1"

sql_part = chatCH.query_LLaMA(example_query)

chatCH.execute_plot_script(python_plot, df)



