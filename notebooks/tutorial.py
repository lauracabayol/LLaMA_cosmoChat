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

sys.path.append('../chatCosmoHub')

from chat import ChatCosmoHub

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_dir = '/data/aai/scratch/lcabayol/chatCosmoHub/cache'

chatCH=ChatCosmoHub(cache_dir=cache_dir,
                   base_model=base_model)

example_query ="Provide the SQL query to get the Gaia G-band mean magnitude, proper motion in right ascension direction, and proper motion in declination direction for sources with proper motion in right assencion and declination greater than 0 mas/yr from table quaia_v1, and generate a Python script to plot a scatter plot of proper motion in ra vs proper motion in dec, where the color of the points represents the magnitude in the G-band."

sql_part, python_plot = chatCH.query_LLaMA(example_query)

sql_part

python_plot

# +
from pyhive import hive
import pandas as pd
conn = hive.connect(
    host='hsrv01.pic.es',
    port='10000',
    database='cosmohub',
    auth='KERBEROS',
    kerberos_service_name='hive',
)
#cursor = conn.cursor()
#cursor.execute(sql)
#print(cursor.description)
#res = cursor.fetchmany(10,)

df = pd.read_sql(sql_part, conn)  


# -

chatCH.execute_plot_script(python_plot, df)



