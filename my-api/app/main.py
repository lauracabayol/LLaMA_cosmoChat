from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
sys.path.append('../LLaMA_cosmoChat/')
from chat import ChatCosmoHub
import pandas as pd

app = FastAPI()

# Configuration
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = '/data/aai/scratch/lcabayol/chatCosmoHub/cache'

# Initialize ChatCosmoHub
chat_ch = ChatCosmoHub(cache_dir=CACHE_DIR, base_model=BASE_MODEL)

class Query(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def get_query_form():
    return '''
        <form method="post" action="/generate">
            <textarea name="query" rows="4" cols="50" placeholder="Enter your query here..."></textarea>
            <br>
            <input type="submit" value="Generate">
        </form>
    '''

@app.post("/generate")
async def generate_query_and_plot(query: str = Form(...)):
    try:
        # Generate SQL and plotting code
        sql_query, python_plot = chat_ch.query_LLaMA(query)
        
        return {
            "sql_query": sql_query,
            "python_plot": python_plot,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}