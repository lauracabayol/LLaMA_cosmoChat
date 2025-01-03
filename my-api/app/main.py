# /bin/python main.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from LLaMA_cosmoChat.chat import ChatCosmoHub

app = FastAPI()
templates = Jinja2Templates(directory="templates")

cache_dir = "/data/aai/scratch/lcabayol/LLaMA_cosmoChat/cache"
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize ChatCosmoHub
chatCH=ChatCosmoHub(cache_dir=cache_dir,
                   base_model=base_model)
# Model for JSON endpoint
class QueryInput(BaseModel):
    query: str

# JSON endpoint
@app.post("/api/generate-sql")
async def generate_sql_json(input_data: QueryInput):
    sql_query = chatCH.query_LLaMA(input_data.query)
    return {"sql_query": sql_query}

# HTML form endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "sql_query": None})

@app.post("/", response_class=HTMLResponse)
async def generate_sql(request: Request, query: str = Form(...)):
    sql_query = chatCH.query_LLaMA(query)
    return templates.TemplateResponse("index.html", {"request": request, "sql_query": sql_query}) 