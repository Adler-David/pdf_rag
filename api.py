from src import *
from fastapi import FastAPI

app = FastAPI()


@app.post("/search-documents")
def search_documents(query: str):
    res = retrieve_relevent_chunks_with_sentecne_transformer(query)
    return res