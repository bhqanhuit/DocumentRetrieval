from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .services import DocumentRetriever
from .models import SearchQuery, SearchResponse
from typing import List

app = FastAPI(title="PDF Search API")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Dependency to get DocumentRetriever instance
retriever = DocumentRetriever(embeddings_dir="data/doc_embedding")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Serve the main HTML page.
    """
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/search", response_model=List[SearchResponse])
async def search_documents(query: SearchQuery):
    """
    Search embedded PDF documents with a query string.
    Returns top 5 matching pages with file paths and page numbers.
    """
    try:
        results = retriever.search(query.query, top_k=5)
        return [
            SearchResponse(file_path=result["file_path"], page_number=result["page_number"])
            for result in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")