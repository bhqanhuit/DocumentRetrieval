from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str

class SearchResponse(BaseModel):
    file_path: str
    page_number: int