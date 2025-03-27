from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    summary: str

class TestRequest(BaseModel):
    text: str