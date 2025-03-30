from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class SingleSummaryResult(BaseModel):
    topic: str
    summary: str
    text_length: int
    processing_time: float
    temperature_used: float
    paragraphs: int
    sentences: int


class SummaryResult(BaseModel):
    index: int
    topic: str
    summary: str
    text_length: int
    processing_time: float
    temperature_used: float
    paragraphs: int
    sentences: int

class ErrorResult(BaseModel):
    index: int
    error: str
    text_sample: str

class ProcessingMetrics(BaseModel):
    total_texts: int
    succeeded: int
    failed: int
    total_time: float
    avg_time_per_text: Optional[float]

class BatchSummarizeResponse(BaseModel):
    results: List[SummaryResult]
    errors: List[ErrorResult]
    metrics: ProcessingMetrics
    params: Dict[str, Any]