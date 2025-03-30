from typing import List, Dict, Any
from pydantic import BaseModel

class BatchSummarizeResponse(BaseModel):
    summaries: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    metrics: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "summaries": [
                    {
                        "index": 0,
                        "summary": "1. Core Subject...\n2. Key Findings...",
                        "text_length": 150
                    }
                ],
                "errors": [],
                "metrics": {
                    "total": 2,
                    "succeeded": 2,
                    "failed": 0,
                    "avg_time": 1.5,
                    "total_time": 3.0
                }
            }
        }