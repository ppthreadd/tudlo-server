from fastapi import APIRouter, Form, HTTPException
from typing import List, Optional
from models.schemas import BatchSummarizeResponse
from services.summarizer_service import SummarizerService
import logging

router = APIRouter()
summarizer_service = SummarizerService()

@router.post("/summarize", response_model=BatchSummarizeResponse)
async def summarize_texts(
    texts: List[str] = Form(..., description="List of texts to summarize"),
    timeout_seconds: Optional[int] = Form(30, description="Timeout per text in seconds")
):
    """
    Process multiple texts from form-data input
    Example cURL:
    curl -X POST http://localhost:8000/summarize \
      -F "texts=First text content" \
      -F "texts=Second text content" \
      -F "timeout_seconds=20"
    """
    try:
        if not texts:
            raise HTTPException(
                status_code=400,
                detail="At least one text must be provided"
            )

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise HTTPException(
                status_code=400,
                detail="No valid (non-empty) texts provided"
            )

        return await summarizer_service.summarize_batch(valid_texts, timeout_seconds)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )