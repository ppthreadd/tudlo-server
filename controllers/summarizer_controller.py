from fastapi import APIRouter, Form, HTTPException
from typing import List, Optional
from models.schemas import BatchSummarizeResponse, SingleSummaryResult
from services.summarizer_service import SummarizerService
import logging

router = APIRouter()
summarizer_service = SummarizerService()

@router.post("/summarize-single", response_model=SingleSummaryResult)
async def summarize_single_text(
    text: str = Form(..., description="Text to summarize"),
    temperature: Optional[float] = Form(0.7, ge=0.0, le=2.0),
    additional_params: Optional[str] = Form(None),
    paragraphs: Optional[int] = Form(0),
    sentences: Optional[int] = Form(0),
    timeout_seconds: Optional[int] = Form(30)
):
    """
    Summarize a single text with customizable formatting:
    - paragraphs=0, sentences=0 → Automatic formatting
    - paragraphs=3, sentences=0 → Exactly 3 paragraphs
    - paragraphs=0, sentences=5 → Automatic paragraphs with 5 sentences each
    - paragraphs=2, sentences=3 → Exactly 2 paragraphs with 3 sentences each
    """
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

        return await summarizer_service.summarize_single(
            text=text,
            temperature=temperature,
            additional_params=additional_params,
            paragraphs=paragraphs,
            sentences=sentences,
            timeout_seconds=timeout_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize", response_model=BatchSummarizeResponse)
async def summarize_texts(
    texts: List[str] = Form(..., description="List of texts to summarize"),
    temperature: Optional[float] = Form(0.7, ge=0.0, le=2.0, description="LLM creativity (0-2)"),
    additional_params: Optional[str] = Form(None, description="Custom instructions"),
    paragraphs: Optional[int] = Form(0, description="Number of paragraphs (0 for auto)"),
    sentences: Optional[int] = Form(0, description="Sentences per paragraph (0 for auto)"),
    timeout_seconds: Optional[int] = Form(30, description="Timeout per text")
):
    """
    Process multiple texts with customizable formatting:
    - paragraphs=0, sentences=0 → Automatic formatting
    - paragraphs=3, sentences=0 → Exactly 3 paragraphs
    - paragraphs=0, sentences=5 → Automatic paragraphs with 5 sentences each
    - paragraphs=2, sentences=3 → Exactly 2 paragraphs with 3 sentences each
    """
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="At least one text must be provided")

        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise HTTPException(status_code=400, detail="No valid (non-empty) texts provided")

        return await summarizer_service.summarize_batch(
            texts=valid_texts,
            temperature=temperature,
            additional_params=additional_params,
            paragraphs=paragraphs,
            sentences=sentences,
            timeout_seconds=timeout_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")