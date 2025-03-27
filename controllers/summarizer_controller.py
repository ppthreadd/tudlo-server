from fastapi import APIRouter, HTTPException
from models.schemas import SummarizeRequest, SummarizeResponse, TestRequest
from services.summarizer_service import SummarizerService

router = APIRouter()
summarizer_service = SummarizerService()

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty.")

        summary = summarizer_service.summarize(request.text)
        return SummarizeResponse(summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_post(request: TestRequest):
    return {"message": f"Received: {request.text}"}