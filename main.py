from fastapi import FastAPI
from controllers.summarizer_controller import router as summarizer_router
from config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# Include routers
app.include_router(summarizer_router, prefix="/api")