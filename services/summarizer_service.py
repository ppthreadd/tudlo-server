from fastapi import HTTPException
from llama_index.core import SummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from models.schemas import SingleSummaryResult
from models.settings import settings
from typing import List, Dict, Any, Optional
import logging
import torch
from asyncio import gather
from datetime import datetime


def _build_prompt(
        additional_params: Optional[str],
        paragraphs: int,
        sentences: int
) -> str:
    """Dynamic prompt generation based on requirements"""
    prompt = "Write a comprehensive summary containing"

    if paragraphs > 0:
        prompt += f" exactly {paragraphs} paragraphs"
        if sentences > 0:
            prompt += f", with {sentences} sentences each"
    elif sentences > 0:
        prompt += f" with {sentences} sentences per paragraph"
    else:
        prompt += " in appropriate length"

    prompt += ".\n\nStructure:\n1. Primary topic\n2. Key findings\n3. Practical applications"

    if additional_params:
        prompt += f"\n\nAdditional Requirements:\n{additional_params}"

    return prompt


class SummarizerService:
    def __init__(self):
        """Initialize service with default settings"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._init_components()
            logging.info(f"SummarizerService initialized (device: {self.device})")
        except Exception as e:
            logging.critical(f"Service initialization failed: {str(e)}")
            raise

    def _init_components(self, temperature: float = 0.3):
        """Initialize LLM and embedding components"""
        # Configure LLM with current temperature
        Settings.llm = Groq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
            max_tokens=512,
            timeout=45
        )

        # Configure embeddings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            device=self.device,
            embed_batch_size=8
        )

        # Configure text processing
        self.splitter = SentenceSplitter(
            chunk_size=8192,
            chunk_overlap=512,
            paragraph_separator="\n\n"
        )

    async def summarize_single(
            self,
            text: str,
            temperature: float = 0.7,
            additional_params: Optional[str] = None,
            paragraphs: int = 0,
            sentences: int = 0,
            timeout_seconds: int = 30
    ) -> SingleSummaryResult:
        """Process a single text with error handling"""
        start_time = datetime.now()

        try:
            if not text.strip():
                raise ValueError("Empty text content")

            self._init_components(temperature)
            prompt = _build_prompt(additional_params, paragraphs, sentences)
            document = Document(text=text)
            nodes = self.splitter.get_nodes_from_documents([document])

            index = SummaryIndex(nodes)
            query_engine = index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
                similarity_top_k=2
            )
            response = await query_engine.aquery(prompt)

            return SingleSummaryResult(
                topic=str(response).split('\n')[0][:100],
                summary=str(response),
                text_length=len(text),
                processing_time=(datetime.now() - start_time).total_seconds(),
                temperature_used=temperature,
                paragraphs=paragraphs,
                sentences=sentences
            )

        except Exception as e:
            logging.error(f"Summarization failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Processing error: {str(e)}"
            )

    async def summarize_batch(
            self,
            texts: List[str],
            temperature: float = 0.7,
            additional_params: Optional[str] = None,
            paragraphs: int = 0,
            sentences: int = 0,
            timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """Process batch with customizable parameters"""
        self._init_components(temperature)

        start_time = datetime.now()
        results = {
            "results": [],
            "errors": [],
            "metrics": {
                "total_texts": len(texts),
                "succeeded": 0,
                "failed": 0,
                "total_time": 0.0
            },
            "params": {
                "temperature": temperature,
                "paragraphs": paragraphs,
                "sentences": sentences,
                "custom_instructions": additional_params
            }
        }

        tasks = []
        for idx, text in enumerate(texts):
            tasks.append(
                self._safe_process_text(
                    text=text,
                    index=idx,
                    temperature=temperature,
                    additional_params=additional_params,
                    paragraphs=paragraphs,
                    sentences=sentences,
                    timeout=timeout_seconds
                )
            )

        batch_results = await gather(*tasks)

        for result in batch_results:
            if isinstance(result, dict) and "error" in result:
                results["errors"].append(result)
                results["metrics"]["failed"] += 1
            else:
                results["results"].append(result)
                results["metrics"]["succeeded"] += 1

        results["metrics"]["total_time"] = (datetime.now() - start_time).total_seconds()
        if results["metrics"]["succeeded"] > 0:
            results["metrics"]["avg_time_per_text"] = (
                    results["metrics"]["total_time"] / results["metrics"]["succeeded"]
            )

        return results

    async def _safe_process_text(
            self,
            text: str,
            index: int,  # This is the correct index we should use
            temperature: float,
            additional_params: Optional[str],
            paragraphs: int,
            sentences: int,
            timeout: int
    ) -> Dict[str, Any]:
        """Process individual text with error handling"""
        try:
            start_time = datetime.now()

            if not text.strip():
                raise ValueError("Empty text content")

            prompt = _build_prompt(additional_params, paragraphs, sentences)
            document = Document(text=text)
            nodes = self.splitter.get_nodes_from_documents([document])

            # Create index but don't return it
            summary_index = SummaryIndex(nodes)
            query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
                similarity_top_k=2
            )
            response = await query_engine.aquery(prompt)

            return {
                "index": index,  # Use the input index parameter, not the SummaryIndex object
                "topic": str(response).split('\n')[0][:100],
                "summary": str(response),
                "text_length": len(text),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "temperature_used": temperature,
                "paragraphs": paragraphs,
                "sentences": sentences
            }

        except Exception as e:
            logging.error(f"Error processing text {index}: {str(e)}")
            return {
                "index": index,  # Again, use the input index
                "error": str(e),
                "text_sample": text[:100] + "..." if text else ""
            }