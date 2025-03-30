from llama_index.core import SummaryIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models.settings import settings
from typing import List, Dict, Any
import logging
import torch
from asyncio import gather, TimeoutError
from datetime import datetime


class SummarizerService:
    def __init__(self):
        """Initialize summarization service with optimized settings"""
        try:
            # Configure LLM with performance settings
            Settings.llm = Groq(
                model=settings.LLM_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.3,
                max_tokens=512,
                timeout=45  # Groq API timeout
            )

            # Configure embeddings with automatic device detection
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=settings.EMBEDDING_MODEL,
                device=self.device,
                embed_batch_size=8  # Optimal for most GPUs
            )

            # Configure text processing
            self.splitter = SentenceSplitter(
                chunk_size=8192,
                chunk_overlap=512,
                paragraph_separator="\n\n"
            )

            logging.info(f"SummarizerService initialized (device: {self.device})")

        except Exception as e:
            logging.critical(f"Service initialization failed: {str(e)}")
            raise

    async def summarize_batch(
            self,
            texts: List[str],
            timeout_per_text: int = 30
    ) -> Dict[str, Any]:
        """
        Process multiple texts with comprehensive error handling
        Args:
            texts: List of text strings to process
            timeout_per_text: Maximum processing time per item (seconds)
        Returns:
            Dictionary with results, errors, and metrics
        """
        start_time = datetime.now()
        results = {
            "summaries": [],
            "errors": [],
            "metrics": {
                "total_texts": len(texts),
                "processed": 0,
                "failed": 0,
                "total_time": 0.0
            }
        }

        try:
            # Process texts in parallel
            tasks = []
            for idx, text in enumerate(texts):
                tasks.append(
                    self._safe_process_text(
                        text=text,
                        index=idx,
                        timeout=timeout_per_text
                    )
                )

            batch_results = await gather(*tasks)

            # Organize results
            for result in batch_results:
                if result.get("error"):
                    results["errors"].append(result)
                    results["metrics"]["failed"] += 1
                else:
                    results["summaries"].append(result)
                    results["metrics"]["processed"] += 1

        except Exception as e:
            logging.error(f"Batch processing failed: {str(e)}")
            results["errors"].append({
                "error": "Batch processing error",
                "details": str(e)
            })

        finally:
            # Calculate metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            results["metrics"]["total_time"] = elapsed
            if results["metrics"]["processed"] > 0:
                results["metrics"]["avg_time_per_text"] = (
                        elapsed / results["metrics"]["processed"]
                )

        return results

    async def _safe_process_text(
            self,
            text: str,
            index: int,
            timeout: int
    ) -> Dict[str, Any]:
        """Wrapper with error handling for individual text processing"""
        try:
            start_time = datetime.now()

            # Basic validation
            if not text or not text.strip():
                raise ValueError("Empty text content")

            # Process the text
            summary = await self._process_text(text)

            return {
                "index": index,
                "summary": summary,
                "text_length": len(text),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            logging.warning(f"Error processing text {index}: {str(e)}")
            return {
                "index": index,
                "error": str(e),
                "text_sample": text[:100] + "..." if text else ""
            }

    async def _process_text(self, text: str) -> str:
        """Core text processing logic"""
        document = Document(text=text)
        nodes = self.splitter.get_nodes_from_documents([document])

        index = SummaryIndex(nodes)
        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
            similarity_top_k=2
        )

        response = await query_engine.aquery(
            "Extract key information in this format:\n"
            "1. Primary Topic\n"
            "2. Most Important Points\n"
            "3. Practical Applications\n"
            "Keep each point under 20 words."
        )

        return str(response)