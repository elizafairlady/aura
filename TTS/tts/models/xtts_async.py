"""
Async request scheduler for XTTS model.

This module provides:
1. AsyncTTSPipeline for handling multiple concurrent TTS requests
2. Queue-based request scheduling
3. Non-blocking synthesis with futures
"""

import asyncio
import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSRequest:
    """Data class for TTS request."""
    text: str
    speaker_wav: str
    language: str
    future: Future
    kwargs: dict


class AsyncTTSPipeline:
    """
    Async pipeline for processing multiple TTS requests.
    
    This doesn't actually batch the forward passes (XTTS doesn't benefit from batching),
    but allows multiple requests to be queued and processed without blocking the caller.
    """
    
    def __init__(self, model, max_queue_size: int = 16, enable_logging: bool = True):
        """
        Initialize the async TTS pipeline.
        
        Args:
            model: XTTS model instance
            max_queue_size: Maximum number of requests in queue before blocking
            enable_logging: Whether to log request processing
        """
        self.model = model
        self.max_queue_size = max_queue_size
        self.enable_logging = enable_logging
        
        # Request queue
        self.request_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Processing thread
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        
        # Start processing thread
        self.start()
    
    def start(self):
        """Start the background processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.processing_thread.start()
            if self.enable_logging:
                logger.info("AsyncTTSPipeline started")
    
    def stop(self):
        """Stop the background processing thread."""
        self.stop_event.set()
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=5.0)
        if self.enable_logging:
            logger.info("AsyncTTSPipeline stopped")
    
    def _process_queue(self):
        """Background thread that processes requests from the queue."""
        while not self.stop_event.is_set():
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=0.1)
                
                try:
                    # Process the request
                    if self.enable_logging:
                        logger.debug(f"Processing request: {request.text[:50]}...")
                    
                    # Use streaming or regular synthesis
                    if request.kwargs.get("streaming", False):
                        # Collect streaming chunks
                        chunks = []
                        for chunk in self.model.tts_streaming(
                            text=request.text,
                            speaker_wav=request.speaker_wav,
                            language=request.language,
                            **{k: v for k, v in request.kwargs.items() if k != "streaming"}
                        ):
                            chunks.append(chunk)
                        
                        # Concatenate all chunks
                        result = np.concatenate(chunks, axis=0)
                    else:
                        # Get conditioning latents
                        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                            audio_path=request.speaker_wav,
                            use_cache=True,
                        )
                        
                        # Regular inference
                        output = self.model.inference(
                            text=request.text,
                            language=request.language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            **request.kwargs
                        )
                        result = output["wav"]
                    
                    # Set the result in the future
                    request.future.set_result(result)
                    self.completed_requests += 1
                    
                    if self.enable_logging:
                        logger.debug(f"Request completed: {len(result)} samples")
                
                except Exception as e:
                    # Set exception in the future
                    request.future.set_exception(e)
                    self.failed_requests += 1
                    logger.error(f"Request failed: {e}")
                
                finally:
                    # Mark task as done
                    self.request_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
    
    def synthesize_async(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        **kwargs
    ) -> Future:
        """
        Submit a TTS request and return a Future immediately.
        
        Args:
            text: Input text to synthesize
            speaker_wav: Path to speaker reference audio
            language: Language code
            **kwargs: Additional synthesis parameters
            
        Returns:
            Future that will contain the synthesized audio (numpy array)
            
        Example:
            >>> pipeline = AsyncTTSPipeline(model)
            >>> future1 = pipeline.synthesize_async("Hello", "speaker.wav", "en")
            >>> future2 = pipeline.synthesize_async("World", "speaker.wav", "en")
            >>> audio1 = future1.result()  # Blocks until ready
            >>> audio2 = future2.result()
        """
        # Create future for the result
        future = Future()
        
        # Create request
        request = TTSRequest(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            future=future,
            kwargs=kwargs
        )
        
        # Add to queue
        try:
            self.request_queue.put(request, block=True, timeout=10.0)
            self.total_requests += 1
            
            if self.enable_logging:
                logger.debug(f"Request queued: {text[:50]}...")
        
        except queue.Full:
            future.set_exception(Exception("Request queue is full"))
        
        return future
    
    async def synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        **kwargs
    ) -> np.ndarray:
        """
        Async version of synthesize that can be awaited.
        
        Args:
            text: Input text to synthesize
            speaker_wav: Path to speaker reference audio
            language: Language code
            **kwargs: Additional synthesis parameters
            
        Returns:
            Synthesized audio as numpy array
            
        Example:
            >>> pipeline = AsyncTTSPipeline(model)
            >>> audio = await pipeline.synthesize("Hello", "speaker.wav", "en")
        """
        future = self.synthesize_async(text, speaker_wav, language, **kwargs)
        
        # Wait for the future in a non-blocking way
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, future.result)
        
        return result
    
    def get_stats(self) -> dict:
        """
        Get statistics about request processing.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "queue_size": self.request_queue.qsize(),
            "max_queue_size": self.max_queue_size,
        }
    
    def print_stats(self):
        """Print processing statistics."""
        stats = self.get_stats()
        print("\n=== AsyncTTSPipeline Statistics ===")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Completed: {stats['completed_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Queue Size: {stats['queue_size']}/{stats['max_queue_size']}")
        print("=" * 36 + "\n")
    
    def wait_until_done(self, timeout: Optional[float] = None):
        """
        Wait until all queued requests are processed.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        """
        try:
            if timeout is not None:
                # Use thread to implement timeout
                done_event = threading.Event()
                
                def wait_thread():
                    self.request_queue.join()
                    done_event.set()
                
                thread = threading.Thread(target=wait_thread, daemon=True)
                thread.start()
                
                if not done_event.wait(timeout=timeout):
                    raise TimeoutError(f"Queue not empty after {timeout} seconds")
            else:
                self.request_queue.join()
        except Exception as e:
            logger.error(f"Error waiting for queue: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.stop()
        except:
            pass
