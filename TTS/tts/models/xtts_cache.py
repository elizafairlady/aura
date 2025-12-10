"""
Cache Manager for XTTS model optimizations.

This module provides caching for:
1. Speaker embeddings (gpt_cond_latent + speaker_embedding)
2. Text encodings (token IDs + embeddings)
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for XTTS speaker embeddings and text encodings."""

    def __init__(self, cache_dir: Optional[str] = None, enable_logging: bool = True):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store persistent cache files. 
                      Defaults to ~/.cache/xtts_cache/
            enable_logging: Whether to log cache hit/miss statistics
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/xtts_cache")
        
        self.cache_dir = Path(cache_dir)
        self.speaker_cache_dir = self.cache_dir / "speaker_embeddings"
        self.text_cache_dir = self.cache_dir / "text_encodings"
        
        # Create cache directories
        self.speaker_cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.speaker_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.text_cache: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self.enable_logging = enable_logging
        self.speaker_hits = 0
        self.speaker_misses = 0
        self.text_hits = 0
        self.text_misses = 0
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute MD5 hash of audio file bytes.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            MD5 hash string
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def _compute_text_hash(self, text: str, language: str) -> str:
        """
        Compute SHA256 hash of text + language.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            SHA256 hash string
        """
        content = f"{text}|{language}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_speaker_embedding(
        self, 
        audio_path: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve cached speaker embedding.
        
        Args:
            audio_path: Path to the reference audio file
            
        Returns:
            Tuple of (gpt_cond_latent, speaker_embedding) or None if not cached
        """
        # Compute cache key
        cache_key = self._compute_file_hash(audio_path)
        
        # Check in-memory cache first
        if cache_key in self.speaker_cache:
            self.speaker_hits += 1
            if self.enable_logging:
                logger.debug(f"Speaker embedding cache HIT (in-memory): {audio_path}")
            return self.speaker_cache[cache_key]
        
        # Check disk cache
        cache_file = self.speaker_cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            try:
                cached_data = torch.load(cache_file, map_location="cpu", weights_only=False)
                gpt_cond_latent = cached_data["gpt_cond_latent"]
                speaker_embedding = cached_data["speaker_embedding"]
                
                # Store in memory for faster subsequent access
                self.speaker_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
                
                self.speaker_hits += 1
                if self.enable_logging:
                    logger.debug(f"Speaker embedding cache HIT (disk): {audio_path}")
                
                return (gpt_cond_latent, speaker_embedding)
            except Exception as e:
                logger.warning(f"Failed to load cached speaker embedding: {e}")
        
        # Cache miss
        self.speaker_misses += 1
        if self.enable_logging:
            logger.debug(f"Speaker embedding cache MISS: {audio_path}")
        
        return None
    
    def save_speaker_embedding(
        self,
        audio_path: str,
        gpt_cond_latent: torch.Tensor,
        speaker_embedding: torch.Tensor
    ):
        """
        Save speaker embedding to cache.
        
        Args:
            audio_path: Path to the reference audio file
            gpt_cond_latent: GPT conditioning latents
            speaker_embedding: Speaker embedding vector
        """
        cache_key = self._compute_file_hash(audio_path)
        
        # Save to in-memory cache
        self.speaker_cache[cache_key] = (
            gpt_cond_latent.cpu(),
            speaker_embedding.cpu()
        )
        
        # Save to disk cache
        cache_file = self.speaker_cache_dir / f"{cache_key}.pt"
        try:
            torch.save(
                {
                    "gpt_cond_latent": gpt_cond_latent.cpu(),
                    "speaker_embedding": speaker_embedding.cpu()
                },
                cache_file
            )
            if self.enable_logging:
                logger.debug(f"Saved speaker embedding to cache: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to save speaker embedding to cache: {e}")
    
    def get_text_encoding(
        self,
        text: str,
        language: str
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached text encoding.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Token IDs tensor or None if not cached
        """
        # Only cache texts shorter than 1000 characters
        if len(text) > 1000:
            return None
        
        cache_key = self._compute_text_hash(text, language)
        
        # Check in-memory cache
        if cache_key in self.text_cache:
            self.text_hits += 1
            if self.enable_logging:
                logger.debug(f"Text encoding cache HIT (in-memory)")
            return self.text_cache[cache_key]
        
        # Check disk cache
        cache_file = self.text_cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            try:
                token_ids = torch.load(cache_file, map_location="cpu", weights_only=True)
                
                # Store in memory
                self.text_cache[cache_key] = token_ids
                
                self.text_hits += 1
                if self.enable_logging:
                    logger.debug(f"Text encoding cache HIT (disk)")
                
                return token_ids
            except Exception as e:
                logger.warning(f"Failed to load cached text encoding: {e}")
        
        # Cache miss
        self.text_misses += 1
        if self.enable_logging:
            logger.debug(f"Text encoding cache MISS")
        
        return None
    
    def save_text_encoding(
        self,
        text: str,
        language: str,
        token_ids: torch.Tensor
    ):
        """
        Save text encoding to cache.
        
        Args:
            text: Input text
            language: Language code
            token_ids: Encoded token IDs
        """
        # Only cache texts shorter than 1000 characters
        if len(text) > 1000:
            return
        
        cache_key = self._compute_text_hash(text, language)
        
        # Save to in-memory cache
        self.text_cache[cache_key] = token_ids.cpu()
        
        # Save to disk cache
        cache_file = self.text_cache_dir / f"{cache_key}.pt"
        try:
            torch.save(token_ids.cpu(), cache_file)
            if self.enable_logging:
                logger.debug(f"Saved text encoding to cache")
        except Exception as e:
            logger.warning(f"Failed to save text encoding to cache: {e}")
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        speaker_total = self.speaker_hits + self.speaker_misses
        text_total = self.text_hits + self.text_misses
        
        return {
            "speaker_hits": self.speaker_hits,
            "speaker_misses": self.speaker_misses,
            "speaker_hit_rate": self.speaker_hits / speaker_total if speaker_total > 0 else 0.0,
            "text_hits": self.text_hits,
            "text_misses": self.text_misses,
            "text_hit_rate": self.text_hits / text_total if text_total > 0 else 0.0,
            "speaker_cache_size": len(self.speaker_cache),
            "text_cache_size": len(self.text_cache),
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print("\n=== XTTS Cache Statistics ===")
        print(f"Speaker Embeddings:")
        print(f"  Hits: {stats['speaker_hits']}")
        print(f"  Misses: {stats['speaker_misses']}")
        print(f"  Hit Rate: {stats['speaker_hit_rate']:.2%}")
        print(f"  In-Memory Cache Size: {stats['speaker_cache_size']}")
        print(f"\nText Encodings:")
        print(f"  Hits: {stats['text_hits']}")
        print(f"  Misses: {stats['text_misses']}")
        print(f"  Hit Rate: {stats['text_hit_rate']:.2%}")
        print(f"  In-Memory Cache Size: {stats['text_cache_size']}")
        print("=" * 30 + "\n")
    
    def clear_memory_cache(self):
        """Clear in-memory caches (keeps disk cache)."""
        self.speaker_cache.clear()
        self.text_cache.clear()
        logger.info("Cleared in-memory caches")
    
    def clear_all_cache(self):
        """Clear both in-memory and disk caches."""
        # Clear memory
        self.clear_memory_cache()
        
        # Clear disk
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.speaker_cache_dir.mkdir(parents=True, exist_ok=True)
            self.text_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset statistics
        self.speaker_hits = 0
        self.speaker_misses = 0
        self.text_hits = 0
        self.text_misses = 0
        
        logger.info("Cleared all caches (memory and disk)")
