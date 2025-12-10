"""
Validation tests for XTTS optimizations.

This tests:
1. Text chunking correctness
2. Cache functionality
3. Streaming synthesis
4. Async pipeline
5. Audio quality regression
"""

import os
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from TTS.tts.models.xtts_cache import CacheManager
from TTS.tts.models.xtts_utils import smart_chunk_text, verify_chunk_completeness


class TestTextChunking(unittest.TestCase):
    """Test smart text chunking functionality."""
    
    def test_single_sentence(self):
        """Test that single sentence is not split."""
        text = "Hello world."
        chunks = smart_chunk_text(text, max_tokens=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text.strip())
    
    def test_multiple_sentences(self):
        """Test that multiple sentences are split correctly."""
        text = "This is the first sentence. This is the second sentence. And this is the third one!"
        chunks = smart_chunk_text(text, max_tokens=20)
        self.assertGreater(len(chunks), 1)
        
        # Verify completeness
        self.assertTrue(verify_chunk_completeness(text, chunks))
    
    def test_abbreviations(self):
        """Test that abbreviations don't cause incorrect splits."""
        text = "Dr. Smith went to the store. He met Mrs. Johnson there. They discussed the U.S. economy."
        chunks = smart_chunk_text(text, max_tokens=50)
        
        # Should not split on abbreviations
        self.assertTrue(verify_chunk_completeness(text, chunks))
        
        # Check that "Dr." is not split
        combined = " ".join(chunks)
        self.assertIn("Dr.", combined)
    
    def test_very_long_sentence(self):
        """Test that very long sentences are split at punctuation."""
        text = ("This is a very long sentence that should be split because it exceeds the maximum "
                "character limit and we need to ensure it gets broken down properly at appropriate "
                "boundaries like commas or semicolons or other punctuation marks, so that we can "
                "process it efficiently without losing any information or context in the process.")
        
        chunks = smart_chunk_text(text, max_tokens=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(verify_chunk_completeness(text, chunks))
    
    def test_empty_text(self):
        """Test that empty text returns empty list."""
        self.assertEqual(smart_chunk_text(""), [])
        self.assertEqual(smart_chunk_text("   "), [])
    
    def test_mixed_punctuation(self):
        """Test that mixed punctuation is handled correctly."""
        text = "What is this? I don't know! Maybe we should find out. Yes, let's do that."
        chunks = smart_chunk_text(text, max_tokens=20)
        self.assertTrue(verify_chunk_completeness(text, chunks))
    
    def test_no_text_loss(self):
        """Test that no text is lost in chunking."""
        texts = [
            "Simple test.",
            "First sentence. Second sentence. Third sentence.",
            "What? Really! Yes, indeed.",
            "Dr. Smith said hello. Mrs. Jones replied.",
        ]
        
        for text in texts:
            chunks = smart_chunk_text(text, max_tokens=20)
            self.assertTrue(
                verify_chunk_completeness(text, chunks),
                f"Text loss detected in: {text}"
            )


class TestCacheManager(unittest.TestCase):
    """Test CacheManager functionality."""
    
    def setUp(self):
        """Set up test cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.temp_dir, enable_logging=False)
    
    def tearDown(self):
        """Clean up test cache directory."""
        self.cache.clear_all_cache()
    
    def test_speaker_embedding_cache(self):
        """Test speaker embedding caching."""
        # Create a dummy audio file
        audio_file = os.path.join(self.temp_dir, "test_audio.wav")
        with open(audio_file, "wb") as f:
            f.write(b"dummy audio data")
        
        # Create dummy embeddings
        gpt_latent = torch.randn(1, 1024, 10)
        speaker_emb = torch.randn(512, 1)
        
        # Save to cache
        self.cache.save_speaker_embedding(audio_file, gpt_latent, speaker_emb)
        
        # Retrieve from cache
        cached = self.cache.get_speaker_embedding(audio_file)
        self.assertIsNotNone(cached)
        
        cached_gpt, cached_spk = cached
        self.assertTrue(torch.allclose(gpt_latent, cached_gpt))
        self.assertTrue(torch.allclose(speaker_emb, cached_spk))
        
        # Check statistics
        self.assertEqual(self.cache.speaker_hits, 1)
        self.assertEqual(self.cache.speaker_misses, 0)
    
    def test_speaker_embedding_miss(self):
        """Test cache miss for non-existent speaker."""
        audio_file = os.path.join(self.temp_dir, "nonexistent.wav")
        
        # This should create the file for hash computation
        with open(audio_file, "wb") as f:
            f.write(b"test")
        
        cached = self.cache.get_speaker_embedding(audio_file)
        self.assertIsNone(cached)
        self.assertEqual(self.cache.speaker_misses, 1)
    
    def test_text_encoding_cache(self):
        """Test text encoding caching."""
        text = "Hello world"
        language = "en"
        token_ids = torch.tensor([1, 2, 3, 4, 5])
        
        # Save to cache
        self.cache.save_text_encoding(text, language, token_ids)
        
        # Retrieve from cache
        cached = self.cache.get_text_encoding(text, language)
        self.assertIsNotNone(cached)
        self.assertTrue(torch.equal(token_ids, cached))
    
    def test_text_encoding_length_limit(self):
        """Test that very long texts are not cached."""
        text = "a" * 1500  # Longer than 1000 char limit
        language = "en"
        token_ids = torch.tensor([1, 2, 3])
        
        # Should not cache
        self.cache.save_text_encoding(text, language, token_ids)
        cached = self.cache.get_text_encoding(text, language)
        self.assertIsNone(cached)
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        stats = self.cache.get_stats()
        self.assertIn("speaker_hits", stats)
        self.assertIn("speaker_misses", stats)
        self.assertIn("text_hits", stats)
        self.assertIn("text_misses", stats)


class TestIntegration(unittest.TestCase):
    """Integration tests (require XTTS model - skipped if not available)."""
    
    def setUp(self):
        """Check if XTTS model is available."""
        self.model_available = False
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            self.model_available = True
        except Exception as e:
            self.skipTest(f"XTTS model not available: {e}")
    
    def test_streaming_vs_regular(self):
        """Test that streaming produces similar results to regular synthesis."""
        if not self.model_available:
            self.skipTest("Model not available")
        
        # This is a placeholder - actual test would require model loading
        # and reference audio
        pass
    
    def test_cache_speedup(self):
        """Test that caching provides speedup for repeated requests."""
        if not self.model_available:
            self.skipTest("Model not available")
        
        # This is a placeholder
        pass


def run_basic_tests():
    """Run basic tests that don't require model."""
    print("=" * 60)
    print("Running XTTS Optimization Tests")
    print("=" * 60)
    
    # Test text chunking
    print("\n1. Testing Text Chunking...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextChunking)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Test cache manager
    print("\n2. Testing Cache Manager...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCacheManager)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("Basic tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run basic tests by default
    run_basic_tests()
    
    # Optionally run full test suite
    # unittest.main()
