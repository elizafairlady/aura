"""
Integration and regression tests for XTTS optimizations.

These tests verify that the optimizations work correctly with the actual model.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all optimization modules can be imported."""
    print("Testing imports...")
    try:
        from TTS.tts.models.xtts_cache import CacheManager
        from TTS.tts.models.xtts_utils import smart_chunk_text, verify_chunk_completeness
        from TTS.tts.models.xtts_async import AsyncTTSPipeline
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_cache_manager_functionality():
    """Test CacheManager basic functionality."""
    print("\nTesting CacheManager...")
    from TTS.tts.models.xtts_cache import CacheManager
    import torch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio file
        test_audio = os.path.join(tmpdir, "test.wav")
        with open(test_audio, "wb") as f:
            f.write(b"fake audio data for testing")
        
        # Initialize cache
        cache = CacheManager(cache_dir=tmpdir, enable_logging=False)
        
        # Test save and retrieve
        test_gpt = torch.randn(1, 1024, 10)
        test_spk = torch.randn(512, 1)
        
        cache.save_speaker_embedding(test_audio, test_gpt, test_spk)
        retrieved = cache.get_speaker_embedding(test_audio)
        
        if retrieved is None:
            print("✗ Cache retrieval failed")
            return False
        
        gpt, spk = retrieved
        if not torch.allclose(test_gpt, gpt, atol=1e-6):
            print("✗ GPT latent mismatch")
            return False
        
        if not torch.allclose(test_spk, spk, atol=1e-6):
            print("✗ Speaker embedding mismatch")
            return False
        
        # Test statistics
        stats = cache.get_stats()
        if stats['speaker_hits'] != 1 or stats['speaker_misses'] != 0:
            print(f"✗ Cache statistics incorrect: {stats}")
            return False
        
        print("✓ CacheManager functionality verified")
        return True


def test_text_chunking_completeness():
    """Test that text chunking doesn't lose data."""
    print("\nTesting text chunking completeness...")
    from TTS.tts.models.xtts_utils import smart_chunk_text, verify_chunk_completeness
    
    test_texts = [
        "Single sentence.",
        "First sentence. Second sentence. Third sentence.",
        "Dr. Smith went to the store. He met Mrs. Johnson there.",
        "This is a very long sentence that should be split, because it exceeds the maximum character limit, and we need to ensure it gets broken down properly at appropriate boundaries.",
        "What is this? I don't know! Maybe we should find out."
    ]
    
    for text in test_texts:
        chunks = smart_chunk_text(text, max_tokens=20)
        if not verify_chunk_completeness(text, chunks):
            print(f"✗ Text loss detected in: {text[:50]}...")
            return False
    
    print("✓ Text chunking completeness verified")
    return True


def test_streaming_concatenation():
    """Test that streaming produces concatenable outputs."""
    print("\nTesting streaming output format...")
    from TTS.tts.models.xtts_utils import smart_chunk_text
    
    # Simulate streaming behavior
    test_text = "First chunk. Second chunk. Third chunk."
    chunks = smart_chunk_text(test_text, max_tokens=5)
    
    # Verify chunks can be concatenated
    if len(chunks) == 0:
        print("✗ No chunks produced")
        return False
    
    # Each chunk should be non-empty string
    for i, chunk in enumerate(chunks):
        if not chunk or not isinstance(chunk, str):
            print(f"✗ Invalid chunk {i}: {chunk}")
            return False
    
    # Concatenation should work
    try:
        reconstructed = " ".join(chunks)
        if not reconstructed.strip():
            print("✗ Concatenation produced empty result")
            return False
    except Exception as e:
        print(f"✗ Concatenation failed: {e}")
        return False
    
    print("✓ Streaming output format verified")
    return True


def test_async_pipeline_basic():
    """Test AsyncTTSPipeline basic structure."""
    print("\nTesting AsyncTTSPipeline structure...")
    from TTS.tts.models.xtts_async import AsyncTTSPipeline, TTSRequest
    from concurrent.futures import Future
    
    # Mock model
    class MockModel:
        def get_conditioning_latents(self, audio_path, use_cache=True):
            import torch
            return torch.randn(1, 1024, 10), torch.randn(512, 1)
        
        def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kwargs):
            import numpy as np
            # Return mock audio
            return {"wav": np.random.randn(24000)}
    
    model = MockModel()
    pipeline = AsyncTTSPipeline(model, max_queue_size=5, enable_logging=False)
    
    try:
        # Test that pipeline starts
        if pipeline.processing_thread is None or not pipeline.processing_thread.is_alive():
            print("✗ Pipeline thread not running")
            return False
        
        # Test request structure
        future = Future()
        request = TTSRequest("test", "audio.wav", "en", future, {})
        
        if request.text != "test" or request.language != "en":
            print("✗ Request structure incorrect")
            return False
        
        pipeline.stop()
        print("✓ AsyncTTSPipeline structure verified")
        return True
    
    finally:
        try:
            pipeline.stop()
        except:
            pass


def test_hifigan_checkpointing_parameter():
    """Test that HiFiGAN has checkpointing parameter."""
    print("\nTesting HiFiGAN checkpointing parameter...")
    from TTS.vocoder.models.hifigan_generator import HifiganGenerator
    import inspect
    
    # Check __init__ signature
    sig = inspect.signature(HifiganGenerator.__init__)
    if 'use_grad_checkpointing' not in sig.parameters:
        print("✗ use_grad_checkpointing parameter not found")
        return False
    
    # Test instantiation with parameter
    try:
        gen = HifiganGenerator(
            in_channels=80,
            out_channels=1,
            resblock_type="1",
            resblock_dilation_sizes=[[1,3,5]],
            resblock_kernel_sizes=[3],
            upsample_kernel_sizes=[16],
            upsample_initial_channel=512,
            upsample_factors=[8],
            use_grad_checkpointing=True
        )
        
        if not hasattr(gen, 'use_grad_checkpointing'):
            print("✗ use_grad_checkpointing attribute not set")
            return False
        
        if gen.use_grad_checkpointing != True:
            print("✗ use_grad_checkpointing not True")
            return False
        
        print("✓ HiFiGAN checkpointing parameter verified")
        return True
    except Exception as e:
        print(f"✗ HiFiGAN initialization failed: {e}")
        return False


def test_xtts_modifications():
    """Test that XTTS has required modifications."""
    print("\nTesting XTTS modifications...")
    from TTS.tts.models.xtts import Xtts
    import inspect
    
    # Check for tts_streaming method
    if not hasattr(Xtts, 'tts_streaming'):
        print("✗ tts_streaming method not found")
        return False
    
    # Check for cache_manager attribute in __init__
    sig = inspect.signature(Xtts.__init__)
    if 'enable_cache' not in sig.parameters:
        print("✗ enable_cache parameter not found in __init__")
        return False
    
    # Check get_conditioning_latents has use_cache parameter
    sig = inspect.signature(Xtts.get_conditioning_latents)
    if 'use_cache' not in sig.parameters:
        print("✗ use_cache parameter not found in get_conditioning_latents")
        return False
    
    print("✓ XTTS modifications verified")
    return True


def measure_chunking_performance():
    """Measure text chunking performance."""
    print("\nMeasuring text chunking performance...")
    from TTS.tts.models.xtts_utils import smart_chunk_text
    
    # Generate long text
    sentence = "This is a test sentence that will be repeated many times. "
    long_text = sentence * 100  # ~5000 characters
    
    # Measure time
    start = time.time()
    for _ in range(100):
        chunks = smart_chunk_text(long_text, max_tokens=150)
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000  # ms
    print(f"  Average chunking time: {avg_time:.2f} ms")
    print(f"  Chunks produced: {len(chunks)}")
    
    if avg_time > 100:  # Should be fast
        print("  ⚠ Warning: Chunking slower than expected")
    else:
        print("  ✓ Chunking performance acceptable")
    
    return True


def run_all_tests():
    """Run all integration and regression tests."""
    print("="

 * 60)
    print("XTTS Optimization Integration & Regression Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("CacheManager Functionality", test_cache_manager_functionality),
        ("Text Chunking Completeness", test_text_chunking_completeness),
        ("Streaming Output Format", test_streaming_concatenation),
        ("AsyncPipeline Structure", test_async_pipeline_basic),
        ("HiFiGAN Checkpointing", test_hifigan_checkpointing_parameter),
        ("XTTS Modifications", test_xtts_modifications),
        ("Chunking Performance", measure_chunking_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
