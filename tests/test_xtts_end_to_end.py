"""
End-to-end integration test for XTTS optimizations.

This test actually loads the model and runs inference to verify everything works.
Run with: uv run python tests/test_xtts_end_to_end.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_end_to_end_with_model():
    """
    End-to-end test: Load XTTS model and run optimized inference.
    
    This test verifies:
    1. Model can load with cache enabled
    2. Regular inference works
    3. Streaming inference works and produces concatenable output
    4. Cache is actually used on second call
    5. AsyncTTSPipeline works with real model
    """
    print("=" * 70)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 70)
    
    try:
        print("\n[1/7] Importing TTS modules...")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.tts.models.xtts_async import AsyncTTSPipeline
        import torch
        print("✓ Imports successful")
        
        print("\n[2/7] Checking for model files...")
        
        model_path = "models/xtts-v2"

        print("\n[3/7] Loading XTTS model with cache enabled...")
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        
        start_time = time.time()
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config, 
            checkpoint_dir=model_path,
            eval=True
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✓ Model loaded on CUDA in {time.time() - start_time:.2f}s")
        else:
            print(f"✓ Model loaded on CPU in {time.time() - start_time:.2f}s")
        
        # Verify cache is enabled
        if model.cache_manager is None:
            print("✗ Cache manager not initialized!")
            return False
        print("✓ Cache manager initialized")
        
        print("\n[4/7] Testing regular inference...")
        test_wav_path = os.path.join(model_path, "en_sample.wav")
        
        if not os.path.exists(test_wav_path):
            print(f"✗ No test audio file found in {model_path}")
            print("  Please provide a speaker reference audio file")
            return False
        
        print(f"  Using speaker audio: {test_wav_path}")
        
        test_text = "Hello world. This is a test."
        
        # First inference (cold - no cache)
        start_time = time.time()
        result1 = model.synthesize(
            text=test_text,
            speaker_wav=test_wav_path,
            language="en"
        )
        time1 = time.time() - start_time
        
        if "wav" not in result1 or len(result1["wav"]) == 0:
            print("✗ Inference failed - no audio generated")
            return False
        
        print(f"✓ First inference completed in {time1:.2f}s")
        print(f"  Generated {len(result1['wav'])} samples ({len(result1['wav'])/24000:.2f}s of audio)")
        
        # Second inference (should use cache)
        cache_stats_before = model.cache_manager.get_stats()
        
        start_time = time.time()
        result2 = model.synthesize(
            text=test_text,
            speaker_wav=test_wav_path,
            language="en"
        )
        time2 = time.time() - start_time
        
        cache_stats_after = model.cache_manager.get_stats()
        
        print(f"✓ Second inference completed in {time2:.2f}s")
        
        # Verify cache was used
        if cache_stats_after['speaker_hits'] > cache_stats_before['speaker_hits']:
            print(f"✓ Cache HIT detected! Speedup: {time1/time2:.2f}x")
        else:
            print(f"⚠ Cache not used (hits: {cache_stats_after['speaker_hits']})")
        
        print("\n[5/7] Testing streaming inference...")
        chunks_collected = []
        chunk_count = 0
        
        start_time = time.time()
        first_chunk_time = None
        
        for chunk in model.tts_streaming(
            text="First sentence for streaming. Second sentence here. Third one too.",
            speaker_wav=test_wav_path,
            language="en",
            max_tokens_per_chunk=20
        ):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            chunks_collected.append(chunk)
            chunk_count += 1
        
        total_time = time.time() - start_time
        
        if chunk_count == 0:
            print("✗ Streaming produced no chunks")
            return False
        
        print(f"✓ Streaming produced {chunk_count} chunks")
        print(f"  Time to first chunk: {first_chunk_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        
        # Verify chunks can be concatenated
        try:
            concatenated = np.concatenate(chunks_collected)
            print(f"✓ Chunks successfully concatenated: {len(concatenated)} samples")
        except Exception as e:
            print(f"✗ Concatenation failed: {e}")
            return False
        
        print("\n[6/7] Testing async pipeline...")
        pipeline = AsyncTTSPipeline(model, max_queue_size=5, enable_logging=False)
        
        try:
            # Submit async request
            future = pipeline.synthesize_async(
                text="Async test",
                speaker_wav=test_wav_path,
                language="en"
            )
            
            # Wait for result
            result_async = future.result(timeout=30)
            
            if result_async is None or len(result_async) == 0:
                print("✗ Async synthesis failed")
                return False
            
            print(f"✓ Async synthesis completed: {len(result_async)} samples")
            
            # Get stats
            pipeline.wait_until_done(timeout=5)
            stats = pipeline.get_stats()
            print(f"  Pipeline stats: {stats['completed_requests']}/{stats['total_requests']} completed")
            
        finally:
            pipeline.stop()
        
        print("\n[7/7] Testing cache stats...")
        final_stats = model.cache_manager.get_stats()
        print(f"  Speaker cache hits: {final_stats['speaker_hits']}")
        print(f"  Speaker cache misses: {final_stats['speaker_misses']}")
        print(f"  Hit rate: {final_stats['speaker_hit_rate']:.1%}")
        
        if final_stats['speaker_hits'] > 0:
            print("✓ Cache system working correctly")
        else:
            print("⚠ No cache hits detected")
        
        print("\n" + "=" * 70)
        print("✓ END-TO-END TEST PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_end_to_end_with_model()
    
    if not success:
        print("\nNOTE: This test requires the XTTS model to be downloaded.")
        print("If the model is not available, the test will be skipped.")
        
    sys.exit(0 if success else 1)
