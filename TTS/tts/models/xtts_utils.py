"""
Utility functions for XTTS model optimizations.

This module provides:
1. Smart text chunking for better memory management
2. Helper functions for streaming and processing
"""

import re
from typing import List


def smart_chunk_text(text: str, max_tokens: int = 150, language: str = "en", tokenizer=None) -> List[str]:
    """
    Split text into chunks at sentence boundaries.
    
    This ensures that:
    - Each chunk is approximately max_tokens tokens long (~4 seconds of speech)
    - Chunks split at sentence boundaries (., !, ?) when possible
    - No text is lost in the chunking process
    - Punctuation is preserved with the chunk it belongs to
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (default 150, approximately 4 seconds)
        language: Language code (used for language-specific splitting rules)
        tokenizer: Optional tokenizer for accurate token counting
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Use actual tokenizer if available for accurate token counting
    def count_tokens(s: str) -> int:
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(s, lang=language))
            except:
                pass
        # Fallback: ~3-4 characters per token (conservative estimate)
        return len(s) // 3
    
    # If text is short enough, return as single chunk
    if count_tokens(text) <= max_tokens:
        return [text.strip()]
    
    # Split on sentence boundaries
    # This regex handles:
    # - Periods, exclamation marks, question marks
    # - Avoids splitting on common abbreviations like "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr."
    # - Avoids splitting on decimal numbers like "3.14"
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+'
    
    # Split into sentences
    sentences = re.split(sentence_endings, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check token count for adding this sentence
        test_chunk = (current_chunk + " " + sentence) if current_chunk else sentence
        
        if count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it and start new chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # If the sentence itself is longer than max_tokens, split it further
            if count_tokens(sentence) > max_tokens:
                # Split on commas or other punctuation
                sub_parts = re.split(r'([,;:])\s+', sentence)
                
                # Reconstruct with punctuation
                reconstructed = []
                for i in range(0, len(sub_parts), 2):
                    if i + 1 < len(sub_parts):
                        reconstructed.append(sub_parts[i] + sub_parts[i + 1])
                    else:
                        reconstructed.append(sub_parts[i])
                
                temp_chunk = ""
                for part in reconstructed:
                    test_chunk = (temp_chunk + " " + part) if temp_chunk else part
                    
                    if count_tokens(test_chunk) <= max_tokens:
                        temp_chunk = test_chunk
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = part
                
                current_chunk = temp_chunk
            else:
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def verify_chunk_completeness(original_text: str, chunks: List[str]) -> bool:
    """
    Verify that chunking didn't lose any text.
    
    Args:
        original_text: Original input text
        chunks: List of text chunks
        
    Returns:
        True if all text is preserved, False otherwise
    """
    # Reconstruct text from chunks
    reconstructed = " ".join(chunks)
    
    # Normalize whitespace for comparison
    original_normalized = " ".join(original_text.split())
    reconstructed_normalized = " ".join(reconstructed.split())
    
    return original_normalized == reconstructed_normalized


def estimate_audio_duration(text: str, language: str = "en") -> float:
    """
    Estimate audio duration in seconds based on text length.
    
    This is a rough heuristic:
    - Average speech rate: ~150 words per minute (~2.5 words per second)
    - Average word length: ~5 characters
    
    Args:
        text: Input text
        language: Language code (for language-specific rates)
        
    Returns:
        Estimated duration in seconds
    """
    # Count words (rough estimate)
    word_count = len(text.split())
    
    # Average speech rates (words per minute) by language
    speech_rates = {
        "en": 150,  # English
        "es": 165,  # Spanish
        "fr": 160,  # French
        "de": 140,  # German
        "it": 160,  # Italian
        "pt": 155,  # Portuguese
        "pl": 145,  # Polish
        "tr": 150,  # Turkish
        "ru": 145,  # Russian
        "nl": 150,  # Dutch
        "cs": 145,  # Czech
        "ar": 140,  # Arabic
        "zh": 160,  # Chinese (characters treated differently)
        "ja": 160,  # Japanese
        "ko": 155,  # Korean
    }
    
    # Get speech rate for language, default to English
    words_per_minute = speech_rates.get(language[:2], 150)
    words_per_second = words_per_minute / 60.0
    
    # Calculate duration
    duration = word_count / words_per_second
    
    return duration


# Test function for validation
def _test_chunking():
    """Test the chunking function with various inputs."""
    
    test_cases = [
        # Simple single sentence
        "Hello world.",
        
        # Multiple sentences
        "This is the first sentence. This is the second sentence. And this is the third one!",
        
        # Long text with abbreviations
        "Dr. Smith went to the store. He met Mrs. Johnson there. They discussed the U.S. economy.",
        
        # Very long sentence
        "This is a very long sentence that should be split because it exceeds the maximum character limit and we need to ensure it gets broken down properly at appropriate boundaries like commas or semicolons or other punctuation marks.",
        
        # Mixed punctuation
        "What is this? I don't know! Maybe we should find out. Yes, let's do that.",
        
        # Empty and whitespace
        "",
        "   ",
        
        # Unicode and special characters
        "Hello! 你好! Bonjour! ¿Cómo estás?",
    ]
    
    print("=== Text Chunking Tests ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: '{text}'")
        
        chunks = smart_chunk_text(text, max_tokens=20)  # Use small limit for testing
        print(f"Chunks ({len(chunks)}):")
        for j, chunk in enumerate(chunks, 1):
            print(f"  {j}. '{chunk}'")
        
        # Verify completeness
        if text.strip():
            is_complete = verify_chunk_completeness(text, chunks)
            print(f"Completeness check: {'✓ PASS' if is_complete else '✗ FAIL'}")
        
        print()


if __name__ == "__main__":
    _test_chunking()
