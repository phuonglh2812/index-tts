import re
import sys
import os
import tempfile

# Implement the key text processing methods directly instead of importing
class TextProcessor:
    def __init__(self):
        self.max_chunk_size = 250
    
    def expand_contractions(self, text):
        """Expand common English contractions for better TTS pronunciation."""
        # print(f"  Input to expand_contractions: {text}") # Debugging
        
        # First, protect possessive 's by temporarily replacing them
        # This is the critical step - mark all apostrophe-s as potential possessives FIRST
        # We'll mark all cases and then selectively handle the known contraction cases later
        text = re.sub(r'(\w+)\'s\b', r'\1_POSS_MARKER_', text)
        # print(f"  After marking possessives: {text}") # Debugging
        
        # Define actual contractions to expand - using the marker we just set
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",  # don't, doesn't, wouldn't, etc.
            r"'re": " are",  # we're, they're, etc.
            r"it_POSS_MARKER_": "it is",   # Special case for "it's" which is always "it is"
            r"that_POSS_MARKER_": "that is",  # Special case for "that's" 
            r"what_POSS_MARKER_": "what is",  # Special case for "what's"
            r"who_POSS_MARKER_": "who is",    # Special case for "who's" 
            r"how_POSS_MARKER_": "how is",    # Special case for "how's"
            r"there_POSS_MARKER_": "there is",  # Special case for "there's"
            r"where_POSS_MARKER_": "where is",  # Special case for "where's"
            r"when_POSS_MARKER_": "when is",    # Special case for "when's"
            r"here_POSS_MARKER_": "here is",    # Special case for "here's"
            r"'d": " would", # I'd, you'd, etc.
            r"'ll": " will", # I'll, you'll, etc.
            r"'ve": " have", # I've, you've, etc.
            r"'m": " am",    # I'm
            r"o'clock": "of the clock",
            r"ma'am": "madam",
            r"ain't": "is not"
        }

        # Special case for 'he', 'she' followed by possessive marker
        pronouns_with_is = ["he", "she"]
        for pronoun in pronouns_with_is:
            text = re.sub(rf"{pronoun}_POSS_MARKER_", f"{pronoun} is", text, flags=re.IGNORECASE)
        
        # print(f"  After pronoun handling: {text}") # Debugging

        for contraction, expansion in contractions.items():
            # Look for contractions with word boundaries
            # original = text # Debugging
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
            # if original != text: # Debugging
            #     print(f"  Applied: {contraction} -> {expansion}") # Debugging
            #     print(f"  Result: {text}") # Debugging
        
        # Restore remaining possessive markers - these are actual possessives, not contractions
        text = text.replace('_POSS_MARKER_', "'s")
        # print(f"  Final after restoring possessives: {text}") # Debugging
        
        return text

    def preprocess_text(self, text):
        """Preprocess text to handle special cases like hyphenated words."""
        # Handle ellipses before expanding contractions
        # Replace ellipses with commas to indicate pauses
        text = re.sub(r'\.{3,}', ',', text)
        
        # Add a comma after question marks for natural pauses
        # Only add if not already followed by a comma or space or end of string
        text = re.sub(r'\?(?![,\s]|$)', ',?', text)

        # Replace standalone hyphens with commas
        # A standalone hyphen is one not preceded and not followed by a word character
        text = re.sub(r'(?<!\w)-(?!\w)', ',', text)
        
        # Remove double quotes but keep the content
        text = text.replace('"', '').replace('"', '')
        
        # Expand contractions (except possessives)
        text = self.expand_contractions(text)
        
        # The previous hyphen handling logic for compound words is no longer needed
        # as the regex above specifically targets standalone hyphens.
        
        # Remove any unusual Unicode characters that might cause issues
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle quotes and brackets consistently - keep single quotes for contractions
        text = text.replace(''', "'").replace(''', "'")
        
        # Expand common abbreviations
        abbr_dict = {
            r"\bMr\.": "Mister",
            r"\bMrs\.": "Misses",
            r"\bDr\.": "Doctor",
            r"\bPh\.D\.": "P H D",  # spacing out letters for better pronunciation
            r"\be\.g\.": "for example",
            r"\bi\.e\.": "that is",
            r"\betc\.": "etcetera",
            r"\bvs\.": "versus",
            r"\bapprox\.": "approximately",
        }
        
        for abbr, expansion in abbr_dict.items():
            text = re.sub(abbr, expansion, text)
        
        return text.strip()
        
    def split_text_into_chunks(self, text):
        """Split text into proper chunks for better TTS processing."""
        # Preprocess text to handle special cases
        text = self.preprocess_text(text)
        
        # Define sentence boundaries - improved pattern to handle various cases
        # This handles sentence endings even when there's no space after punctuation
        sentence_end_patterns = r'(?<=[.!?。！？])\s*|(?<=[\.\?!。！？][\'"])\s+'
        
        # Split by sentence boundaries
        sentences = re.split(sentence_end_patterns, text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s and s.strip()]
        
        # Merge very short sentences with adjacent ones
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            
            # If this is a very short sentence (like "No!" or "Yes!")
            if len(current) <= 5:
                # If not the last sentence, merge with the next one
                if i < len(sentences) - 1:
                    merged_sentences.append(f"{current} {sentences[i+1]}")
                    i += 2  # Skip both sentences
                # If it's the last sentence and there are previous ones, merge with the previous
                elif i > 0 and merged_sentences:
                    merged_sentences[-1] = f"{merged_sentences[-1]} {current}"
                    i += 1
                # If it's the only sentence, keep it
                else:
                    merged_sentences.append(current)
                    i += 1
            else:
                merged_sentences.append(current)
                i += 1
        
        sentences = merged_sentences
        
        # Calculate optimal chunk size - smaller chunks for complex text
        # Start with smaller chunks (around 150-200 chars) instead of maxing out at 250
        target_chunk_size = min(200, self.max_chunk_size)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence:
                continue
                
            # Estimate sentence complexity (presence of numbers, special chars, etc.)
            complexity = sum(1 for c in sentence if c.isdigit() or c in '/@#$%^&*(){}[]<>\\|')
            
            # Adjust chunk size dynamically based on content complexity
            effective_chunk_size = target_chunk_size - (complexity * 5)  # Reduce chunk size for complex text
            effective_chunk_size = max(100, effective_chunk_size)  # Don't go below 100 chars
                
            # If adding this sentence would make chunk too long, start a new chunk
            if len(current_chunk) + len(sentence) > effective_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    # Add proper spacing between sentences
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        # Make sure no chunks are too short or empty
        valid_chunks = []
        for chunk in chunks:
            if len(chunk) < 5:  # Arbitrary minimum length
                print(f"Warning: Very short chunk detected: '{chunk}'")
                # Try to merge with another chunk if possible
                if valid_chunks:
                    valid_chunks[-1] += " " + chunk
                else:
                    valid_chunks.append(chunk)
            else:
                valid_chunks.append(chunk)
                
        return valid_chunks

# Initialize test processor
processor = TextProcessor()

# Define output file path
output_file_path = os.path.join(tempfile.gettempdir(), "text_processing_test_output.txt")

# Redirect stdout to file
sys.stdout = open(output_file_path, 'w', encoding='utf-8')

# Test possessive 's handling
print("=== Testing possessive 's handling ===")
test_texts = [
    "Jessica's future is bright.",
    "The house's color is blue.",
    "She's going to the store.",
    "It's a beautiful day.",
    "He's running fast.",
    "That's my book.",
    "The cat's toy is broken.",
    "The boss's office is upstairs.",
    "The business's new logo",
    "The church's bell",
    "The mother-in-law's house",
    "The Jones's house is on the corner."
]

for text in test_texts:
    # print(f"\nTesting: {text}") # Debugging
    processed = processor.preprocess_text(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    print()

# Test ellipses handling
print("\n=== Testing ellipses handling ===")
test_ellipses = [
    "But the house... Jessica's future...",
    "I wonder... what could happen next?",
    "She said... \"I'm not going!\"...",
    "Wait for it..."
]

for text in test_ellipses:
    processed = processor.preprocess_text(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    print()

# Test hyphen handling
print("\n=== Testing hyphen handling ===")
test_hyphens = [
    "This is a test-case.",
    "A long-awaited moment.",
    "Editor-in-chief.",
    "This is a standalone - hyphen.",
    "Another example - here."
]

for text in test_hyphens:
    processed = processor.preprocess_text(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    print()

# Test question mark handling
print("\n=== Testing question mark handling ===")
test_questions = [
    "How are you?",
    "What happened next?",
    "Is this working? Yes.",
    "Why, hello there? Isn't it a lovely day?"
]

for text in test_questions:
    processed = processor.preprocess_text(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    print()

# Test short sentence merging
print("\n=== Testing short sentence handling ===")
test_short = [
    "No! We must act now.",
    "Yes! I agree with you.",
    "No! No! No! Stop that right now.",
    "Go! Run! Hide! They're coming.",
    "Hi! How are you doing today?",
    "Oh! That's interesting.",
    "We need to go. Now!"  # Test short sentence at end
]

for text in test_short:
    chunks = processor.split_text_into_chunks(text)
    print(f"Original: {text}")
    print(f"Chunks: {chunks}")
    print()

print("All tests completed.")

# Restore stdout
sys.stdout.close()
sys.stdout = sys.__stdout__

print(f"Test output written to {output_file_path}") 