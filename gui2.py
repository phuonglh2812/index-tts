import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
import re
import time
from datetime import datetime
import threading

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using basic contraction expansion.")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
                            QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, 
                            QFileDialog, QProgressBar, QCheckBox, QSlider, 
                            QSpinBox, QGroupBox, QComboBox, QMessageBox, QFrame,
                            QTabWidget, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QPixmap

from indextts.infer import IndexTTS

class AudioGenerationWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str, str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, tts_model, reference_audio, text, max_chunk_size, use_fast_mode, output_folder):
        super().__init__()
        self.tts = tts_model
        self.reference_audio = reference_audio
        self.text = text
        self.max_chunk_size = max_chunk_size
        self.use_fast_mode = use_fast_mode
        self.output_folder = output_folder
        
    def is_past_participle(self, word):
        """Check if a word is likely a past participle using NLTK."""
        if not NLTK_AVAILABLE:
            # Fallback: simple check for common past participle endings
            past_participle_endings = ['ed', 'en', 'ne', 'wn', 'un']
            return any(word.lower().endswith(ending) for ending in past_participle_endings)
        
        try:
            # Get synsets for the word
            synsets = wordnet.synsets(word)
            for synset in synsets:
                # Check if any lemma ends with typical past participle patterns
                for lemma in synset.lemmas():
                    lemma_name = lemma.name().lower()
                    if lemma_name.endswith(('ed', 'en', 'ne', 'wn', 'un', 'ied', 'ged')):
                        return True
            
            # Additional check: if the word is tagged as past participle
            tokens = word_tokenize(word)
            if tokens:
                pos_tags = pos_tag(tokens)
                return pos_tags[0][1] in ['VBN']  # VBN = past participle
                
        except Exception:
            pass
            
        # Fallback to simple pattern matching
        past_participle_patterns = [
            r'.*ed$', r'.*en$', r'.*ne$', r'.*wn$', r'.*un$',
            r'.*ied$', r'.*ged$', r'.*ven$', r'.*ken$', r'.*ten$'
        ]
        return any(re.match(pattern, word.lower()) for pattern in past_participle_patterns)

    def determine_d_expansion(self, text, match_start, match_end):
        """Determine if 'd should be 'would' or 'had' based on context."""
        # Get context around the contraction
        context_before = text[max(0, match_start-50):match_start].lower()
        context_after = text[match_end:match_end+50].lower()
        
        # Extract the word that follows 'd
        words_after = re.findall(r'\b\w+\b', context_after)
        next_word = words_after[0] if words_after else ""
        
        # Strong indicators for 'would' (check these FIRST)
        strong_would_indicators = [
            'like', 'love', 'prefer', 'rather', 'want', 'wish', 'hope',
            'choose', 'expect', 'imagine', 'suppose', 'think', 'say'
        ]
        
        # Strong indicators for 'had' (but check for exceptions)
        strong_had_indicators = [
            'already', 'just', 'recently', 'previously', 
            'earlier', 'before', 'once', 'always', 'often', 'sometimes',
            'barely', 'hardly', 'scarcely', 'better'  # "had better"
        ]
        
        # Base form verbs that definitely indicate 'would'
        base_form_verbs = [
            'go', 'come', 'see', 'get', 'make', 'take', 'give', 'know', 
            'think', 'say', 'tell', 'ask', 'try', 'help', 'work', 'play', 
            'run', 'walk', 'drive', 'eat', 'drink', 'buy', 'sell', 'find', 
            'look', 'feel', 'seem', 'become', 'start', 'stop', 'continue', 
            'begin', 'end', 'finish', 'complete', 'be', 'do', 'have'
        ]
        
        # Check for strong 'would' indicators first
        if next_word in strong_would_indicators:
            return "would"
        
        # Check for base form verbs (strong indicator for 'would')
        if next_word in base_form_verbs:
            return "would"
            
        # Check for infinitive patterns (indicates 'would')
        if re.match(r'\s*(rather|sooner|prefer)', context_after):
            return "would"
            
        # Check for "never/ever + verb" patterns - need to distinguish base verb vs past participle
        if next_word in ['never', 'ever']:
            # Get the word after 'never'/'ever'
            if len(words_after) > 1:
                word_after_never = words_after[1]
                
                # Check if it's a past participle first (indicates 'had')
                if self.is_past_participle(word_after_never):
                    return "had"
                
                # If it's a base form verb, then 'would'
                if word_after_never in base_form_verbs:
                    return "would"
                    
                # Also check if it's a base form verb using NLTK
                if NLTK_AVAILABLE:
                    try:
                        tokens = word_tokenize(word_after_never)
                        if tokens:
                            pos_tags = pos_tag(tokens)
                            tag = pos_tags[0][1]
                            if tag == 'VBN':  # Past participle suggests 'had'
                                return "had"
                            elif tag == 'VB':  # Base form verb suggests 'would'
                                return "would"
                    except Exception:
                        pass
        
        # NLTK-based analysis if available
        if NLTK_AVAILABLE and next_word:
            try:
                # Tokenize and tag the context
                full_context = context_before + " " + context_after
                tokens = word_tokenize(full_context)
                pos_tags = pos_tag(tokens)
                
                # Find the position of our next word
                for i, (token, tag) in enumerate(pos_tags):
                    if token.lower() == next_word.lower():
                        # VB (base form verb) suggests 'would'
                        if tag == 'VB':
                            return "would"
                        # VBN (past participle) suggests 'had'
                        elif tag == 'VBN':
                            return "had"
                        # VBP, VBZ (present form) suggests 'would'
                        elif tag in ['VBP', 'VBZ']:
                            return "would"
                        break
            except Exception:
                pass
        
        # Check for strong 'had' indicators after checking 'would' patterns
        if next_word in strong_had_indicators:
            return "had"
        
        # Check if next word is likely a past participle (indicates 'had')
        if next_word and self.is_past_participle(next_word):
            return "had"
        
        # Pattern-based fallback for definite past participles
        past_participle_patterns = [
            r'\s*(been|done|made|taken|given|seen|gone|said|told|asked|tried|helped|worked|played|walked|driven|eaten|drunk|bought|sold|found|looked|felt|seemed|started|stopped|continued|begun|ended|finished|completed|written|spoken|broken|chosen|forgotten|gotten|hidden|ridden|risen|fallen|flown|grown|known|shown|thrown|worn|torn)'
        ]
        
        for pattern in past_participle_patterns:
            if re.match(pattern, context_after):
                return "had"
        
        # Default to 'would' for ambiguous cases (more common in spoken English)
        return "would"

    def expand_contractions(self, text):
        """Expand common English contractions for better TTS pronunciation with intelligent 'd handling."""
        # First, protect possessive 's by temporarily replacing them
        text = re.sub(r'(\w+)\'s\b', r'\1_POSS_MARKER_', text)
        
        # Handle 'd contractions with context analysis BEFORE other contractions
        d_contractions = re.finditer(r'\b(\w+)\'d\b', text, re.IGNORECASE)
        d_replacements = []
        
        for match in d_contractions:
            full_match = match.group(0)
            pronoun = match.group(1)
            start, end = match.span()
            
            # Determine if 'd should be 'would' or 'had'
            expansion_type = self.determine_d_expansion(text, start, end)
            replacement = f"{pronoun} {expansion_type}"
            
            d_replacements.append((start, end, replacement))
            print(f"Context analysis: '{full_match}' → '{replacement}'")
        
        # Apply 'd replacements in reverse order to maintain positions
        for start, end, replacement in reversed(d_replacements):
            text = text[:start] + replacement + text[end:]
        
        # Define other contractions to expand
        contractions = {
            r"won't": "will not",
            r"can't": "cannot", 
            r"shan't": "shall not",
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
            r"'ll": " will", # I'll, you'll, etc.
            r"'ve": " have", # I've, you've, etc.
            r"'m": " am",    # I'm
            r"o'clock": "of the clock",
            r"ma'am": "madam",
            r"ain't": "is not"
        }

        # Special case for pronouns followed by possessive marker
        pronouns_with_is = ["he", "she"]
        for pronoun in pronouns_with_is:
            text = re.sub(rf"{pronoun}_POSS_MARKER_", f"{pronoun} is", text, flags=re.IGNORECASE)

        # Apply other contractions
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        # Restore remaining possessive markers - these are actual possessives, not contractions
        text = text.replace('_POSS_MARKER_', "'s")
        
        return text

    def preprocess_text(self, text):
        """Preprocess text to handle special cases like hyphenated words."""
        # Handle ellipses before expanding contractions
        # Replace ellipses with commas to indicate pauses
        text = re.sub(r'\.{3,}', ',', text)
        
        # Add a comma after question marks for natural pauses
        # Add a comma after '?' only if it's not already followed by a comma or whitespace
        text = re.sub(r'\?(?![\s,])', '?,', text)

        # Replace standalone hyphens with commas
        # A standalone hyphen is one not preceded and not followed by a word character
        # Ensure this regex is correct: (?<!\w)-(?!\w) should work.
        text = re.sub(r'(?<!\w)-(?!\w)', ',', text)
        
        # Remove double quotes but keep the content
        text = text.replace('"', '').replace('"', '')
        
        # Expand contractions (except possessives)
        text = self.expand_contractions(text)
        
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
        
    def process_chunk_with_retry(self, chunk, chunk_index, total_chunks, temp_file_name, max_retries=2):
        """Process a chunk with automatic retry and subdivision if it fails."""
        print(f"[Chunk {chunk_index+1}/{total_chunks}] Length: {len(chunk)} chars, Text: {chunk[:50]}...")
        
        self.progress_signal.emit((chunk_index * 100) // total_chunks, 
                                f"Generating chunk {chunk_index+1}/{total_chunks}: {chunk[:30]}...")
        
        start_time = time.time()
        
        # Try with original chunk first
        try:
            if self.use_fast_mode:
                self.tts.infer_fast(self.reference_audio, chunk, temp_file_name)
            else:
                self.tts.infer(self.reference_audio, chunk, temp_file_name)
                
            # Check if audio was generated successfully
            if os.path.exists(temp_file_name) and os.path.getsize(temp_file_name) > 1000:
                # Get audio length for analytics
                waveform, sample_rate = torchaudio.load(temp_file_name)
                audio_length = waveform.shape[1] / sample_rate
                
                # Check if audio is too short relative to text length (might indicate partial processing)
                chars_per_second = len(chunk) / audio_length if audio_length > 0 else 0
                if chars_per_second > 30 and len(chunk) > 100:  # Threshold: over 30 chars/sec might be too fast
                    print(f"Warning: Chunk {chunk_index+1} has suspiciously high char/sec ratio: {chars_per_second:.2f}")
                    if max_retries > 0:
                        print(f"Retrying with smaller chunks...")
                        return self._subdivide_and_process(chunk, chunk_index, total_chunks, max_retries-1)
                
                process_time = time.time() - start_time
                rtf = process_time / audio_length if audio_length > 0 else 0
                
                # Log detailed stats
                print(f">> Chunk {chunk_index+1} stats: {len(chunk)} chars → {audio_length:.2f} sec audio")
                print(f">> Chars/sec: {chars_per_second:.2f}, RTF: {rtf:.2f}")
                
                return {
                    'success': True,
                    'file': temp_file_name,
                    'chunk_id': chunk_index+1,
                    'chars': len(chunk),
                    'audio_length': audio_length,
                    'rtf': rtf,
                    'char_to_sec_ratio': chars_per_second
                }
            else:
                print(f"Warning: Chunk {chunk_index+1} produced no or invalid audio")
                if max_retries > 0:
                    print(f"Retrying with smaller chunks...")
                    return self._subdivide_and_process(chunk, chunk_index, total_chunks, max_retries-1)
                return {'success': False}
                
        except Exception as e:
            print(f"Error processing chunk {chunk_index+1}: {str(e)}")
            if max_retries > 0:
                print(f"Retrying with smaller chunks...")
                return self._subdivide_and_process(chunk, chunk_index, total_chunks, max_retries-1)
            
            self.error_signal.emit(f"Error processing chunk {chunk_index+1}: {str(e)}")
            return {'success': False}
    
    def _subdivide_and_process(self, chunk, chunk_index, total_chunks, max_retries):
        """Subdivide a chunk and process the smaller pieces."""
        # Split into roughly two parts at a sentence or clause boundary if possible
        # Use a more comprehensive pattern that matches our updated text processing
        splits = re.split(r'(?<=[.!?])\s+|(?<=[,;:])\s+', chunk)
        
        if len(splits) <= 1:
            # If no good split points, just divide in half
            mid = len(chunk) // 2
            # Try to find a space near the middle
            space_pos = chunk.find(' ', mid-10, mid+10)
            if space_pos != -1:
                splits = [chunk[:space_pos], chunk[space_pos+1:]]
            else:
                splits = [chunk[:mid], chunk[mid:]]
        
        print(f"Subdivided chunk {chunk_index+1} into {len(splits)} smaller chunks")
        
        # Merge very short splits with adjacent ones
        merged_splits = []
        i = 0
        while i < len(splits):
            current = splits[i].strip()
            
            # Skip empty splits
            if not current:
                i += 1
                continue
                
            # If this is a very short phrase and not the last one
            if len(current) <= 5 and i < len(splits) - 1:
                # Merge with the next split
                next_split = splits[i+1].strip()
                if next_split:  # Make sure next split is not empty
                    merged_splits.append(f"{current} {next_split}")
                    i += 2  # Skip both splits
                else:
                    merged_splits.append(current)
                    i += 1
            else:
                merged_splits.append(current)
                i += 1
                
        # Use the improved splits
        splits = [s for s in merged_splits if s.strip()]
        
        results = []
        for i, subchunk in enumerate(splits):
            if not subchunk.strip():
                continue
                
            subchunk = subchunk.strip()
            print(f"  Processing subchunk {i+1}/{len(splits)}: {subchunk[:30]}...")
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            
            result = self.process_chunk_with_retry(subchunk, 
                                                chunk_index + (i/100), # Use decimal to maintain ordering
                                                total_chunks,
                                                temp_file.name, 
                                                max_retries-1)
            
            if result.get('success'):
                results.append(result)
        
        if not results:
            return {'success': False}
            
        # Combine the subchunk audio files into one
        combined_temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        combined_temp_file.close()
        
        # Concatenate all waveforms from successful subchunks
        waveforms = []
        sample_rate = 0
        for result in results:
            waveform, sr = torchaudio.load(result['file'])
            waveforms.append(waveform)
            sample_rate = sr
            
        # Concatenate and save
        if waveforms:
            concatenated = torch.cat(waveforms, dim=1)
            torchaudio.save(combined_temp_file.name, concatenated, sample_rate)
            
            # Calculate combined stats
            total_audio_length = sum(result['audio_length'] for result in results)
            avg_rtf = sum(result['rtf'] for result in results) / len(results)
            
            # Clean up individual subchunk files
            for result in results:
                try:
                    os.remove(result['file'])
                except:
                    pass
            
            return {
                'success': True,
                'file': combined_temp_file.name,
                'chunk_id': chunk_index+1,
                'chars': len(chunk),
                'audio_length': total_audio_length,
                'rtf': avg_rtf,
                'char_to_sec_ratio': len(chunk) / total_audio_length if total_audio_length > 0 else 0,
                'subdivided': True
            }
        
        return {'success': False}

    def run(self):
        try:
            # Create a timestamp for the output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_folder, f"generated_speech_{timestamp}.wav")
            
            # Save the output file path as an attribute so it can be accessed
            self.output_file = output_file
            
            # Process text in chunks for better handling of long scripts
            chunks = self.split_text_into_chunks(self.text)
            total_chunks = len(chunks)
            
            # Update progress
            self.progress_signal.emit(0, f"Processing text in {total_chunks} chunks...")
            
            # Generate audio for each chunk
            temp_files = []
            chunk_stats = []  # Store stats for analysis
            
            for i, chunk in enumerate(chunks):
                # Create a temporary file for each chunk output
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.close()
                
                # Process chunk with retry mechanism
                result = self.process_chunk_with_retry(chunk, i, total_chunks, temp_file.name)
                
                if result.get('success'):
                    temp_files.append(result['file'])
                    chunk_stats.append(result)
                    
                    # Report if this chunk was subdivided
                    if result.get('subdivided'):
                        print(f"Chunk {i+1}/{total_chunks} was successfully processed after subdivision")
            
            # Verify we have files to concatenate
            if not temp_files:
                raise Exception("No audio chunks were successfully generated.")
                
            # Print analytics summary
            print("\n=== Chunk Processing Summary ===")
            avg_rtf = sum(stat['rtf'] for stat in chunk_stats) / len(chunk_stats) if chunk_stats else 0
            avg_char_ratio = sum(stat['char_to_sec_ratio'] for stat in chunk_stats) / len(chunk_stats) if chunk_stats else 0
            print(f"Successfully processed {len(chunk_stats)}/{total_chunks} chunks")
            print(f"Average RTF: {avg_rtf:.2f}")
            print(f"Average chars/sec: {avg_char_ratio:.2f}")
            
            # Find outliers
            rtf_threshold = avg_rtf * 1.5
            outliers = [s for s in chunk_stats if s['rtf'] > rtf_threshold]
            if outliers:
                print(f"Found {len(outliers)} problematic chunks with higher than normal RTF:")
                for o in outliers:
                    print(f"  Chunk {o['chunk_id']}: RTF {o['rtf']:.2f} (avg: {avg_rtf:.2f})")
                
            # Concatenate all audio chunks
            self.progress_signal.emit(90, "Concatenating audio chunks...")
            
            if len(temp_files) == 1:
                # If only one chunk, just copy the file
                waveform, sample_rate = torchaudio.load(temp_files[0])
                torchaudio.save(output_file, waveform, sample_rate)
            else:
                # If multiple chunks, concatenate them
                waveforms = []
                for file in temp_files:
                    waveform, sample_rate = torchaudio.load(file)
                    waveforms.append(waveform)
                    
                # Concatenate all waveforms
                concatenated = torch.cat(waveforms, dim=1)
                
                # Save the concatenated audio
                torchaudio.save(output_file, concatenated, sample_rate)
                
            # Clean up temporary chunk files
            for file in temp_files:
                try:
                    os.remove(file)
                except:
                    pass
                    
            self.progress_signal.emit(100, "Audio generation complete!")
            self.finished_signal.emit(output_file, f"Generated speech saved to {output_file}")
            
        except Exception as e:
            self.error_signal.emit(f"Error generating speech: {str(e)}")

class BatchProcessingWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    file_progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, tts_model, reference_audio, text_files, max_chunk_size, use_fast_mode, output_folder):
        super().__init__()
        self.tts = tts_model
        self.reference_audio = reference_audio
        self.text_files = text_files
        self.max_chunk_size = max_chunk_size
        self.use_fast_mode = use_fast_mode
        self.output_folder = output_folder
        
    def run(self):
        try:
            total_files = len(self.text_files)
            
            for i, text_file in enumerate(self.text_files):
                try:
                    # Update overall progress
                    self.progress_signal.emit(
                        int((i / total_files) * 100),
                        f"Processing file {i+1}/{total_files}: {os.path.basename(text_file)}"
                    )
                    
                    # Read text from file
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Skip empty files
                    if not text_content.strip():
                        self.file_progress_signal.emit(100, f"Skipped empty file: {os.path.basename(text_file)}")
                        continue
                    
                    # Create output filename based on input filename
                    base_filename = os.path.splitext(os.path.basename(text_file))[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(self.output_folder, f"{base_filename}_{timestamp}.wav")
                    
                    # Create audio worker for this file
                    audio_worker = AudioGenerationWorker(
                        self.tts,
                        self.reference_audio,
                        text_content,
                        self.max_chunk_size,
                        self.use_fast_mode,
                        self.output_folder
                    )
                    
                    # Connect signals to relay progress for current file
                    audio_worker.progress_signal.connect(self.relay_file_progress)
                    
                    # Process the file synchronously
                    audio_worker.run()  # Direct call instead of starting thread
                    
                    # Get the generated file path
                    if hasattr(audio_worker, 'output_file'):
                        # Rename to our desired output name
                        os.rename(audio_worker.output_file, output_file)
                        self.file_progress_signal.emit(100, f"Completed: {os.path.basename(text_file)} → {output_file}")
                    
                except Exception as e:
                    self.error_signal.emit(f"Error processing file {os.path.basename(text_file)}: {str(e)}")
            
            # Final progress update
            self.progress_signal.emit(100, f"Batch processing complete! Processed {total_files} files.")
            self.finished_signal.emit(self.output_folder)
            
        except Exception as e:
            self.error_signal.emit(f"Batch processing error: {str(e)}")
    
    def relay_file_progress(self, value, message):
        """Relay progress updates from the audio worker"""
        self.file_progress_signal.emit(value, message)

class IndexTTSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize TTS model
        self.initialize_tts_model()
        
        # Setup UI
        self.initUI()
        
    def initialize_tts_model(self):
        try:
            model_dir = "checkpoints"
            config_path = os.path.join(model_dir, "config.yaml")
            self.tts = IndexTTS(cfg_path=config_path, model_dir=model_dir)
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            self.tts = None
            QMessageBox.critical(self, "Model Loading Error", 
                               f"Failed to load IndexTTS model: {str(e)}\n\nPlease make sure the model files are in the 'checkpoints' directory.")
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("IndexTTS Voice Generator")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title and description
        title_label = QLabel("IndexTTS Voice Generator")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle_label = QLabel("Generate speech from text using zero-shot voice cloning")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add logo or separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Holder for reference audio
        self.reference_audio_path = ""
        
        # Reference voice section (common for both tabs)
        reference_layout = QHBoxLayout()
        self.reference_label = QLabel("Reference Voice:")
        self.reference_path = QLineEdit()
        self.reference_path.setReadOnly(True)
        self.reference_path.setPlaceholderText("No file selected")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_reference_audio)
        
        reference_layout.addWidget(self.reference_label)
        reference_layout.addWidget(self.reference_path, 1)
        reference_layout.addWidget(self.browse_button)
        
        # Create tab widget for single/batch processing
        self.tab_widget = QTabWidget()
        
        # Create single processing tab
        self.single_tab = QWidget()
        self.create_single_tab()
        self.tab_widget.addTab(self.single_tab, "Single Text")
        
        # Create batch processing tab
        self.batch_tab = QWidget()
        self.create_batch_tab()
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")
        
        # Output folder (common for both tabs)
        output_group = QGroupBox("Output Settings")
        output_layout = QHBoxLayout(output_group)
        
        output_label = QLabel("Output Folder:")
        self.output_folder = QLineEdit()
        self.output_folder.setReadOnly(True)
        self.output_folder.setText(os.path.abspath("output"))
        
        if not os.path.exists("output"):
            os.makedirs("output")
            
        self.output_browse = QPushButton("Browse")
        self.output_browse.clicked.connect(self.browse_output_folder)
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_folder, 1)
        output_layout.addWidget(self.output_browse)
        
        # Add all components to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addWidget(separator)
        main_layout.addLayout(reference_layout)
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(output_group)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
        # Style the application
        self.apply_styles()
        
        # Disable buttons if model not loaded
        if not self.model_loaded:
            self.single_generate_button.setEnabled(False)
            self.single_generate_button.setText("Model Not Loaded")
            self.batch_generate_button.setEnabled(False)
            self.batch_generate_button.setText("Model Not Loaded")
            self.single_status_label.setText("Error: IndexTTS model not loaded")
    
    def create_single_tab(self):
        """Create UI for single text processing tab"""
        single_layout = QVBoxLayout(self.single_tab)
        
        # Text input
        text_label = QLabel("Text to Speak:")
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter the text you want to convert to speech...")
        self.text_input.setMinimumHeight(100)
        
        # Options section
        options_group = QGroupBox("Generation Options")
        options_layout = QHBoxLayout(options_group)
        
        # Chunk size option
        chunk_layout = QHBoxLayout()
        chunk_label = QLabel("Max Chunk Size:")
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(50, 500)
        self.chunk_size.setValue(250)
        self.chunk_size.setSuffix(" characters")
        chunk_layout.addWidget(chunk_label)
        chunk_layout.addWidget(self.chunk_size)
        
        # Mode option
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Generation Mode:")
        self.generation_mode = QComboBox()
        self.generation_mode.addItem("Fast Mode (Better Performance)")
        self.generation_mode.addItem("Standard Mode (Higher Quality)")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.generation_mode)
        
        # Add options to layout
        options_layout.addLayout(chunk_layout)
        options_layout.addLayout(mode_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.single_generate_button = QPushButton("Generate Speech")
        self.single_generate_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.single_generate_button.setMinimumHeight(50)
        self.single_generate_button.clicked.connect(self.generate_speech)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMinimumHeight(50)
        self.clear_button.clicked.connect(self.clear_fields)
        
        button_layout.addWidget(self.single_generate_button, 2)
        button_layout.addWidget(self.clear_button, 1)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.single_progress_bar = QProgressBar()
        self.single_progress_bar.setValue(0)
        
        self.single_status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.single_progress_bar)
        progress_layout.addWidget(self.single_status_label)
        
        # Output section
        output_group = QGroupBox("Output")
        output_files_layout = QVBoxLayout(output_group)
        
        self.output_file_label = QLabel("Generated File:")
        self.output_file_path = QLineEdit()
        self.output_file_path.setReadOnly(True)
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        
        output_file_row = QHBoxLayout()
        output_file_row.addWidget(self.output_file_label)
        output_file_row.addWidget(self.output_file_path, 1)
        output_file_row.addWidget(self.play_button)
        
        output_files_layout.addLayout(output_file_row)
        
        # Add all components to single tab layout
        single_layout.addWidget(text_label)
        single_layout.addWidget(self.text_input)
        single_layout.addWidget(options_group)
        single_layout.addLayout(button_layout)
        single_layout.addWidget(progress_group)
        single_layout.addWidget(output_group)
    
    def create_batch_tab(self):
        """Create UI for batch text processing tab"""
        batch_layout = QVBoxLayout(self.batch_tab)
        
        # File list section
        files_group = QGroupBox("Text Files for Batch Processing")
        files_layout = QVBoxLayout(files_group)
        
        # File list
        self.batch_files_list = QListWidget()
        self.batch_files_list.setMinimumHeight(150)
        
        # Buttons for file list management
        file_buttons_layout = QHBoxLayout()
        
        self.add_files_button = QPushButton("Add Files")
        self.add_files_button.clicked.connect(self.add_batch_files)
        
        self.remove_file_button = QPushButton("Remove Selected")
        self.remove_file_button.clicked.connect(self.remove_batch_file)
        
        self.clear_files_button = QPushButton("Clear All")
        self.clear_files_button.clicked.connect(self.clear_batch_files)
        
        file_buttons_layout.addWidget(self.add_files_button)
        file_buttons_layout.addWidget(self.remove_file_button)
        file_buttons_layout.addWidget(self.clear_files_button)
        
        files_layout.addWidget(self.batch_files_list)
        files_layout.addLayout(file_buttons_layout)
        
        # Batch options
        batch_options_group = QGroupBox("Batch Generation Options")
        batch_options_layout = QHBoxLayout(batch_options_group)
        
        # Chunk size option
        batch_chunk_layout = QHBoxLayout()
        batch_chunk_label = QLabel("Max Chunk Size:")
        self.batch_chunk_size = QSpinBox()
        self.batch_chunk_size.setRange(50, 500)
        self.batch_chunk_size.setValue(250)
        self.batch_chunk_size.setSuffix(" characters")
        batch_chunk_layout.addWidget(batch_chunk_label)
        batch_chunk_layout.addWidget(self.batch_chunk_size)
        
        # Mode option
        batch_mode_layout = QHBoxLayout()
        batch_mode_label = QLabel("Generation Mode:")
        self.batch_generation_mode = QComboBox()
        self.batch_generation_mode.addItem("Fast Mode (Better Performance)")
        self.batch_generation_mode.addItem("Standard Mode (Higher Quality)")
        batch_mode_layout.addWidget(batch_mode_label)
        batch_mode_layout.addWidget(self.batch_generation_mode)
        
        # Add options to layout
        batch_options_layout.addLayout(batch_chunk_layout)
        batch_options_layout.addLayout(batch_mode_layout)
        
        # Generate button
        self.batch_generate_button = QPushButton("Start Batch Processing")
        self.batch_generate_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.batch_generate_button.setMinimumHeight(50)
        self.batch_generate_button.clicked.connect(self.start_batch_processing)
        
        # Progress section
        batch_progress_group = QGroupBox("Batch Progress")
        batch_progress_layout = QVBoxLayout(batch_progress_group)
        
        # Overall progress
        overall_label = QLabel("Overall Progress:")
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setValue(0)
        
        # Current file progress
        current_file_label = QLabel("Current File Progress:")
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setValue(0)
        
        self.batch_status_label = QLabel("Ready")
        
        batch_progress_layout.addWidget(overall_label)
        batch_progress_layout.addWidget(self.batch_progress_bar)
        batch_progress_layout.addWidget(current_file_label)
        batch_progress_layout.addWidget(self.file_progress_bar)
        batch_progress_layout.addWidget(self.batch_status_label)
        
        # Add all components to batch tab layout
        batch_layout.addWidget(files_group)
        batch_layout.addWidget(batch_options_group)
        batch_layout.addWidget(self.batch_generate_button)
        batch_layout.addWidget(batch_progress_group)
    
    def browse_reference_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)"
        )
        if file_path:
            self.reference_audio_path = file_path
            self.reference_path.setText(file_path)
            
    def browse_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_folder.text()
        )
        if folder_path:
            self.output_folder.setText(folder_path)
            
    def clear_fields(self):
        self.reference_audio_path = ""
        self.reference_path.setText("")
        self.text_input.clear()
        self.single_progress_bar.setValue(0)
        self.single_status_label.setText("Ready")
        self.output_file_path.clear()
        self.play_button.setEnabled(False)
        
    def generate_speech(self):
        # Check if model is loaded
        if not self.model_loaded:
            QMessageBox.critical(self, "Error", "IndexTTS model is not loaded. Cannot generate speech.")
            return
            
        # Validate inputs
        if not self.reference_audio_path:
            QMessageBox.warning(self, "Input Error", "Please select a reference audio file.")
            return
            
        if not self.text_input.toPlainText().strip():
            QMessageBox.warning(self, "Input Error", "Please enter some text to generate speech.")
            return
            
        # Get parameters
        use_fast_mode = self.generation_mode.currentIndex() == 0
        max_chunk_size = self.chunk_size.value()
        output_folder = self.output_folder.text()
        
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create output folder: {str(e)}")
                return
                
        # Disable UI controls during generation
        self.set_ui_enabled(False)
        
        # Start the worker thread for audio generation
        self.generation_thread = AudioGenerationWorker(
            self.tts, 
            self.reference_audio_path,
            self.text_input.toPlainText(),
            max_chunk_size,
            use_fast_mode,
            output_folder
        )
        
        # Connect signals
        self.generation_thread.progress_signal.connect(self.update_progress)
        self.generation_thread.finished_signal.connect(self.generation_finished)
        self.generation_thread.error_signal.connect(self.generation_error)
        
        # Start generation
        self.generation_thread.start()
        
    def set_ui_enabled(self, enabled):
        """Enable or disable UI controls during processing"""
        self.single_generate_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.chunk_size.setEnabled(enabled)
        self.generation_mode.setEnabled(enabled)
        
        if not enabled:
            self.single_generate_button.setText("Generating...")
        else:
            self.single_generate_button.setText("Generate Speech")
        
    def update_progress(self, value, message):
        """Update progress bar and status label"""
        self.single_progress_bar.setValue(value)
        self.single_status_label.setText(message)
        
    def generation_finished(self, output_file, message):
        """Handle successful generation completion"""
        self.output_file_path.setText(output_file)
        self.single_status_label.setText(message)
        self.play_button.setEnabled(True)
        self.set_ui_enabled(True)
        
        # Show notification
        QMessageBox.information(self, "Success", "Speech generation completed successfully!")
        
    def generation_error(self, error_message):
        """Handle generation error"""
        self.single_status_label.setText(error_message)
        self.set_ui_enabled(True)
        
        # Show error message
        QMessageBox.critical(self, "Error", error_message)
        
    def play_audio(self):
        """Play the generated audio file"""
        audio_path = self.output_file_path.text()
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.warning(self, "Error", "No valid audio file to play.")
            return
            
        # Use system's default audio player to play the file
        if sys.platform == "win32":
            os.startfile(audio_path)
        elif sys.platform == "darwin":  # macOS
            os.system(f"open {audio_path}")
        else:  # Linux
            os.system(f"xdg-open {audio_path}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Check if a generation is in progress
        if hasattr(self, 'generation_thread') and self.generation_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                "Audio generation is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Try to terminate the thread
                self.generation_thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def add_batch_files(self):
        """Add text files to the batch processing list"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Text Files", "", "Text Files (*.txt)"
        )
        
        if file_paths:
            # Add files to the list
            for file_path in file_paths:
                # Check if file is already in the list
                existing_items = [self.batch_files_list.item(i).data(Qt.ItemDataRole.UserRole) 
                                for i in range(self.batch_files_list.count())]
                
                if file_path not in existing_items:
                    item = QListWidgetItem(os.path.basename(file_path))
                    item.setData(Qt.ItemDataRole.UserRole, file_path)
                    self.batch_files_list.addItem(item)
    
    def remove_batch_file(self):
        """Remove selected file from the batch list"""
        selected_items = self.batch_files_list.selectedItems()
        for item in selected_items:
            self.batch_files_list.takeItem(self.batch_files_list.row(item))
    
    def clear_batch_files(self):
        """Clear all files from the batch list"""
        self.batch_files_list.clear()
    
    def start_batch_processing(self):
        """Start batch processing of text files"""
        # Check if model is loaded
        if not self.model_loaded:
            QMessageBox.critical(self, "Error", "IndexTTS model is not loaded. Cannot generate speech.")
            return
            
        # Validate inputs
        if not self.reference_audio_path:
            QMessageBox.warning(self, "Input Error", "Please select a reference audio file.")
            return
            
        if self.batch_files_list.count() == 0:
            QMessageBox.warning(self, "Input Error", "Please add text files for batch processing.")
            return
            
        # Get list of text files
        text_files = []
        for i in range(self.batch_files_list.count()):
            file_path = self.batch_files_list.item(i).data(Qt.ItemDataRole.UserRole)
            text_files.append(file_path)
        
        # Get parameters
        use_fast_mode = self.batch_generation_mode.currentIndex() == 0
        max_chunk_size = self.batch_chunk_size.value()
        output_folder = self.output_folder.text()
        
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create output folder: {str(e)}")
                return
                
        # Disable UI controls during batch processing
        self.set_batch_ui_enabled(False)
        
        # Start the worker thread for batch processing
        self.batch_thread = BatchProcessingWorker(
            self.tts, 
            self.reference_audio_path,
            text_files,
            max_chunk_size,
            use_fast_mode,
            output_folder
        )
        
        # Connect signals
        self.batch_thread.progress_signal.connect(self.update_batch_progress)
        self.batch_thread.file_progress_signal.connect(self.update_file_progress)
        self.batch_thread.finished_signal.connect(self.batch_processing_finished)
        self.batch_thread.error_signal.connect(self.batch_processing_error)
        
        # Start batch processing
        self.batch_thread.start()
    
    def set_batch_ui_enabled(self, enabled):
        """Enable or disable UI controls during batch processing"""
        self.batch_generate_button.setEnabled(enabled)
        self.add_files_button.setEnabled(enabled)
        self.remove_file_button.setEnabled(enabled)
        self.clear_files_button.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)
        self.batch_chunk_size.setEnabled(enabled)
        self.batch_generation_mode.setEnabled(enabled)
        
        if not enabled:
            self.batch_generate_button.setText("Processing...")
        else:
            self.batch_generate_button.setText("Start Batch Processing")
    
    def update_batch_progress(self, value, message):
        """Update overall batch progress"""
        self.batch_progress_bar.setValue(value)
        self.batch_status_label.setText(message)
    
    def update_file_progress(self, value, message):
        """Update current file progress"""
        self.file_progress_bar.setValue(value)
        self.batch_status_label.setText(message)
    
    def batch_processing_finished(self, output_folder):
        """Handle successful batch processing completion"""
        self.batch_status_label.setText(f"Batch processing completed. Files saved to {output_folder}")
        self.set_batch_ui_enabled(True)
        
        # Show notification
        QMessageBox.information(self, "Success", "Batch processing completed successfully!")
    
    def batch_processing_error(self, error_message):
        """Handle batch processing error"""
        self.batch_status_label.setText(error_message)
        self.set_batch_ui_enabled(True)
        
        # Show error message
        QMessageBox.critical(self, "Error", error_message)
    
    def apply_styles(self):
        # Apply custom styles for a modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            #clear_button {
                background-color: #6c757d;
            }
            #clear_button:hover {
                background-color: #5a6268;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                width: 10px;
                margin: 0.5px;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IndexTTSApp()
    window.show()
    sys.exit(app.exec())
