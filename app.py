import os
import sys
import torch
import torchaudio
import numpy as np
import tempfile
import uuid
import time
import json
import sqlite3
import threading
import re
import queue
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from indextts.infer import IndexTTS

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

# Configure application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['DATABASE'] = 'tts_jobs.db'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Job queue for task processing
job_queue = queue.Queue()
is_worker_running = False
queue_lock = threading.Lock()

# Job status tracking
active_jobs = {}

# Initialize TTS model
try:
    model_dir = "checkpoints"
    config_path = os.path.join(model_dir, "config.yaml")
    tts = IndexTTS(cfg_path=config_path, model_dir=model_dir)
    model_loaded = True
except Exception as e:
    print(f"Failed to load IndexTTS model: {str(e)}")
    model_loaded = False
    tts = None

# Database setup
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Create jobs table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS jobs
    (id TEXT PRIMARY KEY, 
    job_type TEXT, 
    status TEXT, 
    created_at TEXT,
    completed_at TEXT,
    reference_audio TEXT,
    output_files TEXT,
    error_message TEXT)
    ''')
    
    # Check if queue_position column exists, add it if it doesn't
    c.execute("PRAGMA table_info(jobs)")
    columns = [column[1] for column in c.fetchall()]
    if 'queue_position' not in columns:
        print("Adding queue_position column to jobs table")
        c.execute("ALTER TABLE jobs ADD COLUMN queue_position INTEGER")
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def save_job(job_id, job_type, status, reference_audio, output_files=None, error_message=None, queue_position=None):
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    completed_at = now if status == 'completed' or status == 'failed' else None
    
    # Convert output_files list to JSON string if provided
    output_files_json = json.dumps(output_files) if output_files else None
    
    # Check if job already exists
    c.execute("SELECT id FROM jobs WHERE id = ?", (job_id,))
    if c.fetchone():
        if queue_position is not None:
            c.execute('''
            UPDATE jobs SET
            status = ?, 
            completed_at = ?,
            output_files = ?,
            error_message = ?,
            queue_position = ?
            WHERE id = ?
            ''', (status, completed_at, output_files_json, error_message, queue_position, job_id))
        else:
            c.execute('''
            UPDATE jobs SET
            status = ?, 
            completed_at = ?,
            output_files = ?,
            error_message = ?
            WHERE id = ?
            ''', (status, completed_at, output_files_json, error_message, job_id))
    else:
        c.execute('''
        INSERT INTO jobs 
        (id, job_type, status, created_at, completed_at, reference_audio, output_files, error_message, queue_position) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, job_type, status, now, completed_at, reference_audio, output_files_json, error_message, queue_position))
    
    conn.commit()
    conn.close()

def update_queue_positions():
    """Update all jobs in queue to reflect their current positions"""
    with queue_lock:
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        
        # Get all queued jobs
        c.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC")
        queued_jobs = c.fetchall()
        
        # Update positions
        for position, (job_id,) in enumerate(queued_jobs):
            c.execute("UPDATE jobs SET queue_position = ? WHERE id = ?", (position + 1, job_id))
            if job_id in active_jobs:
                active_jobs[job_id]['queue_position'] = position + 1
                active_jobs[job_id]['status_message'] = f"Waiting in queue (position {position + 1})"
        
        conn.commit()
        conn.close()

def get_jobs():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    jobs = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Parse JSON fields
    for job in jobs:
        if job['output_files']:
            job['output_files'] = json.loads(job['output_files'])
        else:
            job['output_files'] = []
            
    return jobs

def get_job(job_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job = c.fetchone()
    conn.close()
    
    if job:
        job_dict = dict(job)
        if job_dict['output_files']:
            job_dict['output_files'] = json.loads(job_dict['output_files'])
        else:
            job_dict['output_files'] = []
        return job_dict
    return None

def get_queue_length():
    """Get the number of jobs currently in queue"""
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'")
    count = c.fetchone()[0]
    conn.close()
    return count

# Queue worker thread
def process_queue():
    """Process jobs from the queue one at a time"""
    global is_worker_running
    
    while True:
        try:
            # Get the next job from queue
            job_info = job_queue.get(block=True, timeout=5)  # Wait up to 5 seconds for new jobs
            
            # Process the job
            if job_info['type'] == 'single':
                process_single_text_job(
                    job_info['id'],
                    job_info['reference_audio'],
                    job_info['text'],
                    job_info['max_chunk_size'],
                    job_info['use_fast_mode']
                )
            elif job_info['type'] == 'batch':
                process_batch_job(
                    job_info['id'],
                    job_info['reference_audio'],
                    job_info['text_files'],
                    job_info['max_chunk_size'],
                    job_info['use_fast_mode']
                )
                
            # Update queue positions after job completes
            update_queue_positions()
            
            # Mark task as done
            job_queue.task_done()
            
        except queue.Empty:
            # If queue is empty for too long, exit the worker thread
            with queue_lock:
                if job_queue.empty():
                    is_worker_running = False
                    break
        except Exception as e:
            # Log any errors
            print(f"Error in queue worker: {str(e)}")
            try:
                job_queue.task_done()  # Make sure to mark the task as done even if it fails
            except:
                pass

def start_worker_if_needed():
    """Start the worker thread if it's not already running"""
    global is_worker_running
    
    with queue_lock:
        if not is_worker_running:
            worker_thread = threading.Thread(target=process_queue, daemon=True)
            worker_thread.start()
            is_worker_running = True

def add_to_queue(job_id, job_type, reference_audio, **kwargs):
    """Add a job to the queue and update its status"""
    # Create job info dictionary
    job_info = {
        'id': job_id,
        'type': job_type,
        'reference_audio': reference_audio,
        **kwargs
    }
    
    # Get queue position
    queue_position = get_queue_length() + 1
    
    # Update job status to queued
    active_jobs[job_id]['status'] = 'queued'
    active_jobs[job_id]['queue_position'] = queue_position
    active_jobs[job_id]['status_message'] = f"Waiting in queue (position {queue_position})"
    
    # Save to database
    save_job(job_id, job_type, 'queued', reference_audio, queue_position=queue_position)
    
    # Add to queue
    job_queue.put(job_info)
    
    # Start worker if needed
    start_worker_if_needed()

# Text processing functions
def is_past_participle(word):
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

def determine_d_expansion(text, match_start, match_end):
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
            if is_past_participle(word_after_never):
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
    if next_word and is_past_participle(next_word):
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

def expand_contractions(text):
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
        expansion_type = determine_d_expansion(text, start, end)
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

def preprocess_text(text):
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
    text = expand_contractions(text)
    
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

def split_text_into_chunks(text, max_chunk_size):
    """Split text into proper chunks for better TTS processing."""
    # Preprocess text to handle special cases
    text = preprocess_text(text)
    
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
    target_chunk_size = min(200, max_chunk_size)
    
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

# Audio generation
def process_chunk(tts_model, reference_audio, chunk, output_file, use_fast_mode):
    """Process a single text chunk into audio"""
    try:
        if use_fast_mode:
            tts_model.infer_fast(reference_audio, chunk, output_file)
        else:
            tts_model.infer(reference_audio, chunk, output_file)
        
        # Verify the file was generated and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            return {'success': True, 'file': output_file, 'text': chunk}
        else:
            return {'success': False, 'error': 'Generated audio file is invalid or empty'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Background job processor for single text generation
def process_single_text_job(job_id, reference_audio, text, max_chunk_size, use_fast_mode):
    try:
        # Update job status
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 0
        active_jobs[job_id]['queue_position'] = 0  # No longer in queue
        save_job(job_id, 'single', 'processing', reference_audio, queue_position=0)
        
        # Split text into chunks
        chunks = split_text_into_chunks(text, max_chunk_size)
        total_chunks = len(chunks)
        
        # Generate audio for each chunk
        temp_files = []
        success_count = 0
        
        for i, chunk in enumerate(chunks):
            # Create temp file for audio chunk
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            
            # Process the chunk
            result = process_chunk(tts, reference_audio, chunk, temp_file.name, use_fast_mode == 'fast')
            
            if result['success']:
                temp_files.append(temp_file.name)
                success_count += 1
                
            # Update progress
            progress = int(((i+1) / total_chunks) * 100)
            active_jobs[job_id]['progress'] = progress
            active_jobs[job_id]['status_message'] = f"Processed chunk {i+1}/{total_chunks}"
            
        # Concatenate chunks if we have at least one
        if temp_files:
            # Create output filename based on the reference audio filename
            reference_basename = os.path.splitext(os.path.basename(reference_audio))[0]
            # Remove UUID prefix if present from uploaded files
            if "_" in reference_basename and len(reference_basename.split("_")[0]) == 36:
                reference_basename = "_".join(reference_basename.split("_")[1:])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{reference_basename}_output_{timestamp}.wav"
            output_file = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Concatenate audio files
            if len(temp_files) == 1:
                # If only one chunk, just copy the file
                waveform, sample_rate = torchaudio.load(temp_files[0])
                torchaudio.save(output_file, waveform, sample_rate)
            else:
                # If multiple chunks, concatenate them
                waveforms = []
                sample_rate = None
                
                for file in temp_files:
                    waveform, sr = torchaudio.load(file)
                    waveforms.append(waveform)
                    sample_rate = sr
                
                # Concatenate all waveforms
                concatenated = torch.cat(waveforms, dim=1)
                
                # Save the concatenated audio
                torchaudio.save(output_file, concatenated, sample_rate)
            
            # Clean up temp files
            for file in temp_files:
                try:
                    os.remove(file)
                except:
                    pass
                    
            # Update job status as complete
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['progress'] = 100
            active_jobs[job_id]['status_message'] = "Audio generation complete"
            active_jobs[job_id]['output_files'] = [output_file]
            
            # Save to database
            save_job(job_id, 'single', 'completed', reference_audio, [os.path.basename(output_file)], queue_position=0)
            
            return True
        else:
            # Failed to generate any chunks
            active_jobs[job_id]['status'] = 'failed'
            active_jobs[job_id]['status_message'] = "Failed to generate any audio chunks"
            save_job(job_id, 'single', 'failed', reference_audio, error_message="Failed to generate any audio chunks", queue_position=0)
            return False
            
    except Exception as e:
        # Handle exceptions
        error_message = f"Error in job {job_id}: {str(e)}"
        print(error_message)
        
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['status_message'] = error_message
        
        save_job(job_id, 'single', 'failed', reference_audio, error_message=error_message, queue_position=0)
        return False

# Background job processor for batch text generation
def process_batch_job(job_id, reference_audio, text_files, max_chunk_size, use_fast_mode):
    try:
        # Update job status
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 0
        active_jobs[job_id]['queue_position'] = 0  # No longer in queue
        save_job(job_id, 'batch', 'processing', reference_audio, queue_position=0)
        
        output_files = []
        total_files = len(text_files)
        
        # Get reference audio basename for output naming
        ref_audio_basename = os.path.splitext(os.path.basename(reference_audio))[0]
        # Remove UUID prefix if present from uploaded files
        if "_" in ref_audio_basename and len(ref_audio_basename.split("_")[0]) == 36:
            ref_audio_basename = "_".join(ref_audio_basename.split("_")[1:])
        
        for i, text_file_path in enumerate(text_files):
            try:
                # Read text from file
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Skip empty files
                if not text_content.strip():
                    continue
                
                # Create output filename based on input filename
                base_filename = os.path.splitext(os.path.basename(text_file_path))[0]
                # Remove UUID prefix if present from uploaded files
                if "_" in base_filename and len(base_filename.split("_")[0]) == 36:
                    base_filename = "_".join(base_filename.split("_")[1:])
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{ref_audio_basename}_{base_filename}_{timestamp}.wav"
                output_file = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                
                # Update status
                active_jobs[job_id]['status_message'] = f"Processing file {i+1}/{total_files}: {base_filename}"
                
                # Split text into chunks
                chunks = split_text_into_chunks(text_content, max_chunk_size)
                total_chunks = len(chunks)
                
                # Generate audio for each chunk
                temp_files = []
                
                for j, chunk in enumerate(chunks):
                    # Create temp file for audio chunk
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_file.close()
                    
                    # Process the chunk
                    result = process_chunk(tts, reference_audio, chunk, temp_file.name, use_fast_mode == 'fast')
                    
                    if result['success']:
                        temp_files.append(temp_file.name)
                    
                    # Update file progress
                    file_progress = int(((j+1) / total_chunks) * 100)
                    active_jobs[job_id]['file_progress'] = file_progress
                
                # Concatenate chunks if we have at least one
                if temp_files:
                    if len(temp_files) == 1:
                        # If only one chunk, just copy the file
                        waveform, sample_rate = torchaudio.load(temp_files[0])
                        torchaudio.save(output_file, waveform, sample_rate)
                    else:
                        # If multiple chunks, concatenate them
                        waveforms = []
                        sample_rate = None
                        
                        for file in temp_files:
                            waveform, sr = torchaudio.load(file)
                            waveforms.append(waveform)
                            sample_rate = sr
                        
                        # Concatenate all waveforms
                        concatenated = torch.cat(waveforms, dim=1)
                        
                        # Save the concatenated audio
                        torchaudio.save(output_file, concatenated, sample_rate)
                    
                    # Add to output files
                    output_files.append(output_file)
                    
                    # Clean up temp files
                    for file in temp_files:
                        try:
                            os.remove(file)
                        except:
                            pass
                            
                    # Update progress
                    active_jobs[job_id]['progress'] = int(((i+1) / total_files) * 100)
            
            except Exception as e:
                print(f"Error processing file {text_file_path}: {str(e)}")
        
        # Job completed
        if output_files:
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['progress'] = 100
            active_jobs[job_id]['status_message'] = "Batch processing complete"
            active_jobs[job_id]['output_files'] = output_files
            
            # Save to database
            save_job(job_id, 'batch', 'completed', reference_audio, 
                    [os.path.basename(file) for file in output_files], queue_position=0)
        else:
            active_jobs[job_id]['status'] = 'failed'
            active_jobs[job_id]['status_message'] = "Failed to generate any audio files"
            save_job(job_id, 'batch', 'failed', reference_audio, 
                   error_message="Failed to generate any audio files", queue_position=0)
            
    except Exception as e:
        # Handle exceptions
        error_message = f"Error in batch job {job_id}: {str(e)}"
        print(error_message)
        
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['status_message'] = error_message
        
        save_job(job_id, 'batch', 'failed', reference_audio, error_message=error_message, queue_position=0)

# Routes
@app.route('/')
def index():
    queue_length = get_queue_length()
    return render_template('index.html', model_loaded=model_loaded, queue_length=queue_length)

@app.route('/history')
def history():
    jobs = get_jobs()
    return render_template('history.html', jobs=jobs)

@app.route('/job/<job_id>')
def job_details(job_id):
    job = get_job(job_id)
    if job:
        # Include active job data if available
        active_job_data = active_jobs.get(job_id, {})
        return render_template('job_details.html', job=job, active_job=active_job_data)
    else:
        return redirect(url_for('history'))

@app.route('/api/queue_status')
def queue_status():
    """Get information about the current queue status"""
    queue_length = get_queue_length()
    
    # Get first job in queue
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1")
    next_job = c.fetchone()
    conn.close()
    
    next_job_id = next_job['id'] if next_job else None
    
    return jsonify({
        'success': True,
        'queue_length': queue_length,
        'next_job_id': next_job_id
    })

@app.route('/api/upload_reference', methods=['POST'])
def upload_reference():
    if 'reference' not in request.files:
        return jsonify({'success': False, 'error': 'No reference file provided'})
        
    file = request.files['reference']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file:
        # Secure the filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': os.path.basename(file_path)
        })
    
    return jsonify({'success': False, 'error': 'Unknown error'})

@app.route('/api/upload_text_files', methods=['POST'])
def upload_text_files():
    if 'text_files' not in request.files:
        return jsonify({'success': False, 'error': 'No text files provided'})
    
    files = request.files.getlist('text_files')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
            
        # Only process .txt files
        if file and file.filename.endswith('.txt'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            file.save(file_path)
            
            uploaded_files.append({
                'file_path': file_path,
                'filename': os.path.basename(file_path)
            })
    
    if uploaded_files:
        return jsonify({
            'success': True,
            'files': uploaded_files
        })
    else:
        return jsonify({'success': False, 'error': 'No valid text files uploaded'})

@app.route('/api/generate_speech', methods=['POST'])
def generate_speech():
    # Check if model is loaded
    if not model_loaded:
        return jsonify({'success': False, 'error': 'TTS model not loaded'})
    
    # Validate inputs
    data = request.form
    
    if 'reference_audio' not in data:
        return jsonify({'success': False, 'error': 'No reference audio provided'})
        
    reference_audio = data.get('reference_audio')
    text = data.get('text', '')
    max_chunk_size = int(data.get('max_chunk_size', 250))
    use_fast_mode = data.get('generation_mode', 'fast')
    
    if not os.path.exists(reference_audio):
        return jsonify({'success': False, 'error': 'Reference audio file does not exist'})
        
    if not text.strip():
        return jsonify({'success': False, 'error': 'No text provided'})
    
    # Create job ID and initialize job tracking
    job_id = str(uuid.uuid4())
    
    # Get initial queue position
    queue_position = get_queue_length() + 1
    
    active_jobs[job_id] = {
        'id': job_id,
        'type': 'single',
        'status': 'queued',
        'progress': 0,
        'queue_position': queue_position,
        'status_message': f'Waiting in queue (position {queue_position})',
        'reference_audio': os.path.basename(reference_audio),
        'created_at': datetime.now().isoformat(),
        'output_files': []
    }
    
    # Save initial job state to database
    save_job(job_id, 'single', 'queued', reference_audio, queue_position=queue_position)
    
    # Add job to processing queue
    add_to_queue(job_id, 'single', reference_audio, text=text, max_chunk_size=max_chunk_size, use_fast_mode=use_fast_mode)
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'queue_position': queue_position,
        'message': f'Speech generation job queued (position {queue_position})'
    })

@app.route('/api/batch_process', methods=['POST'])
def batch_process():
    # Check if model is loaded
    if not model_loaded:
        return jsonify({'success': False, 'error': 'TTS model not loaded'})
    
    # Validate inputs
    data = request.form
    
    if 'reference_audio' not in data:
        return jsonify({'success': False, 'error': 'No reference audio provided'})
        
    reference_audio = data.get('reference_audio')
    text_files = json.loads(data.get('text_files', '[]'))
    max_chunk_size = int(data.get('max_chunk_size', 250))
    use_fast_mode = data.get('generation_mode', 'fast')
    
    if not os.path.exists(reference_audio):
        return jsonify({'success': False, 'error': 'Reference audio file does not exist'})
        
    if not text_files:
        return jsonify({'success': False, 'error': 'No text files provided'})
    
    # Validate all text files exist
    for file_path in text_files:
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': f'Text file does not exist: {file_path}'})
    
    # Create job ID and initialize job tracking
    job_id = str(uuid.uuid4())
    
    # Get initial queue position
    queue_position = get_queue_length() + 1
    
    active_jobs[job_id] = {
        'id': job_id,
        'type': 'batch',
        'status': 'queued',
        'progress': 0,
        'file_progress': 0,
        'queue_position': queue_position,
        'status_message': f'Waiting in queue (position {queue_position})',
        'reference_audio': os.path.basename(reference_audio),
        'created_at': datetime.now().isoformat(),
        'output_files': []
    }
    
    # Save initial job state to database
    save_job(job_id, 'batch', 'queued', reference_audio, queue_position=queue_position)
    
    # Add job to processing queue
    add_to_queue(job_id, 'batch', reference_audio, text_files=text_files, max_chunk_size=max_chunk_size, use_fast_mode=use_fast_mode)
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'queue_position': queue_position,
        'message': f'Batch processing job queued (position {queue_position})'
    })

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    # Return active job status if available
    if job_id in active_jobs:
        return jsonify({
            'success': True, 
            'job': active_jobs[job_id]
        })
    
    # Otherwise look up from database
    job = get_job(job_id)
    if job:
        return jsonify({
            'success': True,
            'job': job,
            'from_db': True
        })
    
    return jsonify({
        'success': False,
        'error': 'Job not found'
    })

@app.route('/api/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a queued job (only works for jobs that haven't started processing)"""
    # Check if job exists
    job = get_job(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'})
    
    # Can only cancel queued jobs
    if job['status'] != 'queued':
        return jsonify({'success': False, 'error': 'Only queued jobs can be canceled'})
    
    # Update job status in database
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    c.execute("UPDATE jobs SET status = 'canceled', completed_at = ? WHERE id = ?", 
              (datetime.now().isoformat(), job_id))
    conn.commit()
    conn.close()
    
    # Update active job status if it exists
    if job_id in active_jobs:
        active_jobs[job_id]['status'] = 'canceled'
        active_jobs[job_id]['status_message'] = 'Job canceled by user'
    
    # Update queue positions for remaining jobs
    update_queue_positions()
    
    return jsonify({
        'success': True,
        'message': 'Job canceled successfully'
    })

@app.route('/api/job/recent')
def get_recent_jobs():
    """Get the most recent jobs for the homepage"""
    jobs = get_jobs()
    # Return just the most recent 10 jobs
    recent_jobs = jobs[:10] if jobs else []
    
    return jsonify({
        'success': True,
        'jobs': recent_jobs
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

# Initialize database at startup
with app.app_context():
    init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 