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

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
                            QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, 
                            QFileDialog, QProgressBar, QCheckBox, QSlider, 
                            QSpinBox, QGroupBox, QComboBox, QMessageBox, QFrame)
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
        
    def expand_contractions(self, text):
        """Expand common English contractions for better TTS pronunciation."""
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",  # don't, doesn't, wouldn't, etc.
            r"'re": " are",  # we're, they're, etc.
            r"'s": " is",    # he's, she's, it's (possessive case is handled differently)
            r"'d": " would", # I'd, you'd, etc.
            r"'ll": " will", # I'll, you'll, etc.
            r"'ve": " have", # I've, you've, etc.
            r"'m": " am",    # I'm
            r"o'clock": "of the clock",
            r"ma'am": "madam",
            r"ain't": "is not"
        }

        for contraction, expansion in contractions.items():
            # Look for contractions with word boundaries
            text = re.sub(r'\b' + contraction, expansion, text, flags=re.IGNORECASE)
        
        return text

    def preprocess_text(self, text):
        """Preprocess text to handle special cases like hyphenated words."""
        # Expand contractions first
        text = self.expand_contractions(text)
        
        # Convert hyphenated phrases like "hand-to-mouth" to "hand to mouth"
        text = re.sub(r'(\w+)-to-(\w+)', r'\1 to \2', text)
        
        # Handle other common hyphenated words that should be spoken as separate words
        text = re.sub(r'(\w+)-(\w+)', lambda m: f"{m.group(1)} {m.group(2)}" 
                     if len(m.group(1)) > 1 and len(m.group(2)) > 1 
                     else m.group(0), text)
        
        # Remove any unusual Unicode characters that might cause issues
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle quotes and brackets consistently
        text = text.replace('"', ' ').replace('"', ' ').replace(''', "'").replace(''', "'")
        
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
        sentence_end_patterns = r'(?<=[.!?。！？…])\s*|(?<=[\.\?!。！？…][\'"])\s+'
        
        # Split by sentence boundaries
        sentences = re.split(sentence_end_patterns, text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s and s.strip()]
        
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
        
        # Create input section
        input_group = QGroupBox("Input Settings")
        input_layout = QVBoxLayout(input_group)
        
        # Reference voice section
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
        
        # Text input
        text_label = QLabel("Text to Speak:")
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter the text you want to convert to speech...")
        self.text_input.setMinimumHeight(100)
        
        # Options section
        options_group = QGroupBox("Generation Options")
        options_layout = QHBoxLayout(options_group)
        
        # Left options column
        left_options = QVBoxLayout()
        
        # Chunk size option
        chunk_layout = QHBoxLayout()
        chunk_label = QLabel("Max Chunk Size:")
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(50, 500)
        self.chunk_size.setValue(250)
        self.chunk_size.setSuffix(" characters")
        chunk_layout.addWidget(chunk_label)
        chunk_layout.addWidget(self.chunk_size)
        left_options.addLayout(chunk_layout)
        
        # Mode option
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Generation Mode:")
        self.generation_mode = QComboBox()
        self.generation_mode.addItem("Fast Mode (Better Performance)")
        self.generation_mode.addItem("Standard Mode (Higher Quality)")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.generation_mode)
        left_options.addLayout(mode_layout)
        
        # Right options column
        right_options = QVBoxLayout()
        
        # Output folder option
        output_layout = QHBoxLayout()
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
        right_options.addLayout(output_layout)
        
        # Add options to layout
        options_layout.addLayout(left_options)
        options_layout.addLayout(right_options)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("Generate Speech")
        self.generate_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.generate_button.setMinimumHeight(50)
        self.generate_button.clicked.connect(self.generate_speech)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMinimumHeight(50)
        self.clear_button.clicked.connect(self.clear_fields)
        
        button_layout.addWidget(self.generate_button, 2)
        button_layout.addWidget(self.clear_button, 1)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
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
        
        # Add all components to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addWidget(separator)
        main_layout.addLayout(reference_layout)
        main_layout.addWidget(text_label)
        main_layout.addWidget(self.text_input)
        main_layout.addWidget(options_group)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(progress_group)
        main_layout.addWidget(output_group)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
        # Style the application
        self.apply_styles()
        
        # Disable generate button if model not loaded
        if not self.model_loaded:
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Model Not Loaded")
            self.status_label.setText("Error: IndexTTS model not loaded")
    
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
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
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
        self.generate_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.chunk_size.setEnabled(enabled)
        self.generation_mode.setEnabled(enabled)
        
        if not enabled:
            self.generate_button.setText("Generating...")
        else:
            self.generate_button.setText("Generate Speech")
        
    def update_progress(self, value, message):
        """Update progress bar and status label"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def generation_finished(self, output_file, message):
        """Handle successful generation completion"""
        self.output_file_path.setText(output_file)
        self.status_label.setText(message)
        self.play_button.setEnabled(True)
        self.set_ui_enabled(True)
        
        # Show notification
        QMessageBox.information(self, "Success", "Speech generation completed successfully!")
        
    def generation_error(self, error_message):
        """Handle generation error"""
        self.status_label.setText(error_message)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IndexTTSApp()
    window.show()
    sys.exit(app.exec())
