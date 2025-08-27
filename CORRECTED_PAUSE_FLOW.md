# Corrected Flow: Smart Chunking + Pause Integration

## Flow thực tế theo yêu cầu:

```
Text input: "Hello world this is a long sentence (pause) How are you doing today my friend (pause) Goodbye everyone"
User settings: max_chunk_size = 200, pause_marker = "(pause)", pause_duration = 1.0s
↓

1. SPLIT BY PAUSE MARKERS FIRST
   split_text_by_pauses()
   ↓
   Pause Segments: 
   - {"text": "Hello world this is a long sentence", "add_pause": true}
   - {"text": "How are you doing today my friend", "add_pause": true}  
   - {"text": "Goodbye everyone", "add_pause": false}

2. SMART CHUNKING FOR EACH SEGMENT
   For each segment, apply split_text_into_chunks() with user's max_chunk_size
   ↓
   Segment 1: "Hello world this is a long sentence" (39 chars < 200) → 1 chunk
   Segment 2: "How are you doing today my friend" (33 chars < 200) → 1 chunk
   Segment 3: "Goodbye everyone" (16 chars < 200) → 1 chunk

3. TTS PROCESSING
   Process each chunk → Audio files
   ↓
   Audio1 (from segment 1, chunk 1)
   Audio2 (from segment 2, chunk 1)  
   Audio3 (from segment 3, chunk 1)

4. CONCATENATION WITH PAUSES
   Audio1 + Silence(1s) + Audio2 + Silence(1s) + Audio3
```

## Example với text dài hơn:

```
Text: "This is a very long sentence that definitely exceeds the chunk size limit set by user (pause) Another very long sentence that also might need to be split into multiple chunks based on user settings"
User settings: max_chunk_size = 50

1. PAUSE SPLIT:
   Segment 1: "This is a very long sentence that definitely exceeds the chunk size limit set by user" (add_pause: true)
   Segment 2: "Another very long sentence that also might need to be split into multiple chunks based on user settings" (add_pause: false)

2. CHUNKING EACH SEGMENT:
   Segment 1 (82 chars > 50): 
   → Chunk 1.1: "This is a very long sentence that definitely"
   → Chunk 1.2: "exceeds the chunk size limit set by user"
   
   Segment 2 (101 chars > 50):
   → Chunk 2.1: "Another very long sentence that also might"  
   → Chunk 2.2: "need to be split into multiple chunks"
   → Chunk 2.3: "based on user settings"

3. TTS PROCESSING:
   Audio1.1, Audio1.2 (từ segment 1)
   Audio2.1, Audio2.2, Audio2.3 (từ segment 2)

4. CONCATENATION:
   Audio1.1 + Audio1.2 + Silence(1s) + Audio2.1 + Audio2.2 + Audio2.3
   └─ Segment 1 ─┘                    └─── Segment 2 ────┘
```

## Core Logic Changes:

### Updated `process_single_text_job()` function:

```python
def process_single_text_job(job_id, reference_audio, text, max_chunk_size, 
                          use_fast_mode, pause_marker='(pause)', pause_duration=1.0):
    try:
        # 1. SPLIT BY PAUSE MARKERS FIRST
        pause_segments = split_text_by_pauses(text, pause_marker)
        
        final_waveforms = []
        sample_rate = None
        total_segments = len(pause_segments)
        
        for segment_idx, segment in enumerate(pause_segments):
            segment_text = segment['text']
            add_pause = segment['add_pause']
            
            # 2. CHUNKING WITHIN EACH SEGMENT (with user's max_chunk_size)
            chunks = split_text_into_chunks(segment_text, max_chunk_size)
            
            # 3. PROCESS ALL CHUNKS IN THIS SEGMENT
            segment_waveforms = []
            for chunk_idx, chunk in enumerate(chunks):
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.close()
                
                result = process_chunk(tts, reference_audio, chunk, temp_file.name, use_fast_mode == 'fast')
                
                if result['success']:
                    waveform, sr = torchaudio.load(temp_file.name)
                    segment_waveforms.append(waveform)
                    sample_rate = sr
                    
                    # Clean up temp file
                    os.remove(temp_file.name)
                
                # Update progress: (segment_progress + chunk_progress) / total_segments
                chunk_progress = (chunk_idx + 1) / len(chunks) 
                overall_progress = int(((segment_idx + chunk_progress) / total_segments) * 100)
                active_jobs[job_id]['progress'] = overall_progress
                active_jobs[job_id]['status_message'] = f"Segment {segment_idx+1}/{total_segments}, Chunk {chunk_idx+1}/{len(chunks)}"
            
            # 4. CONCATENATE ALL CHUNKS IN THIS SEGMENT
            if segment_waveforms:
                segment_audio = torch.cat(segment_waveforms, dim=1)
                final_waveforms.append(segment_audio)
                
                # 5. ADD PAUSE AFTER SEGMENT (if needed)
                if add_pause:
                    silence = create_silence_audio(pause_duration, sample_rate)
                    final_waveforms.append(silence)
        
        # 6. FINAL CONCATENATION (all segments + pauses)
        if final_waveforms:
            final_audio = torch.cat(final_waveforms, dim=1)
            
            # Save output file (existing logic)...
            output_file = create_output_filename(reference_audio)
            torchaudio.save(output_file, final_audio, sample_rate)
            
            # Success handling...
            
    except Exception as e:
        # Error handling...
```

## Key Points:

### ✅ Chunking Logic:
- **Tự động**: Minimum 50-100 chars, maximum theo user setting
- **Thông minh**: Vẫn giữ sentence boundary detection
- **Linh hoạt**: User control hoàn toàn qua UI

### ✅ Pause Integration: 
- **Ưu tiên**: Pause markers được xử lý TRƯỚC chunking
- **Kết hợp**: Chunks trong cùng segment được nối liền
- **Chính xác**: Pause chỉ xuất hiện giữa các segments

### ✅ Progress Tracking:
- **Chi tiết**: Hiển thị segment và chunk hiện tại
- **Chính xác**: Progress tính theo tổng công việc thực tế

### ✅ UI Control:
- **max_chunk_size**: User input (50-500 chars)
- **pause_marker**: User customizable 
- **pause_duration**: User configurable (0.1-10s)

## Implementation Priority:

1. **Database** - Add pause columns
2. **Core Functions** - `split_text_by_pauses()`, `create_silence_audio()` 
3. **Modified Logic** - Update `process_single_text_job()` with this flow
4. **API** - Accept pause parameters from UI
5. **Frontend** - Add pause settings fields

Đây có đúng ý bạn không?