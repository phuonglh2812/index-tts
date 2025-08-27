# Plan Implementation: Custom Pause Feature for app.py

## Tổng quan tính năng
Cho phép user:
1. Định nghĩa ký hiệu pause tùy chỉnh (mặc định: `(pause)`)
2. Đặt thời lượng pause (0.1s - 10s)
3. Text sẽ được cắt tại vị trí pause và chèn silence vào audio cuối

## 1. Database Schema Changes

### Thêm columns vào table `jobs`:
```sql
ALTER TABLE jobs ADD COLUMN pause_marker TEXT DEFAULT '(pause)';
ALTER TABLE jobs ADD COLUMN pause_duration REAL DEFAULT 1.0;
```

**File cần sửa**: `app.py` - function `init_db()`
- Thêm logic check và add columns nếu chưa tồn tại
- Tương tự như đã làm với `queue_position` column

## 2. Frontend UI Changes

### A. Template HTML cần thêm:
**File**: `templates/index.html`

```html
<!-- Trong phần Single Text Generation -->
<div class="form-group">
    <label>Pause Settings</label>
    <div class="row">
        <div class="col-md-6">
            <label for="pause_marker">Pause Marker:</label>
            <input type="text" id="pause_marker" name="pause_marker" 
                   value="(pause)" placeholder="(pause)" class="form-control">
        </div>
        <div class="col-md-6">
            <label for="pause_duration">Pause Duration (seconds):</label>
            <input type="number" id="pause_duration" name="pause_duration" 
                   value="1.0" min="0.1" max="10.0" step="0.1" class="form-control">
        </div>
    </div>
</div>
```

### B. JavaScript cần update:
- Update AJAX calls trong `generate_speech()` để gửi thêm pause parameters
- Tương tự cho batch processing

## 3. Backend API Changes

### A. Route `/api/generate_speech` - Nhận thêm parameters:
```python
# Trong function generate_speech():
pause_marker = data.get('pause_marker', '(pause)')
pause_duration = float(data.get('pause_duration', 1.0))

# Validate pause_duration
if not (0.1 <= pause_duration <= 10.0):
    return jsonify({'success': False, 'error': 'Pause duration must be between 0.1 and 10.0 seconds'})

# Pass to queue
add_to_queue(job_id, 'single', reference_audio, 
            text=text, max_chunk_size=max_chunk_size, 
            use_fast_mode=use_fast_mode,
            pause_marker=pause_marker, pause_duration=pause_duration)
```

### B. Route `/api/batch_process` - Tương tự thêm pause parameters

### C. Function `save_job()` - Thêm pause parameters:
```python
def save_job(job_id, job_type, status, reference_audio, output_files=None, 
            error_message=None, queue_position=None, 
            pause_marker=None, pause_duration=None):
    # Update INSERT/UPDATE queries để include pause parameters
```

## 4. Core Text Processing Changes

### A. Tạo function mới `split_text_by_pauses()`:
**Vị trí**: Thêm vào app.py, sau function `split_text_into_chunks()`

```python
def split_text_by_pauses(text, pause_marker='(pause)'):
    """
    Split text by pause markers and return segments with pause indicators
    Returns: List of {'text': str, 'add_pause': bool}
    """
    if pause_marker not in text:
        return [{'text': text, 'add_pause': False}]
    
    # Split by pause marker
    segments = []
    parts = text.split(pause_marker)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if part:  # Skip empty parts
            segments.append({
                'text': part,
                'add_pause': i < len(parts) - 1  # Add pause except for last segment
            })
    
    return segments
```

### B. Function tạo silence audio:
```python
def create_silence_audio(duration_seconds, sample_rate=24000):
    """Create silent audio tensor"""
    num_samples = int(duration_seconds * sample_rate)
    silence = torch.zeros(1, num_samples, dtype=torch.float32)
    return silence
```

## 5. Core Processing Logic Changes

### A. Modify `process_single_text_job()`:
```python
def process_single_text_job(job_id, reference_audio, text, max_chunk_size, 
                          use_fast_mode, pause_marker='(pause)', pause_duration=1.0):
    try:
        # Existing validation code...
        
        # NEW: Split by pause markers first
        pause_segments = split_text_by_pauses(text, pause_marker)
        
        # Process each segment
        final_waveforms = []
        sample_rate = None
        
        for segment_idx, segment in enumerate(pause_segments):
            segment_text = segment['text']
            add_pause = segment['add_pause']
            
            # Process segment text into chunks (existing logic)
            chunks = split_text_into_chunks(segment_text, max_chunk_size)
            segment_temp_files = []
            
            # Generate audio for each chunk in this segment
            for chunk in chunks:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.close()
                
                result = process_chunk(tts, reference_audio, chunk, temp_file.name, use_fast_mode == 'fast')
                if result['success']:
                    segment_temp_files.append(temp_file.name)
            
            # Concatenate chunks in this segment
            if segment_temp_files:
                segment_waveforms = []
                for file in segment_temp_files:
                    waveform, sr = torchaudio.load(file)
                    segment_waveforms.append(waveform)
                    sample_rate = sr
                    
                # Concatenate segment chunks
                segment_audio = torch.cat(segment_waveforms, dim=1)
                final_waveforms.append(segment_audio)
                
                # Add pause if needed
                if add_pause:
                    silence = create_silence_audio(pause_duration, sample_rate)
                    final_waveforms.append(silence)
                
                # Cleanup segment temp files
                for file in segment_temp_files:
                    try:
                        os.remove(file)
                    except:
                        pass
            
            # Update progress
            progress = int(((segment_idx + 1) / len(pause_segments)) * 100)
            active_jobs[job_id]['progress'] = progress
            active_jobs[job_id]['status_message'] = f"Processed segment {segment_idx+1}/{len(pause_segments)}"
        
        # Final concatenation
        if final_waveforms:
            # Create output filename (existing logic)
            reference_basename = os.path.splitext(os.path.basename(reference_audio))[0]
            if "_" in reference_basename and len(reference_basename.split("_")[0]) == 36:
                reference_basename = "_".join(reference_basename.split("_")[1:])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{reference_basename}_output_{timestamp}.wav"
            output_file = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Concatenate all final waveforms (segments + pauses)
            final_audio = torch.cat(final_waveforms, dim=1)
            torchaudio.save(output_file, final_audio, sample_rate)
            
            # Success handling (existing code)...
            
    except Exception as e:
        # Error handling (existing code)...
```

### B. Modify `process_batch_job()` - Tương tự thêm pause parameters

### C. Update `add_to_queue()` function:
```python
def add_to_queue(job_id, job_type, reference_audio, **kwargs):
    # Existing code...
    
    # Save pause parameters to database
    pause_marker = kwargs.get('pause_marker', '(pause)')
    pause_duration = kwargs.get('pause_duration', 1.0)
    
    save_job(job_id, job_type, 'queued', reference_audio, 
             queue_position=queue_position,
             pause_marker=pause_marker, pause_duration=pause_duration)
```

## 6. Testing Strategy

### A. Unit Tests cần thêm:
1. Test `split_text_by_pauses()` với các cases:
   - Text không có pause marker
   - Text có 1 pause marker
   - Text có nhiều pause markers
   - Text bắt đầu/kết thúc bằng pause marker
   - Pause marker rỗng hoặc invalid

2. Test `create_silence_audio()`:
   - Các duration khác nhau
   - Sample rates khác nhau

### B. Integration Tests:
1. End-to-end test với pause markers
2. Test batch processing với pause
3. Test edge cases (pause markers liên tiếp, etc.)

## 7. Backward Compatibility

### A. Database Migration:
- Columns mới có DEFAULT values
- Existing jobs sẽ hoạt động bình thường

### B. API Compatibility:
- Pause parameters là optional
- Default behavior giống như trước

### C. UI Compatibility:
- Form fields mới có default values
- Không bắt buộc user phải nhập

## 8. Implementation Order

1. **Database changes** - Add columns, update init_db()
2. **Backend functions** - Add pause processing functions
3. **Core logic** - Modify job processing functions
4. **API updates** - Update routes to handle new parameters
5. **Frontend UI** - Add form fields
6. **JavaScript** - Update AJAX calls
7. **Testing** - Unit + integration tests

## 9. Files cần modify:

### Chính:
- `app.py` - Tất cả backend logic
- `templates/index.html` - UI fields
- Database schema update

### Có thể cần:
- CSS styles cho new fields
- JavaScript validation
- Error handling updates

## 10. Risk Assessment

### Low Risk:
- Database changes (có DEFAULT values)
- New functions (không ảnh hưởng existing code)

### Medium Risk:
- Modify core processing logic
- Audio concatenation changes

### Mitigation:
- Thorough testing
- Feature flag để enable/disable
- Rollback plan

## Kết luận
Tính năng này có thể implement một cách an toàn bằng cách:
1. Thêm logic mới mà không thay đổi existing flows
2. Sử dụng parameters optional với default values
3. Test kỹ lưỡng trước khi deploy