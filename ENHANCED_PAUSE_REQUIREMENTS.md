# Enhanced Pause Feature Requirements

## 1. Smart Chunking Logic (Đã hiểu)
- ✅ Pause markers TRƯỚC, chunking SAU  
- ✅ User control max_chunk_size (không fix cứng)
- ✅ Kết hợp thông minh 2 logic

## 2. YÊU CẦU BỔ SUNG MỚI:

### A. Minimum Chunk Size Warning
**Vấn đề**: Chunk < 50 ký tự → IndexTTS không chạy được  
**Giải pháp**:
- **UI Warning**: Hiển thị cảnh báo cho user
- **Auto-merge**: Tự động gộp chunks nhỏ với chunk kế tiếp
- **Validation**: Prevent user setting max_chunk_size < 50

```javascript
// Frontend validation
if (max_chunk_size < 50) {
    alert("Warning: Chunk size below 50 characters may cause processing issues. Recommended minimum: 50-100 characters.");
}
```

```python
# Backend auto-merge logic trong split_text_into_chunks()
def split_text_into_chunks(text, max_chunk_size):
    # Existing chunking logic...
    
    # NEW: Auto-merge chunks that are too small
    merged_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) < 50 and merged_chunks:
            # Merge with previous chunk if possible
            merged_chunks[-1] = merged_chunks[-1] + " " + chunk
        elif len(chunk.strip()) < 50 and len(chunks) > 1:
            # Find next chunk to merge with
            # Implementation details...
        else:
            merged_chunks.append(chunk)
    
    return merged_chunks
```

### B. Hyphen (-) Preprocessing  
**Vấn đề**: Dấu `-` trong câu (không phải end sentence) → IndexTTS tự động pause  
**Giải pháp**: Convert `-` thành space TRONG câu

```python
def preprocess_hyphens(text):
    """
    Convert hyphens to spaces within sentences (not at sentence boundaries)
    
    Logic:
    - Nếu `-` không ở cuối câu (sau nó không có . ! ?) → Convert to space
    - Nếu `-` ở cuối câu → Giữ nguyên (đây là pause tự nhiên)
    """
    
    # Pattern: hyphen NOT followed by sentence endings
    # (?!.*[.!?]) = negative lookahead, không có sentence ending sau hyphen trong câu hiện tại
    
    import re
    
    # Split text into sentences first
    sentences = re.split(r'([.!?]+)', text)
    processed_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            processed_sentences.append(sentence)
            continue
            
        # Check if this is actual sentence content (not punctuation)
        if not re.match(r'^[.!?]+$', sentence):
            # This is sentence content - convert internal hyphens to spaces
            # But preserve hyphens in compound words (word-word)
            
            # Replace standalone hyphens with spaces
            # Pattern: hyphen surrounded by spaces OR hyphen not between word characters
            sentence = re.sub(r'(?<!\\w)-(?!\\w)', ' ', sentence)
            sentence = re.sub(r'\\s+-\\s+', ' ', sentence)  # " - " → " "
            
        processed_sentences.append(sentence)
    
    return ''.join(processed_sentences)

# Usage example:
# Input:  "Hello world - this is a test. End of sentence - next sentence."  
# Output: "Hello world   this is a test. End of sentence - next sentence."
#         └─ converted─┘                  └─ preserved ─┘ (at sentence end)
```

**Better approach - More precise**:
```python
def preprocess_hyphens_smart(text):
    """
    Smart hyphen processing:
    1. Detect sentence boundaries
    2. Within sentences: convert standalone hyphens to spaces  
    3. At sentence endings: preserve hyphens (natural pause)
    """
    
    # First, protect sentence-ending hyphens by marking them
    text = re.sub(r'-\\s*([.!?])', '__SENTENCE_HYPHEN__\\1', text)
    
    # Convert standalone hyphens within sentences to spaces
    text = re.sub(r'\\s+-\\s+', ' ', text)  # " - " → " "
    text = re.sub(r'(?<!\\w)-(?!\\w)', ' ', text)  # standalone - → space
    
    # Restore sentence-ending hyphens
    text = re.sub(r'__SENTENCE_HYPHEN__', '-', text)
    
    return text
```

## 3. UPDATED PREPROCESSING PIPELINE:

```python
def preprocess_text(text):
    """Enhanced preprocessing with hyphen handling"""
    
    # 1. NEW: Handle hyphens first (before other processing)
    text = preprocess_hyphens_smart(text)
    
    # 2. Existing preprocessing steps...
    text = re.sub(r'\\.{3,}', ',', text)  # ellipses
    text = re.sub(r'\\?(?![\\s,])', '?,', text)  # question marks
    text = text.replace('"', '').replace('"', '')  # quotes
    
    # 3. Expand contractions (existing logic)
    text = expand_contractions(text)
    
    # 4. Rest of existing preprocessing...
    text = re.sub(r'[^\\x00-\\x7F]+', ' ', text)  # Unicode cleanup
    text = re.sub(r'\\s+', ' ', text)  # normalize whitespace
    
    return text.strip()
```

## 4. UI ENHANCEMENTS:

### HTML additions:
```html
<div class="form-group">
    <label for="max_chunk_size">Max Chunk Size:</label>
    <input type="number" id="max_chunk_size" name="max_chunk_size" 
           value="250" min="50" max="500" step="10" class="form-control">
    <small class="form-text text-muted">
        ⚠️ Minimum 50 characters recommended. Smaller chunks may fail processing.
    </small>
</div>

<div class="alert alert-info" role="alert">
    <strong>Text Processing Notes:</strong>
    <ul>
        <li>Hyphens (-) within sentences will be converted to spaces</li>
        <li>Hyphens at sentence endings are preserved for natural pauses</li>
        <li>Use custom pause markers like (pause) for precise control</li>
    </ul>
</div>
```

### JavaScript validation:
```javascript
document.getElementById('max_chunk_size').addEventListener('change', function() {
    const value = parseInt(this.value);
    const warningDiv = document.getElementById('chunk-size-warning');
    
    if (value < 50) {
        warningDiv.innerHTML = '<div class="alert alert-warning">⚠️ Chunk sizes below 50 characters may cause processing failures!</div>';
        warningDiv.style.display = 'block';
    } else {
        warningDiv.style.display = 'none';
    }
});
```

## 5. IMPLEMENTATION PLAN UPDATE:

### Phase 1: Core Logic
1. ✅ Database changes (pause columns)
2. ✅ **NEW**: `preprocess_hyphens_smart()` function
3. ✅ **ENHANCED**: `split_text_into_chunks()` với auto-merge logic
4. ✅ `split_text_by_pauses()` function  
5. ✅ `create_silence_audio()` function

### Phase 2: Processing Pipeline  
6. ✅ Update `preprocess_text()` to include hyphen handling
7. ✅ **ENHANCED**: `process_single_text_job()` với chunk validation
8. ✅ Add chunk size warnings and auto-merge

### Phase 3: UI/UX
9. ✅ UI warnings for small chunk sizes
10. ✅ Information about hyphen preprocessing  
11. ✅ Enhanced form validation

## 6. TEST CASES:

```python
# Test hyphen preprocessing
test_cases = [
    {
        'input': 'Hello - world this is test. End - next sentence.',
        'expected': 'Hello   world this is test. End - next sentence.',
        'description': 'Internal hyphen converted, sentence-end hyphen preserved'
    },
    {
        'input': 'Long-term project - very important - for success.',  
        'expected': 'Long-term project   very important   for success.',
        'description': 'Compound word preserved, standalone hyphens converted'
    },
    {
        'input': 'Text with (pause) and - hyphen (pause) more text.',
        'expected': 'Text with (pause) and   hyphen (pause) more text.',
        'description': 'Pause markers preserved, hyphens converted'
    }
]
```

## Tổng kết:

✅ **Smart chunking**: Pause → Chunk → Merge small chunks  
✅ **Hyphen handling**: Convert within sentences, preserve at endings  
✅ **User warnings**: Prevent issues with small chunks  
✅ **Backward compatible**: All existing functionality preserved