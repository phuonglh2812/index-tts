# CORRECTED: Hyphen Processing Logic

## Hiểu đúng yêu cầu:

### ✅ CONVERT (- thành space):
1. **Hyphens giữa 2 từ**: `mother-in-law` → `mother in law`
2. **Compound words**: `long-term` → `long term`  
3. **ALL hyphens connecting words** → Convert to spaces

### ✅ PRESERVE (giữ nguyên -):
1. **Standalone hyphens with spaces**: `hello - this is a test` → `hello - this is a test`
2. **Spaced hyphens**: ` - ` (có space 2 bên) → Giữ nguyên

## Corrected Logic:

```python
def preprocess_hyphens_smart(text):
    """
    Convert word-connecting hyphens to spaces
    Preserve standalone/spaced hyphens
    
    Examples:
    - "mother-in-law" → "mother in law" (CONVERT)
    - "long-term project" → "long term project" (CONVERT)  
    - "hello - this is test" → "hello - this is test" (PRESERVE)
    - "end sentence - next sentence" → "end sentence - next sentence" (PRESERVE)
    """
    
    import re
    
    # Convert hyphens that are directly between words (no spaces)
    # Pattern: word-word → word word
    text = re.sub(r'(?<=\\w)-(?=\\w)', ' ', text)
    
    return text

# Test cases:
test_cases = [
    {
        'input': 'mother-in-law is here',
        'expected': 'mother in law is here',
        'description': 'Compound word hyphens converted'
    },
    {
        'input': 'long-term project',  
        'expected': 'long term project',
        'description': 'Word-connecting hyphen converted'
    },
    {
        'input': 'hello - this is test',
        'expected': 'hello - this is test',
        'description': 'Spaced hyphen preserved'
    },
    {
        'input': 'end sentence - next sentence',
        'expected': 'end sentence - next sentence', 
        'description': 'Spaced hyphen preserved (ANY position)'
    },
    {
        'input': 'mix word-word and spaced - hyphen',
        'expected': 'mix word word and spaced - hyphen',
        'description': 'Mixed: convert connecting, preserve spaced'
    }
]
```

## Regex Explanation:

```python
r'(?<=\\w)-(?=\\w)'
```

- `(?<=\\w)` = **Positive lookbehind**: Phải có word character trước `-`
- `-` = **The hyphen itself**  
- `(?=\\w)` = **Positive lookahead**: Phải có word character sau `-`

**Kết quả**: Chỉ match hyphens **directly between words** (không có spaces)

## Updated Preprocessing Pipeline:

```python
def preprocess_text(text):
    """Enhanced preprocessing with corrected hyphen handling"""
    
    # 1. NEW: Convert word-connecting hyphens to spaces
    text = preprocess_hyphens_smart(text)
    
    # 2. Handle ellipses before expanding contractions
    text = re.sub(r'\\.{3,}', ',', text)
    
    # 3. Add comma after question marks for natural pauses  
    text = re.sub(r'\\?(?![\\s,])', '?,', text)

    # 4. Replace standalone hyphens with commas (EXISTING LOGIC)
    # Note: This now only affects spaced hyphens since word-hyphens are already converted
    text = re.sub(r'(?<!\\w)-(?!\\w)', ',', text)
    
    # 5. Remove double quotes but keep content
    text = text.replace('"', '').replace('"', '')
    
    # 6. Expand contractions (existing logic)
    text = expand_contractions(text)
    
    # 7. Remove unusual Unicode characters
    text = re.sub(r'[^\\x00-\\x7F]+', ' ', text)
    
    # 8. Normalize whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    return text.strip()
```

## Complete Flow Example:

```
Input: "My mother-in-law said hello - this is important - for long-term success."

Step 1 - preprocess_hyphens_smart():
"My mother in law said hello - this is important - for long term success."
 └─converted─┘                    └─preserved─┘        └─converted─┘

Step 2-3 - Other preprocessing (ellipses, questions):
"My mother in law said hello - this is important - for long term success."

Step 4 - Replace standalone hyphens with commas:
"My mother in law said hello , this is important , for long term success."
                              ↑                    ↑
                         (spaced hyphens became commas for pause)

Final result for TTS:
"My mother in law said hello , this is important , for long term success."
```

## Key Points:

✅ **Word-connecting hyphens**: `word-word` → `word word` (ALWAYS convert)  
✅ **Spaced hyphens**: ` - ` → `,` (becomes comma for natural pause)  
✅ **Any position**: Logic works at sentence beginning, middle, end  
✅ **Compound words**: All compound words broken into separate words  

Đây có đúng ý bạn không?