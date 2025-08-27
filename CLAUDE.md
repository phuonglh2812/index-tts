# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

IndexTTS is a GPT-style text-to-speech (TTS) system that can generate speech from text with voice cloning capabilities. It's primarily focused on Chinese TTS with enhanced punctuation control and pinyin correction features.

## Core Architecture

The system consists of several key components:

- **indextts/**: Main package containing the TTS implementation
  - **infer.py**: Core inference engine with IndexTTS class
  - **cli.py**: Command-line interface entry point
  - **gpt/**: GPT model implementation with conformer encoder
  - **BigVGAN/**: Neural vocoder for high-quality audio generation
  - **vqvae/**: Vector quantized variational autoencoder for mel spectrogram encoding
  - **utils/**: Utility modules for text processing, feature extraction, and checkpoints

- **checkpoints/**: Model weights and configuration
  - **config.yaml**: Main configuration file with model parameters
  - Model files: `gpt.pth`, `dvae.pth`, `bigvgan_generator.pth`, etc.

## Development Commands

### Setup and Installation
```bash
# Create conda environment
conda create -n index-tts python=3.10
conda activate index-tts

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .

# For web UI support
pip install -e ".[webui]"
```

### Running the System
```bash
# Command-line inference
indextts "Your text here" --voice reference_voice.wav --output output.wav

# Web interface
python webui.py

# Regression testing
python tests/regression_test.py
```

### Model Downloads
Download required models using huggingface-cli:
```bash
huggingface-cli download IndexTeam/Index-TTS \
  bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints
```

## Key Configuration

The main configuration is in `checkpoints/config.yaml` with the following structure:
- **dataset**: Mel spectrogram and audio processing settings
- **gpt**: GPT model architecture (1280 dim, 24 layers, conformer conditioning)
- **vqvae**: Vector quantization parameters
- **bigvgan**: Neural vocoder configuration with snake activation

## Entry Points

1. **CLI**: `indextts.cli:main` - Command-line interface
2. **Python API**: `IndexTTS` class from `indextts.infer`
3. **Web UI**: `webui.py` - Gradio-based web interface
4. **Alternative GUI**: `gui2.py` - Alternative desktop interface

## Text Processing

The system includes sophisticated text normalization:
- Chinese pinyin correction support
- Punctuation-based pause control
- Bilingual (Chinese/English) text processing
- Sentence segmentation and tokenization

## Model Components

- **GPT Model**: Autoregressive text-to-mel generation with conformer encoder
- **DVAE**: Discrete variational autoencoder for mel spectrogram encoding
- **BigVGAN**: High-quality neural vocoder with snake activation and anti-aliasing

## Testing

Run the regression test suite:
```bash
python tests/regression_test.py
```

This tests various text types including Chinese, English, mixed bilingual content, and long-form text generation.

## Output Structure

Generated audio files are saved to:
- `outputs/` - Standard output directory
- `output/` - Alternative output location (legacy)

## Device Support

The system automatically detects and uses available hardware:
- CUDA GPUs (preferred, with optional FP16)
- Apple Silicon MPS
- CPU fallback