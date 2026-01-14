# Uplift TTS Integration for Pipecat

A Pipecat integration for [Uplift AI](https://upliftai.org)'s Text-to-Speech HTTP API, providing natural-sounding voice synthesis for conversational AI applications using the Urdu language.

## Introduction

Uplift AI provides high-quality, low-latency text-to-speech synthesis optimized for conversational AI applications. This integration enables seamless use of Uplift's TTS capabilities within Pipecat pipelines.

Integration Type: This is a TTSService integration (HTTP-based service without word/timestamp alignment), using the Uplift API endpoint: https://api.upliftai.org/v1/synthesis/text-to-speech

This is the first integration of Uplift AI with Pipecat, starting with their TTS service. Future integrations will include additional Uplift AI services.

**Features:**
- 4 distinct voices: Info/Edu, Gen Z, Dada Jee, Nostalgic News
- Multiple audio formats: WAV, MP3, OGG, ULAW
- Dynamic voice and format switching
- Automatic WAV header handling
- Full Pipecat metrics support

## Installation

```bash
pip install pipecat-ai aiohttp
```

Get your Uplift API key at [upliftai.org](https://upliftai.org).

## Usage

### Basic Usage

```python
from pipecat.services.uplift.tts import UpliftHttpTTSService
from pipecat.pipeline.pipeline import Pipeline

tts = UpliftHttpTTSService(
    api_key="sk_api_your_key_here",
    voice_id="v_8eelc901",  # Info/Edu voice
)

pipeline = Pipeline([
    # ... your STT and LLM services
    tts,
    transport.output(),
])
```

### Available Voices

- `v_8eelc901` - Info/Edu (educational, informative)
- `v_kwmp7zxt` - Gen Z (casual, contemporary)
- `v_yypgzenx` - Dada Jee
- `v_30s70t3a` - Nostalgic News (news-style)

### Audio Formats

WAV: `WAV_22050_16`, `WAV_22050_32`  
MP3: `MP3_22050_32`, `MP3_22050_64`, `MP3_22050_128`  
OGG: `OGG_22050_16`  
ULAW: `ULAW_8000_8`

## Running the Example

### Setup

1. Create a `.env` file:
```bash
UPLIFT_API_KEY=sk_api_your_key_here
OPENAI_API_KEY=your_openai_api_key
```

2. Run the example:
```bash
python example.py
```

3. Open `http://localhost:7860/client` in your browser and start speaking.

The bot will listen, process your speech with OpenAI, and respond using Uplift TTS.

## Compatibility

**Tested with:** Pipecat v0.0.99  
**Minimum:** Pipecat v0.0.85+, Python 3.10+

## Attribution

This integration was developed by [havkerboi123 as a personal contribution].

## License

BSD-2-Clause License - matching Pipecat's license terms.

```
Copyright (c) 2024-2026, [Your Name/Organization]
SPDX-License-Identifier: BSD 2-Clause License
```

## Support

- Integration issues: Open an issue on this repository
- Uplift API: [upliftai.org/support](https://upliftai.org/support)
- Pipecat: [docs.pipecat.ai](https://docs.pipecat.ai)