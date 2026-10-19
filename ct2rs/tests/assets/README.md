# Test Assets

This directory contains or references assets used during testing.

## Open Speech Repository (OSR) Sample

- **File**: `test.wav` (downloaded dynamically during Whisper tests)
- **Source URL**: [https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav](https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav)
- **Origin**: [Open Speech Repository (OSR)](https://www.voiptroubleshooter.com/open_speech/index.html) hosted by VoIP Troubleshooter (Telchemy).
- **Terms of Use**:
  The speech materials are freely useable for product development, testing, research, and any reasonable application. Under the repository's conditions of use:
  - Users are permitted to copy, download, broadcast, modify, and incorporate the materials into test equipment and software.
  - Users must identify the source of the speech materials by explicitly citing or labeling them as **"Open Speech Repository"**.

## faster-whisper Hotwords Sample

- **File**: `hotwords.wav` (committed)
- **Source URL**: [https://github.com/SYSTRAN/faster-whisper/blob/9e472ae9de8bb374552f531f752654554b76b2d6/tests/data/hotwords.mp3](https://github.com/SYSTRAN/faster-whisper/blob/9e472ae9de8bb374552f531f752654554b76b2d6/tests/data/hotwords.mp3)
- **Origin**: test data of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN, used by its `test_hotwords` test.
- **Conversion**: `ffmpeg -i hotwords.mp3 -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact hotwords.wav` (16 kHz mono 16-bit PCM).
- **License**: MIT License, Copyright (c) 2023 SYSTRAN.
