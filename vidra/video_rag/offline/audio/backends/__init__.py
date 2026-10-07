"""The model backends the registries in `models.py` resolve to, imported only
when asked for by name.

    whisper.py    a Transcriber (faster-whisper)
    pyannote.py   a Diarizer
    cuda.py       preloads the CUDA DLLs CTranslate2 loads by name
"""
