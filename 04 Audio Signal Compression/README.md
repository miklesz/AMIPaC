# Audio Signal Compression

Use `audio_comp.ipynb` as the current student exercise.

## Required Tools

Install the Python dependencies from the notebook, then restart Jupyter if widgets were upgraded:

```bash
pip install --upgrade ipympl librosa ffmpeg-python
```

The system `ffmpeg` and `ffprobe` executables must also be installed and visible in `PATH`. Check them before using the widget GUI:

```bash
ffmpeg -hide_banner -version
ffprobe -hide_banner -version
```

## GUI Fallback

If the widget GUI does not render or `ffprobe` is not found from Jupyter, complete the core comparison with command-line FFmpeg and document the commands in the report.

Example:

```bash
ffmpeg -y -i "Audio Sequences/vqegMM2_C01_Aorig.wav" -c:a libmp3lame -b:a 128k /tmp/amipac04-128k.mp3
ffmpeg -y -i "Audio Sequences/vqegMM2_C01_Aorig.wav" -c:a aac -b:a 128k /tmp/amipac04-128k.m4a
ffmpeg -y -i "Audio Sequences/vqegMM2_C01_Aorig.wav" -strict -2 -c:a vorbis -b:a 128k /tmp/amipac04-128k.ogg
ffprobe -hide_banner /tmp/amipac04-128k.mp3
ls -lh "Audio Sequences/vqegMM2_C01_Aorig.wav" /tmp/amipac04-128k.*
```

If your FFmpeg build uses a different Vorbis encoder name, check available encoders with:

```bash
ffmpeg -hide_banner -encoders | grep -i vorbis
```

Report the codec, bitrate, output file size and subjective quality observations.
