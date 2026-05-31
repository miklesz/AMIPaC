# Video Signal Compression

Use `video_compression.ipynb` as the current student exercise.

## Current Workflow

The current exercise path is Python/Jupyter based and uses OpenCV, FFmpeg through `ffmpeg-python`, and the local `compvid` helpers. Simulink files in `old/` are legacy material and are not required for the main exercise.

Install the Python dependencies from the notebook, then restart Jupyter if widgets were upgraded:

```bash
pip install --upgrade ipywidgets ffmpeg-python opencv-python matplotlib
```

The system `ffmpeg` executable must also be installed and visible in `PATH`.

## Report

The report should cover the assignment tasks listed at the end of `video_compression.ipynb`: source properties, extracted segment, grayscale/resized/compressed variants, file-size comparison and a short discussion of compression impact.

## Legacy Simulink Files

If a teacher explicitly asks to use the legacy Simulink model, open one of the versioned files in `old/` manually. Do not rely on hard-coded absolute paths from another computer.
