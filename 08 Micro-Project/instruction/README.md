# Micro-Project: HLS Web Player

Andrzej Matiolanski, Mikolaj Leszczuk

## Introduction

The purpose of this micro-project is to build and customize a browser-based HTTP Live Streaming (HLS) player. The player should use `hls.js` where Media Source Extensions are available and fall back to native HLS playback when the browser supports it directly.

This exercise follows the streaming workflows from Exercises 06 and 07. Here the focus moves from transport setup to adaptive HTTP delivery, playlist generation, browser playback and basic player diagnostics.

## Learning Goals

After completing this exercise, the student should be able to:

- generate HLS media playlists and a master playlist from a local source,
- serve HLS assets over HTTP for browser playback,
- explain how an HLS master playlist points to multiple bitrate variants,
- use `hls.js` to load an HLS stream in a web player,
- implement manual bitrate selection,
- display basic information about the active stream,
- inspect playlist and segment requests in the browser.

## Prerequisites

### Required software

Install:

- `ffmpeg` and `ffprobe`,
- Python 3, used only for a local static HTTP server,
- a current desktop browser such as Chrome, Firefox, Edge or Safari.

Typical installation commands:

```bash
# Ubuntu
sudo apt update
sudo apt install ffmpeg python3

# macOS with Homebrew
brew install ffmpeg python
```

Check the tools:

```bash
ffmpeg -hide_banner -version
ffprobe -hide_banner -version
python3 --version
```

### Lab materials

Run commands from the exercise root directory:

```bash
cd "08 Micro-Project"
```

Useful files:

- `scripts/create-sample-source.sh`
- `scripts/generate-hls.sh`
- `scripts/serve-player.sh`
- `player/index.html`
- `player/app.js`
- `player/styles.css`

Generated files are written under `player/streams/`.

## Expected Outputs

Each student should produce:

1. one local source video,
2. one generated HLS master playlist with at least three variants,
3. one working browser player loaded over HTTP,
4. manual quality switching in the player,
5. displayed stream information: bitrate, resolution and codecs when available,
6. a short request log or browser Network tab screenshot showing `.m3u8` and segment requests,
7. a short H.264 vs H.265 size comparison,
8. a short codec/report research note,
9. a final project URL or repository link submitted through the Moodle assignment form.

## Exercise

### Part 1. Create or select source media

You can use your own short video file, a screen recording or the generated sample source from this exercise. The generated source is deterministic and does not require downloading external media:

```bash
scripts/create-sample-source.sh media/source.mp4 60
```

Alternative Linux screen capture:

```bash
xdpyinfo | grep dimensions
ffmpeg -f x11grab -framerate 30 -video_size 1280x720 -i :0.0 \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p media/source.mp4
```

Alternative macOS screen capture:

```bash
ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg -f avfoundation -framerate 30 -i "<screen device index>:<audio device index>" \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p media/source.mp4
```

Inspect the result:

```bash
ffprobe -hide_banner media/source.mp4
```

Answer:

1. What are the container format, video codec and audio codec?
2. What are the resolution, duration and average bitrate?
3. If you used screen capture, what display name and capture size did you use?
4. Why is a local source preferable to a public web video for a reproducible lab?

### Part 2. Compare codecs

Find available encoders for H.264, H.265 and VP9:

```bash
ffmpeg -hide_banner -encoders | grep -E 'libx264|libx265|libvpx-vp9'
```

Create an H.265 version of the source:

```bash
ffmpeg -hide_banner -y -i media/source.mp4 \
  -c:v libx265 -preset medium -tag:v hvc1 \
  -c:a copy media/source_h265.mp4
```

Compare file sizes:

```bash
ls -lh media/source.mp4 media/source_h265.mp4
```

Research checkpoint: open the current Encoding.com resource library and identify the latest available Global Media Format Report. When this exercise was refreshed, the official resource page listed "Global Media Format Report 2023"; if a newer official edition appears, use the newest one and record its title and year.

From that report, record:

- the leading video codec,
- the leading audio codec,
- the top cloud storage providers listed in the report.

If the report is unavailable, record that instead of guessing.

If you use an external H.264/H.265 comparison source, record the claimed target bitrate reduction and cite the source. Treat it as context, then compare it with your own local transcode result.

Answer:

1. Which H.264, H.265 and VP9 encoder names are available on your machine?
2. How much smaller or larger is `media/source_h265.mp4` compared with `media/source.mp4`?
3. Which report title and year did you use?
4. Which video codec, audio codec and cloud storage providers were leading in that report?
5. What target bitrate reduction does your H.264/H.265 comparison source claim, if you used one?
6. Why can a single local transcode differ from industry-wide codec trends?

### Part 3. Generate HLS variants

Generate a small adaptive HLS package:

```bash
scripts/generate-hls.sh media/source.mp4 player/streams/sample
```

The script creates:

- `player/streams/sample/master.m3u8`
- `player/streams/sample/180p/playlist.m3u8`
- `player/streams/sample/360p/playlist.m3u8`
- `player/streams/sample/720p/playlist.m3u8`
- MPEG-TS media segments in each variant directory.

Open the master playlist:

```bash
cat player/streams/sample/master.m3u8
```

Answer:

1. Which tags point to available bitrate variants?
2. Which attributes describe bandwidth and resolution?
3. How are media playlists different from the master playlist?

### Part 4. Validate playback with VLC

Before testing the browser player, validate the generated HLS playlist with VLC:

```bash
# Ubuntu
vlc player/streams/sample/master.m3u8

# macOS
"/Applications/VLC.app/Contents/MacOS/VLC" player/streams/sample/master.m3u8
```

Answer:

1. Does playback start correctly?
2. Does VLC expose the available quality variants?
3. What failure do you see if segments are missing or paths are wrong?

### Part 5. Serve the player over HTTP

Do not open the player through `file://`. HLS playback in browsers should be tested through HTTP so playlist and segment requests behave like real web delivery.

Start the local server:

```bash
scripts/serve-player.sh 8008
```

Open:

```text
http://127.0.0.1:8008/
```

After generating local HLS assets, select "Local ABR sample". That predefined source points to:

```text
streams/sample/master.m3u8
```

Answer:

1. Does the player load the local HLS stream?
2. Which quality level is selected automatically after playback starts?
3. What happens when you switch quality manually?

### Part 6. Inspect browser requests

Open browser Developer Tools and switch to the Network tab. Start playback and filter requests by:

```text
m3u8
ts
```

The starter player also logs HLS requests observed through `hls.js` events.

Use `curl` to inspect the playlist through the same local HTTP server:

```bash
curl http://127.0.0.1:8008/streams/sample/master.m3u8
```

Answer:

1. Which request loads the master playlist?
2. Which requests load media playlists?
3. Which requests load media segments?
4. How does the request pattern change after a manual bitrate switch?

### Part 7. Customize the micro-project player

The starter player is intentionally small. Extend it into your submitted project.

Required features:

- successful HLS playback,
- a predefined list of at least 2-3 HLS playlists,
- a custom URL input,
- manual bitrate switching,
- automatic bitrate mode,
- display of bitrate, width, height and codec information when available,
- visible log of requested playlists and segments or equivalent evidence from browser Developer Tools.

Optional features:

- display of recent buffer and playback statistics,
- persistent custom playlist list,
- clearer error handling for missing playlists or CORS failures,
- access control mock-up for premium content,
- country/IP based access check using a documented API or a server-side implementation.

If you implement IP or geolocation based access control, document what is enforced on the server side and what is only a browser-side demonstration. Browser-only checks are easy to bypass and should not be described as secure content protection.

## Report

If a report is required, include:

1. source media description from `ffprobe`,
2. H.264 vs H.265 size comparison,
3. codec/report research note,
4. generated HLS master playlist,
5. player URL or repository URL,
6. screenshots or notes showing playback,
7. manual quality switching evidence,
8. request log evidence,
9. short conclusion on how HLS differs from the direct UDP, RTP, HTTP and RTSP workflows from Exercises 06 and 07.

## Troubleshooting

### The player works in VLC but not in the browser

Check that you opened the player through `http://127.0.0.1:8008/`, not through `file://`.

If the stream is remote, check CORS headers. Browser playback requires cross-origin access when playlists or segments are loaded from a different origin.

### The browser reports that HLS is unsupported

Use a current desktop browser. `hls.js` requires Media Source Extensions in most browsers. Safari can use native HLS playback for many streams.

### The generated stream has no audio

Use `ffprobe` to check whether the source file has an audio stream. The sample source generated by `scripts/create-sample-source.sh` includes AAC audio.

### Copied commands do not work

Use plain ASCII hyphens in command-line options, for example `-hls_time`, not typographic dashes copied from rich-text documents.

## References

1. hls.js project and usage guide, <https://github.com/video-dev/hls.js/>
2. hls.js latest demo, <https://hlsjs.video-dev.org/demo/>
3. RFC 8216: HTTP Live Streaming, <https://www.rfc-editor.org/rfc/rfc8216>
4. FFmpeg Documentation, <https://ffmpeg.org/documentation.html>
5. FFmpeg HLS muxer documentation, <https://ffmpeg.org/ffmpeg-formats.html#hls-2>
6. VLC User Documentation, <https://docs.videolan.me/vlc-user/>
7. Encoding.com Resource Library, <https://www.encoding.com/resources/>
