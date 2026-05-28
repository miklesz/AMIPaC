const video = document.getElementById("video");
const sourceForm = document.getElementById("source-form");
const sourceSelect = document.getElementById("source-select");
const customUrl = document.getElementById("custom-url");
const statusText = document.getElementById("status");
const qualityControls = document.getElementById("quality-controls");
const requestLog = document.getElementById("request-log");

const metrics = {
  bitrate: document.getElementById("metric-bitrate"),
  resolution: document.getElementById("metric-resolution"),
  videoCodec: document.getElementById("metric-video-codec"),
  audioCodec: document.getElementById("metric-audio-codec"),
};

let hls = null;
let activeUrl = sourceSelect.value;
const maxRequestRows = 40;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function resetMetrics() {
  metrics.bitrate.textContent = "-";
  metrics.resolution.textContent = "-";
  metrics.videoCodec.textContent = "-";
  metrics.audioCodec.textContent = "-";
}

function logRequest(url) {
  if (!url) {
    return;
  }

  const row = document.createElement("li");
  row.textContent = url;
  requestLog.prepend(row);

  while (requestLog.children.length > maxRequestRows) {
    requestLog.lastElementChild.remove();
  }
}

function formatBitrate(bitsPerSecond) {
  if (!bitsPerSecond) {
    return "-";
  }

  if (bitsPerSecond >= 1000000) {
    return `${(bitsPerSecond / 1000000).toFixed(2)} Mbps`;
  }

  return `${Math.round(bitsPerSecond / 1000)} kbps`;
}

function levelLabel(level, index) {
  const height = level.height ? `${level.height}p` : `Level ${index}`;
  return `${height} ${formatBitrate(level.bitrate)}`;
}

function updateMetrics(levelIndex) {
  if (!hls || !hls.levels || levelIndex < 0) {
    metrics.resolution.textContent = video.videoWidth && video.videoHeight
      ? `${video.videoWidth} x ${video.videoHeight}`
      : "-";
    return;
  }

  const level = hls.levels[levelIndex];
  if (!level) {
    return;
  }

  metrics.bitrate.textContent = formatBitrate(level.bitrate);
  metrics.resolution.textContent = level.width && level.height
    ? `${level.width} x ${level.height}`
    : "-";
  metrics.videoCodec.textContent = level.videoCodec || level.codecSet || "-";
  metrics.audioCodec.textContent = level.audioCodec || "-";
}

function renderQualityControls() {
  qualityControls.replaceChildren();

  const autoButton = document.createElement("button");
  autoButton.type = "button";
  autoButton.textContent = "Auto";
  autoButton.className = "secondary";
  autoButton.classList.toggle("active", !hls || hls.currentLevel === -1);
  autoButton.addEventListener("click", () => {
    if (!hls) {
      return;
    }
    hls.currentLevel = -1;
    setStatus("Automatic quality selection");
    renderQualityControls();
  });
  qualityControls.append(autoButton);

  if (!hls || !hls.levels || hls.levels.length === 0) {
    return;
  }

  hls.levels.forEach((level, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = levelLabel(level, index);
    button.classList.toggle("active", hls.currentLevel === index);
    button.addEventListener("click", () => {
      hls.currentLevel = index;
      setStatus(`Manual quality: ${levelLabel(level, index)}`);
      updateMetrics(index);
      renderQualityControls();
    });
    qualityControls.append(button);
  });
}

function attachHlsEvents(instance) {
  instance.on(Hls.Events.MANIFEST_LOADING, (_event, data) => {
    logRequest(data.url);
    setStatus("Loading manifest");
  });

  instance.on(Hls.Events.LEVEL_LOADING, (_event, data) => {
    logRequest(data.url);
  });

  instance.on(Hls.Events.FRAG_LOADING, (_event, data) => {
    logRequest(data.frag && data.frag.url);
  });

  instance.on(Hls.Events.MANIFEST_PARSED, () => {
    setStatus("Manifest loaded");
    renderQualityControls();
    updateMetrics(instance.currentLevel);
    video.play().catch(() => {
      setStatus("Manifest loaded. Press play to start.");
    });
  });

  instance.on(Hls.Events.LEVEL_SWITCHED, (_event, data) => {
    updateMetrics(data.level);
    renderQualityControls();
  });

  instance.on(Hls.Events.ERROR, (_event, data) => {
    const detail = data.details || data.type || "unknown error";
    setStatus(`HLS error: ${detail}`, true);
  });
}

function destroyHls() {
  if (hls) {
    hls.destroy();
    hls = null;
  }
}

function loadSource(url) {
  activeUrl = url;
  destroyHls();
  resetMetrics();
  requestLog.replaceChildren();
  video.removeAttribute("src");
  video.load();

  if (window.Hls && Hls.isSupported()) {
    hls = new Hls({
      capLevelToPlayerSize: false,
      enableWorker: true,
    });
    attachHlsEvents(hls);
    hls.loadSource(url);
    hls.attachMedia(video);
    renderQualityControls();
    return;
  }

  if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = url;
    logRequest(url);
    setStatus("Using native HLS playback");
    video.play().catch(() => {
      setStatus("Source loaded. Press play to start.");
    });
    renderQualityControls();
    return;
  }

  setStatus("This browser does not support HLS playback", true);
  renderQualityControls();
}

sourceForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const url = customUrl.value.trim() || sourceSelect.value;
  loadSource(url);
});

sourceSelect.addEventListener("change", () => {
  customUrl.value = "";
  loadSource(sourceSelect.value);
});

video.addEventListener("loadedmetadata", () => {
  if (!hls) {
    metrics.resolution.textContent = `${video.videoWidth} x ${video.videoHeight}`;
  }
});

video.addEventListener("error", () => {
  setStatus("Video element error", true);
});

loadSource(activeUrl);
