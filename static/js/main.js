let currentVideoId = null;
let currentSegments = [];
let pollTimer = null;
let elapsedTimer = null;
let jobStartTime = null;

// ---------------- Toast notifications ---------------- //
function showToast(message, type = 'info', duration = 4000) {
    const host = document.getElementById('toastHost');
    const el = document.createElement('div');
    el.className = `toast-item ${type}`;
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => el.remove(), duration);
}

// ---------------- Drag & drop upload ---------------- //
const dropZone = document.getElementById('dropZone');
const videoFileInput = document.getElementById('videoFile');
const fileNameLabel = document.getElementById('fileNameLabel');

dropZone.addEventListener('click', () => videoFileInput.click());

['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    })
);

['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    })
);

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) {
        videoFileInput.files = files;
        fileNameLabel.textContent = files[0].name;
    }
});

videoFileInput.addEventListener('change', () => {
    fileNameLabel.textContent = videoFileInput.files[0] ? videoFileInput.files[0].name : 'No file selected';
});

// ---------------- Quality presets ---------------- //
document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const preset = QUALITY_PRESETS[btn.dataset.preset];
        if (!preset) return;
        document.getElementById('modelSize').value = preset.model_size;
        document.getElementById('videoQuality').value = preset.video_quality;
        document.getElementById('summaryLength').value = preset.summary_length;
        document.getElementById('numParts').value = preset.num_parts;
        document.getElementById('beamSize').value = preset.beam_size;
    });
});

// If the user manually changes a dropdown, no preset is "active" anymore
['modelSize', 'videoQuality', 'summaryLength', 'numParts', 'beamSize'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    });
});

// ---------------- Progress UI helpers ---------------- //
function setProgress(pct, stage) {
    const wrap = document.getElementById('progressWrap');
    wrap.classList.remove('d-none');
    document.getElementById('progressBar').style.width = `${pct}%`;
    document.getElementById('progressPct').textContent = `${pct}%`;
    document.getElementById('progressStage').textContent = stage;
}

function setLive(isLive) {
    document.getElementById('recDot').classList.toggle('live', isLive);
    document.getElementById('headerWaveform').classList.toggle('live', isLive);
}

function resetProgress() {
    document.getElementById('progressWrap').classList.add('d-none');
    document.getElementById('progressBar').style.width = '0%';
    stopElapsedTimer();
}

function startElapsedTimer() {
    jobStartTime = Date.now();
    document.getElementById('progressElapsed').textContent = '0:00';
    if (elapsedTimer) clearInterval(elapsedTimer);
    elapsedTimer = setInterval(() => {
        const secs = Math.floor((Date.now() - jobStartTime) / 1000);
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        document.getElementById('progressElapsed').textContent = `${m}:${String(s).padStart(2, '0')}`;
    }, 1000);
}

function stopElapsedTimer() {
    if (elapsedTimer) {
        clearInterval(elapsedTimer);
        elapsedTimer = null;
    }
}

// ---------------- Submit & poll job ---------------- //
document.getElementById('processForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('videoFile').files[0];
    const urlInput = document.getElementById('videoUrl').value.trim();

    if (!fileInput && !urlInput) {
        showToast('Please choose a file or paste a video URL first.', 'error');
        return;
    }

    if (fileInput) formData.append('file', fileInput);
    if (urlInput) formData.append('url', urlInput);
    formData.append('model_size', document.getElementById('modelSize').value);
    formData.append('video_quality', document.getElementById('videoQuality').value);
    formData.append('summary_length', document.getElementById('summaryLength').value);
    formData.append('num_parts', document.getElementById('numParts').value);
    formData.append('beam_size', document.getElementById('beamSize').value);
    formData.append('translate_to_english', document.getElementById('translateEnglish').checked ? 'true' : 'false');
    formData.append('keep_original_too', document.getElementById('keepOriginalToo').checked ? 'true' : 'false');

    document.getElementById('processBtn').disabled = true;
    setProgress(2, 'Uploading / queuing job…');
    document.getElementById('resultsSection').classList.add('d-none');

    try {
        const response = await fetch('/api/process', { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to start processing.');
        }

        showToast('Processing started — you can watch progress below.', 'info');
        startElapsedTimer();
        setLive(true);
        pollJob(data.job_id);
    } catch (err) {
        document.getElementById('processBtn').disabled = false;
        stopElapsedTimer();
        setLive(false);
        resetProgress();
        showToast(err.message || 'An error occurred starting the job.', 'error');
    }
});

function pollJob(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${jobId}`);
            const job = await res.json();

            if (res.status === 404) {
                clearInterval(pollTimer);
                stopElapsedTimer();
                setLive(false);
                showToast('Lost track of this job — please try again.', 'error');
                document.getElementById('processBtn').disabled = false;
                return;
            }

            setProgress(job.progress || 0, job.stage || 'Working…');

            if (job.status === 'done') {
                clearInterval(pollTimer);
                stopElapsedTimer();
                setLive(false);
                document.getElementById('processBtn').disabled = false;
                renderResult(job.result);
                showToast('Video processed successfully!', 'success');
            } else if (job.status === 'error') {
                clearInterval(pollTimer);
                stopElapsedTimer();
                setLive(false);
                document.getElementById('processBtn').disabled = false;
                showToast(`Processing failed: ${job.error}`, 'error', 6000);
            }
        } catch (err) {
            clearInterval(pollTimer);
            stopElapsedTimer();
            setLive(false);
            document.getElementById('processBtn').disabled = false;
            showToast('Lost connection while checking job status.', 'error');
        }
    }, 1200);
}

// ---------------- Render results ---------------- //
function renderResult(data) {
    currentVideoId = data.video_id;
    currentSegments = data.segments || [];

    const player = document.getElementById('videoPlayer');
    player.src = data.video_path;
    document.getElementById('downloadBtn').href = data.video_path;

    document.getElementById('summaryText').innerText = data.summary;

    const langBadge = document.getElementById('languageBadge');
    if (data.detected_language_name) {
        const modeLabel = data.task === 'translate' ? '→ English' : '(original)';
        langBadge.textContent = `Detected: ${data.detected_language_name} ${modeLabel}`;
        langBadge.classList.remove('d-none');
    } else {
        langBadge.classList.add('d-none');
    }

    const originalWrap = document.getElementById('originalTranscriptWrap');
    const originalText = document.getElementById('originalTranscriptText');
    const toggleBtn = document.getElementById('toggleOriginalBtn');
    if (data.original_transcript) {
        originalWrap.classList.remove('d-none');
        originalText.textContent = data.original_transcript;
        originalText.classList.add('d-none');
        toggleBtn.textContent = 'Show original-language transcript';
        toggleBtn.onclick = () => {
            const hidden = originalText.classList.toggle('d-none');
            toggleBtn.textContent = hidden ? 'Show original-language transcript' : 'Hide original-language transcript';
        };
    } else {
        originalWrap.classList.add('d-none');
    }

    const partsContainer = document.getElementById('partsContainer');
    partsContainer.innerHTML = '';
    data.parts.forEach((part, i) => {
        const div = document.createElement('div');
        div.className = 'segment-card';
        div.innerHTML = `
            <div>
                <p class="segment-card__title">${escapeHtml(part.title)}</p>
                <p class="segment-card__summary">${escapeHtml(part.summary)}</p>
            </div>
            <div class="segment-card__slide">
                ${part.slide ? `<img src="${part.slide}" alt="Key frame for ${escapeHtml(part.title)}">` : '<div class="segment-card__empty">no slide</div>'}
            </div>
        `;
        partsContainer.appendChild(div);
    });

    renderTranscript(currentSegments);

    document.getElementById('resultsSection').classList.remove('d-none');
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function renderTranscript(segments) {
    const box = document.getElementById('transcriptContainer');
    box.innerHTML = '';
    segments.forEach((seg, idx) => {
        const line = document.createElement('div');
        line.className = 'transcript-line';
        line.dataset.start = seg.start;
        line.dataset.end = seg.end;
        line.innerHTML = `<span class="transcript-time">${formatTime(seg.start)}</span><span>${escapeHtml(seg.text.trim())}</span>`;
        line.addEventListener('click', () => {
            const player = document.getElementById('videoPlayer');
            player.currentTime = seg.start;
            player.play();
        });
        box.appendChild(line);
    });

   // Highlight the active line as the video plays (scrolls only inside transcript box)
const player = document.getElementById('videoPlayer');
player.addEventListener('timeupdate', () => {
  const t = player.currentTime;
  const box = document.getElementById('transcriptContainer');
  document.querySelectorAll('.transcript-line').forEach(line => {
    const s = parseFloat(line.dataset.start);
    const eTime = parseFloat(line.dataset.end);
    const isActive = t >= s && t < eTime;
    const wasActive = line.classList.contains('active');
    line.classList.toggle('active', isActive);
    if (isActive && !wasActive && box && !box.matches(':hover')) {
      const targetTop = line.offsetTop - box.offsetTop - (box.clientHeight / 2) + (line.clientHeight / 2);
      box.scrollTo({ top: targetTop, behavior: 'smooth' });
    }
  });
});
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str ?? '';
    return div.innerHTML;
}

// ---------------- Copy transcript ---------------- //
document.getElementById('copyTranscriptBtn').addEventListener('click', async () => {
    if (!currentSegments.length) return;
    const fullText = currentSegments.map(s => s.text.trim()).join(' ');
    try {
        await navigator.clipboard.writeText(fullText);
        showToast('Transcript copied to clipboard.', 'success', 2500);
    } catch (err) {
        showToast('Could not copy — please select and copy manually.', 'error');
    }
});

// ---------------- Keyword search ---------------- //
document.getElementById('searchBtn').addEventListener('click', runSearch);
document.getElementById('searchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runSearch();
});

async function runSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query || !currentVideoId) {
        showToast('Process a video first, then search its transcript.', 'info');
        return;
    }

    const searchBtn = document.getElementById('searchBtn');
    searchBtn.disabled = true;
    searchBtn.textContent = 'Searching…';

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: currentVideoId, query }),
        });
        const data = await response.json();
        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = '';

        if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = `<div class="empty-state">No matching transcript terms found.</div>`;
            return;
        }

        data.results.forEach(res => {
            const div = document.createElement('div');
            div.className = 'search-result';
            div.innerHTML = `
                <div>
                    <span class="chip accent result-jump" style="cursor:pointer;">⏱ ${res.timestamp}</span>
                    <blockquote>"${escapeHtml(res.text.trim())}"</blockquote>
                </div>
                <div>
                    ${res.slide_url ? `<img src="${res.slide_url}" alt="Matched slide">` : ''}
                </div>
            `;
            div.querySelector('.result-jump').addEventListener('click', () => {
                const player = document.getElementById('videoPlayer');
                player.currentTime = res.start_time;
                player.play();
                document.getElementById('videoPlayer').scrollIntoView({ behavior: 'smooth', block: 'center' });
            });
            resultsContainer.appendChild(div);
        });
    } catch (err) {
        showToast('Search failed — please try again.', 'error');
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = 'Search';
    }
}
