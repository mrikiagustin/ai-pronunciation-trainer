"use strict";

const SHADOWING_SENTENCES = [
    "Think about the thing you want.",
    "The three brothers think alike.",
    "This thing is thick.",
    "Those three things are theirs.",
    "They think that the weather is cold.",
    "The thief took three thick books.",
    "I think this is the right thing.",
    "That thing belongs to them.",
    "These three things are thirty dollars.",
    "The weather is better than yesterday.",
    "I think that they will come this Thursday.",
    "The mother and father thanked their three children.",
    "They walked through the thick forest.",
    "I thought that the weather would be warmer.",
    "The author wrote three thoughtful stories."
];

const state = {
    currentIndex: 0,
    recorder: null,
    stream: null,
    isRecording: false,
    isStarting: false,
    sessionId: 0,
    assessmentQueue: Promise.resolve(),
    results: SHADOWING_SENTENCES.map(() => ({ status: "pending", score: null })),
    recordingUrls: []
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
    cacheElements();
    renderQueue();
    renderCurrentSentence();
    loadVoices();

    elements.recordControl.addEventListener("click", toggleRecording);
    elements.restartButton.addEventListener("click", restartSession);
    document.addEventListener("keydown", handleKeyboardShortcut);
    window.addEventListener("beforeunload", releaseResources);
});

function cacheElements() {
    elements.practiceCard = document.getElementById("practice-card");
    elements.recordControl = document.getElementById("record-control");
    elements.sentenceNumber = document.getElementById("sentence-number");
    elements.sentenceText = document.getElementById("sentence-text");
    elements.sentenceHint = document.getElementById("sentence-hint");
    elements.statusChip = document.getElementById("status-chip");
    elements.controlTitle = document.getElementById("control-title");
    elements.controlSubtitle = document.getElementById("control-subtitle");
    elements.spaceAction = document.getElementById("space-action");
    elements.liveMessage = document.getElementById("live-message");
    elements.progressText = document.getElementById("progress-text");
    elements.progressBar = document.getElementById("progress-bar");
    elements.queue = document.getElementById("sentence-queue");
    elements.assessmentCount = document.getElementById("assessment-count");
    elements.resultsPanel = document.getElementById("results-panel");
    elements.resultsStatus = document.getElementById("results-status");
    elements.averageScore = document.getElementById("average-score");
    elements.restartButton = document.getElementById("restart-button");
}

function loadVoices() {
    if (!("speechSynthesis" in window)) {
        return;
    }
    window.speechSynthesis.getVoices();
    window.speechSynthesis.addEventListener?.("voiceschanged", () => window.speechSynthesis.getVoices(), { once: true });
}

function handleKeyboardShortcut(event) {
    if (event.code !== "Space" || event.repeat || event.altKey || event.ctrlKey || event.metaKey) {
        return;
    }

    const tagName = event.target.tagName;
    if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") {
        return;
    }

    event.preventDefault();
    toggleRecording();
}

async function toggleRecording() {
    if (state.currentIndex >= SHADOWING_SENTENCES.length || state.isStarting) {
        return;
    }

    if (state.isRecording) {
        stopRecordingAndAdvance();
    } else {
        await startShadowing();
    }
}

async function startShadowing() {
    state.isStarting = true;
    setControlDisabled(true);
    setMessage("Menyiapkan mikrofon… izinkan akses jika browser memintanya.");

    try {
        const stream = await getMicrophoneStream();
        const recorderOptions = getRecorderOptions();
        const recorder = recorderOptions ? new MediaRecorder(stream, recorderOptions) : new MediaRecorder(stream);

        state.recorder = recorder;
        const recordingChunks = [];

        recorder.addEventListener("dataavailable", event => {
            if (event.data.size > 0) {
                recordingChunks.push(event.data);
            }
        });

        const sentenceIndex = state.currentIndex;
        const sessionId = state.sessionId;
        recorder.addEventListener("stop", () => finalizeRecording(sentenceIndex, recorder.mimeType, recordingChunks, sessionId), { once: true });
        recorder.start(250);
        state.isRecording = true;
        speakReference(SHADOWING_SENTENCES[sentenceIndex], sentenceIndex, sessionId);
        renderRecordingState();
    } catch (error) {
        console.error("Could not start shadowing:", error);
        setMessage(microphoneErrorMessage(error), true);
    } finally {
        state.isStarting = false;
        setControlDisabled(false);
    }
}

async function getMicrophoneStream() {
    if (!navigator.mediaDevices?.getUserMedia || !("MediaRecorder" in window)) {
        throw new Error("unsupported-browser");
    }

    if (state.stream?.active) {
        return state.stream;
    }

    state.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
        }
    });
    return state.stream;
}

function getRecorderOptions() {
    if (typeof MediaRecorder.isTypeSupported !== "function") {
        return undefined;
    }

    const preferredTypes = [
        "audio/ogg;codecs=opus",
        "audio/webm;codecs=opus",
        "audio/webm"
    ];
    const mimeType = preferredTypes.find(type => MediaRecorder.isTypeSupported(type));
    return mimeType ? { mimeType } : undefined;
}

function speakReference(text, sentenceIndex, sessionId) {
    if (!("speechSynthesis" in window)) {
        setMessage("Perekaman berjalan, tetapi browser ini tidak mendukung suara referensi.", true);
        return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    utterance.voice = voices.find(voice => voice.lang?.toLowerCase().startsWith("en-us"))
        || voices.find(voice => voice.lang?.toLowerCase().startsWith("en"))
        || null;
    utterance.lang = "en-US";
    utterance.rate = 0.82;
    utterance.pitch = 1;
    utterance.volume = 1;
    utterance.addEventListener("error", event => {
        const wasCancelled = event.error === "canceled" || event.error === "interrupted";
        const isCurrentRecording = state.isRecording
            && state.currentIndex === sentenceIndex
            && state.sessionId === sessionId;
        if (!wasCancelled && isCurrentRecording) {
            setMessage("Suara referensi gagal diputar. Rekaman tetap berjalan; tekan spasi untuk berhenti.", true);
        }
    }, { once: true });
    window.speechSynthesis.speak(utterance);
}

function stopRecordingAndAdvance() {
    if (!state.recorder || state.recorder.state === "inactive") {
        return;
    }

    window.speechSynthesis?.cancel();
    state.isRecording = false;
    state.recorder.stop();

    const finishedIndex = state.currentIndex;
    state.results[finishedIndex] = { status: "queued", score: null };
    state.currentIndex += 1;
    renderProgress();
    renderQueue();

    if (state.currentIndex < SHADOWING_SENTENCES.length) {
        renderCurrentSentence();
        setMessage("Rekaman sebelumnya sedang dinilai di background. Tekan spasi saat siap melanjutkan.");
    } else {
        showSessionComplete();
    }
}

function finalizeRecording(sentenceIndex, recorderMimeType, recordingChunks, sessionId) {
    if (sessionId !== state.sessionId) {
        return;
    }

    const mimeType = recorderMimeType || recordingChunks[0]?.type || "audio/webm";
    const blob = new Blob(recordingChunks, { type: mimeType });

    if (blob.size < 1000) {
        state.results[sentenceIndex] = { status: "error", score: null };
        renderQueue();
        updateResultsSummary();
        return;
    }

    state.recordingUrls[sentenceIndex] = URL.createObjectURL(blob);
    enqueueAssessment(sentenceIndex, blob, sessionId);
}

function enqueueAssessment(sentenceIndex, blob, sessionId) {
    state.results[sentenceIndex] = { status: "processing", score: null };
    renderQueue();

    state.assessmentQueue = state.assessmentQueue
        .then(() => submitAssessment(sentenceIndex, blob, sessionId))
        .catch(error => {
            console.error("Assessment queue error:", error);
        });
}

async function submitAssessment(sentenceIndex, blob, sessionId) {
    if (sessionId !== state.sessionId) {
        return;
    }

    try {
        const base64Audio = await blobToDataUrl(blob);
        const response = await fetch(document.body.dataset.assessmentUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                title: SHADOWING_SENTENCES[sentenceIndex],
                base64Audio,
                language: "en"
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        const score = Number(data.pronunciation_accuracy);
        if (!Number.isFinite(score)) {
            throw new Error("Invalid assessment response");
        }

        if (sessionId === state.sessionId) {
            state.results[sentenceIndex] = { status: "done", score };
        }
    } catch (error) {
        console.error(`Assessment ${sentenceIndex + 1} failed:`, error);
        if (sessionId === state.sessionId) {
            state.results[sentenceIndex] = { status: "error", score: null };
        }
    }

    if (sessionId !== state.sessionId) {
        return;
    }
    renderQueue();
    updateResultsSummary();
}

function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.addEventListener("load", () => resolve(reader.result), { once: true });
        reader.addEventListener("error", () => reject(reader.error), { once: true });
        reader.readAsDataURL(blob);
    });
}

function renderRecordingState() {
    elements.practiceCard.classList.add("is-recording");
    elements.statusChip.querySelector("span").textContent = "Sedang merekam";
    elements.controlTitle.textContent = "Tekan untuk berhenti";
    elements.controlSubtitle.textContent = "rekaman akan langsung masuk antrean";
    elements.spaceAction.textContent = "Berhenti & lanjut";
    elements.recordControl.setAttribute("aria-label", "Berhenti merekam dan lanjut ke kalimat berikutnya");
    setMessage("Suara referensi sedang diputar. Ikuti ucapannya, lalu tekan spasi saat selesai.");
}

function renderCurrentSentence() {
    const index = state.currentIndex;
    const sentence = SHADOWING_SENTENCES[index];
    elements.practiceCard.classList.remove("is-recording");
    elements.sentenceNumber.textContent = `Kalimat ${String(index + 1).padStart(2, "0")}`;
    elements.sentenceText.textContent = sentence;
    elements.sentenceHint.innerHTML = hintForSentence(sentence);
    elements.statusChip.querySelector("span").textContent = "Siap dimulai";
    elements.controlTitle.textContent = "Tekan untuk mulai";
    elements.controlSubtitle.textContent = "atau gunakan tombol spasi";
    elements.spaceAction.textContent = "Putar & rekam";
    elements.recordControl.setAttribute("aria-label", `Mulai shadowing kalimat ${index + 1}`);
}

function hintForSentence(sentence) {
    if (/\b(thought|think|thick|three|thing|thirty|Thursday|thanked|through|author)\b/i.test(sentence)) {
        return "Fokus pada bunyi <strong>th</strong> dan irama kalimat.";
    }
    return "Jaga tempo, penekanan kata, dan intonasi suara referensi.";
}

function renderProgress() {
    const completed = Math.min(state.currentIndex, SHADOWING_SENTENCES.length);
    elements.progressText.textContent = `${completed} / ${SHADOWING_SENTENCES.length}`;
    elements.progressBar.style.width = `${(completed / SHADOWING_SENTENCES.length) * 100}%`;
}

function renderQueue() {
    elements.queue.replaceChildren(...SHADOWING_SENTENCES.map((sentence, index) => {
        const item = document.createElement("li");
        item.className = "queue-item";
        if (index < state.currentIndex) item.classList.add("done");
        if (index === state.currentIndex && state.currentIndex < SHADOWING_SENTENCES.length) item.classList.add("current");
        if (index === state.currentIndex) item.setAttribute("aria-current", "step");

        const number = document.createElement("span");
        number.className = "number";
        number.textContent = String(index + 1).padStart(2, "0");

        const copy = document.createElement("span");
        copy.className = "copy";
        copy.textContent = sentence;
        copy.title = sentence;

        const result = document.createElement("span");
        const assessment = state.results[index];
        result.className = "item-result";
        if (assessment.status === "done") {
            result.textContent = `${Math.round(assessment.score)}%`;
            result.setAttribute("aria-label", `Skor ${Math.round(assessment.score)} persen`);
        } else if (assessment.status === "processing" || assessment.status === "queued") {
            result.classList.add("loading");
            result.setAttribute("aria-label", "Sedang dinilai");
        } else if (assessment.status === "error") {
            result.classList.add("error");
            result.textContent = "!";
            result.setAttribute("aria-label", "Penilaian gagal");
        }

        item.append(number, copy, result);
        return item;
    }));

    const currentItem = elements.queue.querySelector(".current");
    currentItem?.scrollIntoView({ block: "nearest", behavior: "smooth" });

    const assessed = state.results.filter(result => result.status === "done").length;
    elements.assessmentCount.textContent = `${assessed} dinilai`;
}

function showSessionComplete() {
    elements.practiceCard.classList.remove("is-recording");
    elements.practiceCard.hidden = true;
    elements.resultsPanel.hidden = false;
    updateResultsSummary();
    elements.resultsPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function updateResultsSummary() {
    const completedResults = state.results.filter(result => result.status === "done");
    const failedResults = state.results.filter(result => result.status === "error").length;
    const pendingResults = state.results.length - completedResults.length - failedResults;

    if (completedResults.length) {
        const average = completedResults.reduce((sum, result) => sum + result.score, 0) / completedResults.length;
        elements.averageScore.textContent = `${Math.round(average)}%`;
    } else {
        elements.averageScore.textContent = "—";
    }

    if (pendingResults > 0) {
        elements.resultsStatus.textContent = `${completedResults.length} dari ${state.results.length} penilaian selesai. Sisanya tetap berjalan di background.`;
    } else if (failedResults > 0) {
        elements.resultsStatus.textContent = `${completedResults.length} penilaian selesai, ${failedResults} gagal diproses.`;
    } else {
        elements.resultsStatus.textContent = "Semua penilaian selesai. Skor rata-rata sudah diperbarui.";
    }
}

function restartSession() {
    window.speechSynthesis?.cancel();
    state.sessionId += 1;
    state.assessmentQueue = Promise.resolve();
    state.recordingUrls.forEach(url => {
        if (url) URL.revokeObjectURL(url);
    });
    state.recordingUrls = [];
    state.currentIndex = 0;
    state.isRecording = false;
    state.isStarting = false;
    state.results = SHADOWING_SENTENCES.map(() => ({ status: "pending", score: null }));
    elements.practiceCard.hidden = false;
    elements.resultsPanel.hidden = true;
    renderProgress();
    renderQueue();
    renderCurrentSentence();
    setMessage("Tekan spasi untuk memutar audio referensi dan mulai merekam.");
}

function setControlDisabled(disabled) {
    elements.recordControl.disabled = disabled;
    elements.recordControl.setAttribute("aria-busy", String(disabled));
}

function setMessage(message, isError = false) {
    elements.liveMessage.textContent = message;
    elements.liveMessage.classList.toggle("is-error", isError);
}

function microphoneErrorMessage(error) {
    if (error.message === "unsupported-browser") {
        return "Browser ini belum mendukung perekaman. Gunakan Chrome, Edge, atau Firefox versi terbaru.";
    }
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
        return "Akses mikrofon ditolak. Izinkan mikrofon pada pengaturan situs, lalu coba lagi.";
    }
    if (error.name === "NotFoundError") {
        return "Mikrofon tidak ditemukan. Sambungkan mikrofon lalu coba lagi.";
    }
    return "Mikrofon belum dapat digunakan. Periksa perangkat audio lalu coba lagi.";
}

function releaseResources() {
    window.speechSynthesis?.cancel();
    state.stream?.getTracks().forEach(track => track.stop());
    state.recordingUrls.forEach(url => {
        if (url) URL.revokeObjectURL(url);
    });
}
