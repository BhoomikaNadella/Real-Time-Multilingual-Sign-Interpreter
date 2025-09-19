// ASL Sign Recognition - Main JavaScript File

// --- Global State ---
let isCameraActive = true;
let sessionStats = {
  signsDetected: 0,
  totalConfidence: 0,
  predictionCount: 0,
};
let apiErrorCount = 0;
const MAX_API_ERRORS = 5;

// --- DOM Element Selection ---
const themeToggle = document.getElementById("theme-toggle");
const body = document.body;
const videoFeed = document.getElementById("video_feed");
const skeletonLoader = document.getElementById("skeleton-loader");
const statusIndicator = document.getElementById("status-indicator");
const toggleCameraBtn = document.getElementById("toggle-camera-btn");
const screenshotBtn = document.getElementById("screenshot-btn");
const fullscreenBtn = document.getElementById("fullscreen-btn");
const videoWrapper = document.getElementById("video-wrapper");
const languageSelect = document.getElementById("language-select");
const languageForm = document.getElementById("language-form");
const helpBtn = document.getElementById("help-btn");
const helpModal = document.getElementById("help-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const predictionDisplay = document.getElementById("prediction-display");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceText = document.getElementById("confidence-text");
const signsDetectedElement = document.getElementById("signs-detected");
const avgConfidenceElement = document.getElementById("avg-confidence");
const flashContainer = document.getElementById("flash-container");

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  initializeTheme();
  initializeVideoFeed();
  initializeCameraControls();
  initializeLanguageSelection();
  initializeHelpModal();
  initializeKeyboardShortcuts();

  // Start polling for prediction data
  const predictionInterval = setInterval(() => {
    if (isCameraActive) {
      fetchPredictionData(predictionInterval);
    }
  }, 500); // Fetch data every 500ms

  // Update session stats every second
  setInterval(updateSessionStats, 1000);

  // Auto-remove initial Flask flash messages
  setTimeout(clearFlashMessages, 3000);
});

// --- Core Functions ---

function initializeTheme() {
  const savedTheme = localStorage.getItem("theme") || "light";
  body.setAttribute("data-theme", savedTheme);

  themeToggle.addEventListener("click", () => {
    const currentTheme = body.getAttribute("data-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    body.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  });
}

function initializeVideoFeed() {
  videoFeed.addEventListener("load", () => {
    skeletonLoader.style.display = "none";
    videoFeed.style.display = "block";
    updateConnectionStatus(true);
  });
  videoFeed.addEventListener("error", () => {
    skeletonLoader.style.display = "block";
    videoFeed.style.display = "none";
    updateConnectionStatus(false);
  });
}

function initializeCameraControls() {
  toggleCameraBtn.addEventListener("click", () => {
    isCameraActive = !isCameraActive;
    videoFeed.style.visibility = isCameraActive ? "visible" : "hidden";
    const icon = toggleCameraBtn.querySelector("i");
    const text = toggleCameraBtn.querySelector("span");
    if (isCameraActive) {
      icon.className = "fas fa-pause btn-icon";
      text.textContent = "Pause";
      showFlashMessage("Camera resumed");
    } else {
      icon.className = "fas fa-play btn-icon";
      text.textContent = "Resume";
      showFlashMessage("Camera paused");
    }
  });

  screenshotBtn.addEventListener("click", takeScreenshot);
  fullscreenBtn.addEventListener("click", toggleFullscreen);
}

function initializeLanguageSelection() {
  languageSelect.addEventListener("change", () => {
    languageForm.submit();
  });
}

function initializeHelpModal() {
  helpBtn.addEventListener("click", () => helpModal.showModal());
  closeModalBtn.addEventListener("click", () => helpModal.close());
  helpModal.addEventListener("click", (e) => {
    if (e.target === helpModal) helpModal.close();
  });
}

function initializeKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (document.activeElement.tagName === "SELECT") return;
    switch (e.code) {
      case "Space":
        e.preventDefault();
        toggleCameraBtn.click();
        break;
      case "KeyS":
        screenshotBtn.click();
        break;
      case "KeyF":
        fullscreenBtn.click();
        break;
      case "Escape":
        if (helpModal.open) helpModal.close();
        break;
    }
  });
}

async function fetchPredictionData(intervalId) {
  try {
    const response = await fetch("/api/prediction");
    if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

    const data = await response.json();
    updatePredictionUI(data.prediction, data.confidence);

    if (
      data.prediction !== "Ready..." &&
      data.prediction !== "..." &&
      data.prediction !== "Low Confidence"
    ) {
      sessionStats.signsDetected++;
      sessionStats.totalConfidence += data.confidence;
      sessionStats.predictionCount++;
    }
    apiErrorCount = 0; // Reset on success
  } catch (error) {
    console.error("Error fetching prediction:", error);
    apiErrorCount++;
    if (apiErrorCount > MAX_API_ERRORS) {
      updatePredictionUI("Connection Lost", 0);
      clearInterval(intervalId); // Stop polling
    }
  }
}

// --- UI Update Functions ---

function updatePredictionUI(prediction, confidence) {
  predictionDisplay.textContent = prediction;
  const confidencePercent = Math.round(confidence * 100);
  confidenceText.textContent = `${confidencePercent}%`;
  confidenceFill.style.width = `${confidencePercent}%`;
}

function updateConnectionStatus(connected) {
  const statusText = statusIndicator.querySelector("span");
  if (connected) {
    statusIndicator.classList.add("status-connected");
    statusText.textContent = "Camera Connected";
  } else {
    statusIndicator.classList.remove("status-connected");
    statusText.textContent = "Camera Disconnected";
  }
}

function updateSessionStats() {
  signsDetectedElement.textContent = sessionStats.signsDetected;
  if (sessionStats.predictionCount > 0) {
    const avg =
      (sessionStats.totalConfidence / sessionStats.predictionCount) * 100;
    avgConfidenceElement.textContent = `${avg.toFixed(0)}%`;
  }
}

function showFlashMessage(message, type = "success") {
  const flash = document.createElement("div");
  flash.className = "flash";
  flash.innerHTML = `<i class="fas fa-check-circle"></i> <span>${message}</span>`;
  flashContainer.appendChild(flash);
  setTimeout(() => {
    flash.style.animation = "slideOut 0.3s ease forwards";
    setTimeout(() => flash.remove(), 300);
  }, 3000);
}

function clearFlashMessages() {
  const flashMessages = document.querySelectorAll(".flash");
  flashMessages.forEach((flash) => {
    flash.style.animation = "slideOut 0.3s ease forwards";
    setTimeout(() => flash.remove(), 300);
  });
}

// --- Helper Functions ---

function takeScreenshot() {
  const canvas = document.createElement("canvas");
  canvas.width = videoFeed.naturalWidth || 640;
  canvas.height = videoFeed.naturalHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
  const link = document.createElement("a");
  const timestamp = new Date().toISOString().replace(/:/g, "-").slice(0, 19);
  link.download = `asl-capture-${timestamp}.png`;
  link.href = canvas.toDataURL();
  link.click();
  showFlashMessage("Screenshot saved!");
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    videoWrapper.requestFullscreen().catch((err) => {
      alert(
        `Error attempting to enable full-screen mode: ${err.message} (${err.name})`
      );
    });
  } else {
    document.exitFullscreen();
  }
}
