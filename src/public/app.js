// ═══════════════════════════════════════════════════════════════
//  PostureCheck — Client Application
//  MediaPipe Pose Landmarker with training-based posture classification
//  Uses 3D WorldLandmarks for camera-angle-invariant metrics
// ═══════════════════════════════════════════════════════════════

import { PoseLandmarker, FilesetResolver } from "/vision_bundle.mjs";

// ── DOM Elements ──────────────────────────────────────

const $ = (id) => document.getElementById(id);

const video       = $("webcam");
const canvas      = $("overlay");
const ctx         = canvas.getContext("2d");
const statusEl    = $("status");
const alertOverlay = $("alert-overlay");
const loadingOverlay = $("loading-overlay");
const scoreValueEl = $("score-value");

// Training
const trainGoodBtn  = $("train-good-btn");
const trainBadBtn   = $("train-bad-btn");
const trainClearBtn = $("train-clear-btn");
const trainUndoBtn  = $("train-undo-btn");
const goodCountEl   = $("good-count");
const badCountEl    = $("bad-count");
const trainingStatusEl = $("training-status");
const sampleListEl  = $("sample-list");

// Sliders
const threshScoreEl   = $("thresh-score");
const alertDelayEl    = $("alert-delay");
const alertCooldownEl = $("alert-cooldown");
const soundEnabledEl  = $("sound-enabled");
const notifyEnabledEl = $("notify-enabled");
const notifStatusEl   = $("notif-status");
const notifPermBtn    = $("notif-perm-btn");
const soundSelectEl   = $("sound-select");
const soundPreviewBtn = $("sound-preview-btn");
const exportBtn       = $("export-config-btn");
const importBtn       = $("import-config-btn");
const configTextarea  = $("config-textarea");
const configStatus    = $("config-status");

// ── State ─────────────────────────────────────────────

let poseLandmarker = null;
let running = false;
let lastFrameTime = -1;

// Camera angle: 'front' | 'front-left' | 'front-right' | 'side-left' | 'side-right'
let cameraAngle = "front";

// Current smoothed feature vector
let currentFeatures = null;
const SMOOTH = 0.3;
let smoothedFeatures = null;

// Training data: arrays of { label: 'good'|'bad', features: number[], timestamp: number }
let trainingSamples = [];
let trainedModel = null; // { goodCentroid, badCentroid } once enough samples

// Alert timing
let badPostureSince = null;
let lastAlertTime = 0;

// Selected alert sound
let selectedSound = "chime";

// Audio — must be resumed on user gesture
let audioCtx = null;

// Browser Notifications
let notifPermission = Notification?.permission ?? "denied";

// Title flashing for background tab
let titleFlashInterval = null;
const TITLE_DEFAULT = "PostureCheck";

// WebSocket
let ws = null;

// ══════════════════════════════════════════════════════════
//  Camera Angle Profiles
//
//  MediaPipe returns 33 landmarks. WorldLandmarks are 3D coordinates
//  in METERS relative to hip midpoint — these are approximately
//  camera-angle-invariant (the model internally estimates 3D pose
//  from the 2D image). However, at steep angles:
//
//  - Some landmarks are occluded (low visibility)
//  - The z-depth estimate becomes less reliable
//  - Different metrics become more/less informative
//
//  Each profile defines:
//  - Which features to compute and their weights for classification
//  - Human-readable metric labels for the UI
//  - Which landmark IDs are "posture-critical" for drawing
// ══════════════════════════════════════════════════════════

const ANGLE_PROFILES = {
  "front": {
    label: "Front-facing",
    metrics: [
      { name: "Shoulder Tilt",   fn: "shoulderTilt",  weight: 1.0 },
      { name: "Head Forward",    fn: "headForwardZ",  weight: 0.6 },
      { name: "Lateral Lean",    fn: "lateralLean",   weight: 1.0 },
      { name: "Head Drop",       fn: "headDrop",      weight: 0.8 },
    ],
    postureIds: [0, 7, 8, 11, 12],
  },
  "front-left": {
    label: "45° from left",
    metrics: [
      { name: "Shoulder Tilt",   fn: "shoulderTilt",  weight: 0.7 },
      { name: "Head Forward",    fn: "headForwardZ",  weight: 1.0 },
      { name: "Spine Angle",     fn: "spineAngle",    weight: 1.0 },
      { name: "Head Drop",       fn: "headDrop",      weight: 0.8 },
    ],
    postureIds: [0, 7, 11, 12, 23, 24],
  },
  "front-right": {
    label: "45° from right",
    metrics: [
      { name: "Shoulder Tilt",   fn: "shoulderTilt",  weight: 0.7 },
      { name: "Head Forward",    fn: "headForwardZ",  weight: 1.0 },
      { name: "Spine Angle",     fn: "spineAngle",    weight: 1.0 },
      { name: "Head Drop",       fn: "headDrop",      weight: 0.8 },
    ],
    postureIds: [0, 8, 11, 12, 23, 24],
  },
  "side-left": {
    label: "Left side view",
    metrics: [
      { name: "Head Forward",    fn: "headForwardZ",  weight: 1.0 },
      { name: "Spine Angle",     fn: "spineAngle",    weight: 1.0 },
      { name: "Head Drop",       fn: "headDrop",      weight: 0.9 },
      { name: "Neck Angle",      fn: "neckAngle",     weight: 0.8 },
    ],
    postureIds: [0, 7, 11, 23],
  },
  "side-right": {
    label: "Right side view",
    metrics: [
      { name: "Head Forward",    fn: "headForwardZ",  weight: 1.0 },
      { name: "Spine Angle",     fn: "spineAngle",    weight: 1.0 },
      { name: "Head Drop",       fn: "headDrop",      weight: 0.9 },
      { name: "Neck Angle",      fn: "neckAngle",     weight: 0.8 },
    ],
    postureIds: [0, 8, 12, 24],
  },
};

// ── Skeleton connections ──────────────────────────────

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [23, 25], [25, 27],
  [24, 26], [26, 28],
];

// ══════════════════════════════════════════════════════════
//  Feature Extraction Functions
//
//  Each function receives (norm, world) — the normalized 2D
//  landmarks and 3D world landmarks. World landmarks are in
//  meters and more reliable for angle-invariant measurements,
//  but normalized coords are used as fallback when world data
//  has low visibility.
// ══════════════════════════════════════════════════════════

const FEATURE_FNS = {
  // Degrees of shoulder tilt from horizontal
  // Uses 2D normalized coords (y-axis) — always visible from front/angle
  shoulderTilt(norm, world) {
    const ls = norm[11], rs = norm[12];
    if (ls.visibility < 0.4 || rs.visibility < 0.4) return 0;
    const dy = rs.y - ls.y;
    const dx = Math.abs(rs.x - ls.x) || 0.001;
    return Math.atan2(dy, dx) * (180 / Math.PI);
  },

  // Head forward offset in Z (depth) — uses 3D world coords
  // Positive = head is forward of shoulders (slouching)
  // This is THE key metric for side/angled views
  headForwardZ(norm, world) {
    const nose = world[0];
    const ls = world[11], rs = world[12];
    if (norm[0].visibility < 0.4 || norm[11].visibility < 0.4 || norm[12].visibility < 0.4) return 0;
    const shoulderMidZ = (ls.z + rs.z) / 2;
    return nose.z - shoulderMidZ; // negative = forward of shoulders
  },

  // Lateral lean — horizontal offset of head center from shoulder center
  // Uses 2D normalized coords. Signed: negative = leaning left, positive = right
  lateralLean(norm, world) {
    const nose = norm[0];
    const ls = norm[11], rs = norm[12];
    if (nose.visibility < 0.4 || ls.visibility < 0.4 || rs.visibility < 0.4) return 0;
    const shoulderMidX = (ls.x + rs.x) / 2;
    const shoulderWidth = Math.abs(rs.x - ls.x) || 0.001;
    // Normalize by shoulder width so it's scale-invariant
    return (nose.x - shoulderMidX) / shoulderWidth;
  },

  // Vertical distance from nose to shoulder midpoint (normalized)
  // Decreases when you slouch/drop your head
  headDrop(norm, world) {
    const nose = norm[0];
    const ls = norm[11], rs = norm[12];
    if (nose.visibility < 0.4 || ls.visibility < 0.4 || rs.visibility < 0.4) return 0;
    const shoulderMidY = (ls.y + rs.y) / 2;
    return shoulderMidY - nose.y; // positive = head above shoulders
  },

  // Spine angle from vertical — uses 3D world coords
  // Angle between (hip midpoint → shoulder midpoint) and vertical
  spineAngle(norm, world) {
    const ls = world[11], rs = world[12];
    const lh = world[23], rh = world[24];
    if (norm[11].visibility < 0.3 || norm[12].visibility < 0.3) return 0;
    // If hips not visible, estimate from shoulders only using Y position
    if (norm[23].visibility < 0.2 || norm[24].visibility < 0.2) {
      // Use head-to-shoulder angle as proxy
      const nose = world[0];
      const sMidX = (ls.x + rs.x) / 2;
      const sMidY = (ls.y + rs.y) / 2;
      const dx = nose.x - sMidX;
      const dy = nose.y - sMidY; // y is inverted in world (up = negative)
      return Math.atan2(dx, Math.abs(dy) || 0.001) * (180 / Math.PI);
    }
    const hipMidX = (lh.x + rh.x) / 2;
    const hipMidY = (lh.y + rh.y) / 2;
    const sMidX = (ls.x + rs.x) / 2;
    const sMidY = (ls.y + rs.y) / 2;
    const dx = sMidX - hipMidX;
    const dy = hipMidY - sMidY; // positive = upward
    return Math.atan2(dx, dy || 0.001) * (180 / Math.PI);
  },

  // Neck angle — angle from shoulder midpoint to nose, relative to vertical
  // Particularly useful for side views to detect forward head posture
  neckAngle(norm, world) {
    const nose = world[0];
    const ls = world[11], rs = world[12];
    if (norm[0].visibility < 0.4 || norm[11].visibility < 0.3 || norm[12].visibility < 0.3) return 0;
    const sMidX = (ls.x + rs.x) / 2;
    const sMidY = (ls.y + rs.y) / 2;
    const sMidZ = (ls.z + rs.z) / 2;
    // 3D angle: use XZ plane for forward lean, and Y for vertical
    const dx = nose.x - sMidX;
    const dz = nose.z - sMidZ;
    const dy = nose.y - sMidY;
    const horizontal = Math.sqrt(dx * dx + dz * dz);
    return Math.atan2(horizontal, Math.abs(dy) || 0.001) * (180 / Math.PI);
  },
};

// ══════════════════════════════════════════════════════════
//  Extract full feature vector for current angle profile
// ══════════════════════════════════════════════════════════

function extractFeatures(norm, world) {
  const profile = ANGLE_PROFILES[cameraAngle];
  return profile.metrics.map((m) => {
    const val = FEATURE_FNS[m.fn](norm, world);
    return val * m.weight;
  });
}

function extractRawMetricValues(norm, world) {
  const profile = ANGLE_PROFILES[cameraAngle];
  return profile.metrics.map((m) => FEATURE_FNS[m.fn](norm, world));
}

// ══════════════════════════════════════════════════════════
//  Initialization
// ══════════════════════════════════════════════════════════

async function init() {
  setupEventListeners();
  connectWebSocket();
  loadSavedState();
  updateCameraAngleUI();
  updateMetricLabels();

  // Request browser notification permission early
  await requestNotificationPermission();

  console.log("[PostureCheck] Loading MediaPipe WASM...");
  const vision = await FilesetResolver.forVisionTasks("/wasm");

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/models/pose_landmarker_lite.task" },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  console.log("[PostureCheck] PoseLandmarker ready");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
  });

  video.srcObject = stream;
  await video.play();
  await waitForVideoReady(video);

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  console.log(`[PostureCheck] Video: ${video.videoWidth}x${video.videoHeight}`);

  running = true;
  enableTrainingButtons();
  loadingOverlay.classList.add("hidden");
  updateStatus();
  detect();
}

function waitForVideoReady(v) {
  return new Promise((resolve) => {
    const check = () => {
      if (v.videoWidth > 0 && v.videoHeight > 0 && v.readyState >= 2) resolve();
      else requestAnimationFrame(check);
    };
    check();
  });
}

// ══════════════════════════════════════════════════════════
//  Detection Loop
// ══════════════════════════════════════════════════════════

function detect() {
  if (!running) return;

  const now = performance.now();
  if (now === lastFrameTime) { requestAnimationFrame(detect); return; }
  lastFrameTime = now;

  let result;
  try {
    result = poseLandmarker.detectForVideo(video, now);
  } catch (err) {
    console.error("[PostureCheck] detect error:", err);
    requestAnimationFrame(detect);
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (result.landmarks?.length > 0 && result.worldLandmarks?.length > 0) {
    const norm  = result.landmarks[0];
    const world = result.worldLandmarks[0];

    // Extract features
    const features = extractFeatures(norm, world);
    const rawValues = extractRawMetricValues(norm, world);

    // Smooth
    if (!smoothedFeatures) {
      smoothedFeatures = [...features];
    } else {
      for (let i = 0; i < features.length; i++) {
        smoothedFeatures[i] = smoothedFeatures[i] * (1 - SMOOTH) + features[i] * SMOOTH;
      }
    }
    currentFeatures = smoothedFeatures;

    // Classify
    const score = classifyPosture(smoothedFeatures);
    const threshold = parseFloat(threshScoreEl.value);
    const isGood = score === null ? null : score >= threshold;

    // Draw
    const profile = ANGLE_PROFILES[cameraAngle];
    drawSkeleton(norm, isGood, new Set(profile.postureIds));
    updateMetricsUI(rawValues, score, isGood);
    handleAlerts(isGood);
  }

  requestAnimationFrame(detect);
}

// ══════════════════════════════════════════════════════════
//  Classification — Weighted centroid distance
//
//  We compute the centroid (mean feature vector) of all "good"
//  samples and all "bad" samples. Then for each frame we measure
//  the distance to each centroid and compute a 0→1 score where
//  1 = perfectly matching good, 0 = perfectly matching bad.
// ══════════════════════════════════════════════════════════

function classifyPosture(features) {
  if (!trainedModel) return null;

  const distGood = vectorDist(features, trainedModel.goodCentroid);
  const distBad  = vectorDist(features, trainedModel.badCentroid);

  // If centroids are nearly identical, can't classify
  const totalDist = distGood + distBad;
  if (totalDist < 0.0001) return 0.5;

  // Score: 0 = bad, 1 = good
  // closer to good centroid → higher score
  return distBad / totalDist;
}

function vectorDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function retrainModel() {
  const profile = ANGLE_PROFILES[cameraAngle];
  const numFeatures = profile.metrics.length;

  // Filter samples for current camera angle
  const goodSamples = trainingSamples.filter((s) => s.label === "good" && s.angle === cameraAngle);
  const badSamples  = trainingSamples.filter((s) => s.label === "bad"  && s.angle === cameraAngle);

  if (goodSamples.length < 3 || badSamples.length < 3) {
    trainedModel = null;
    return;
  }

  // Compute centroids
  const goodCentroid = new Array(numFeatures).fill(0);
  const badCentroid  = new Array(numFeatures).fill(0);

  for (const s of goodSamples) {
    for (let i = 0; i < numFeatures; i++) goodCentroid[i] += s.features[i];
  }
  for (let i = 0; i < numFeatures; i++) goodCentroid[i] /= goodSamples.length;

  for (const s of badSamples) {
    for (let i = 0; i < numFeatures; i++) badCentroid[i] += s.features[i];
  }
  for (let i = 0; i < numFeatures; i++) badCentroid[i] /= badSamples.length;

  trainedModel = { goodCentroid, badCentroid };
  console.log("[PostureCheck] Model retrained", {
    good: goodSamples.length, bad: badSamples.length,
    goodCentroid: goodCentroid.map((v) => v.toFixed(3)),
    badCentroid: badCentroid.map((v) => v.toFixed(3)),
  });
}

// ══════════════════════════════════════════════════════════
//  Training — Capture Samples
// ══════════════════════════════════════════════════════════

function captureSample(label) {
  if (!currentFeatures) return;

  const sample = {
    label,
    angle: cameraAngle,
    features: [...currentFeatures],
    timestamp: Date.now(),
  };

  trainingSamples.push(sample);
  retrainModel();
  updateTrainingUI();
  saveState();

  // Flash camera border
  const el = document.querySelector(".camera-container");
  el.classList.remove("flash-good", "flash-bad");
  void el.offsetWidth; // reflow
  el.classList.add(label === "good" ? "flash-good" : "flash-bad");
  setTimeout(() => el.classList.remove("flash-good", "flash-bad"), 500);
}

function undoLastSample() {
  if (trainingSamples.length === 0) return;
  trainingSamples.pop();
  retrainModel();
  updateTrainingUI();
  saveState();
}

function clearAllSamples() {
  if (!confirm("Clear all training samples? This cannot be undone.")) return;
  trainingSamples = [];
  trainedModel = null;
  updateTrainingUI();
  saveState();
}

// ══════════════════════════════════════════════════════════
//  Skeleton Drawing
// ══════════════════════════════════════════════════════════

function drawSkeleton(landmarks, isGood, postureIds) {
  const w = canvas.width;
  const h = canvas.height;

  const goodClr = "#00e676";
  const badClr  = "#ff1744";
  const neutralClr = "#7c4dff";
  const dimClr  = "rgba(255, 255, 255, 0.25)";

  const activeClr = isGood === null ? neutralClr : isGood ? goodClr : badClr;

  for (const [i, j] of CONNECTIONS) {
    const a = landmarks[i], b = landmarks[j];
    if (a.visibility < 0.35 || b.visibility < 0.35) continue;

    const isPosture = postureIds.has(i) && postureIds.has(j);
    ctx.strokeStyle = isPosture ? activeClr : dimClr;
    ctx.lineWidth = isPosture ? 4 : 2;

    if (isPosture) {
      ctx.shadowColor = activeClr;
      ctx.shadowBlur = 10;
    }

    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    if (pt.visibility < 0.35) continue;

    const isPosture = postureIds.has(i);
    ctx.fillStyle = isPosture ? activeClr : dimClr;

    if (isPosture) {
      ctx.shadowColor = activeClr;
      ctx.shadowBlur = 12;
    }

    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, isPosture ? 6 : 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

// ══════════════════════════════════════════════════════════
//  UI Updates
// ══════════════════════════════════════════════════════════

function updateStatus() {
  const goodCount = trainingSamples.filter((s) => s.label === "good" && s.angle === cameraAngle).length;
  const badCount  = trainingSamples.filter((s) => s.label === "bad"  && s.angle === cameraAngle).length;

  if (!trainedModel) {
    statusEl.textContent = `Untrained — Need ${Math.max(0, 3 - goodCount)} good, ${Math.max(0, 3 - badCount)} bad`;
    statusEl.className = "status untrained";
  }
}

function updateMetricsUI(rawValues, score, isGood) {
  const profile = ANGLE_PROFILES[cameraAngle];

  // Update status badge
  if (!trainedModel) {
    updateStatus();
  } else if (isGood) {
    statusEl.textContent = "Good Posture ✓";
    statusEl.className = "status good";
  } else {
    statusEl.textContent = "Bad Posture ✗";
    statusEl.className = "status bad";
  }

  // Update metric bars
  for (let i = 0; i < 4; i++) {
    const val = rawValues[i] ?? 0;
    const valEl = $(`val-m${i}`);
    const barEl = $(`bar-m${i}`);

    // Format based on metric type
    const name = profile.metrics[i]?.fn ?? "";
    let displayVal, barPct;

    if (name.includes("Angle") || name === "shoulderTilt") {
      displayVal = `${val.toFixed(1)}°`;
      barPct = Math.min(Math.abs(val) / 30 * 100, 100);
    } else if (name === "headForwardZ") {
      displayVal = `${(val * 100).toFixed(1)}cm`;
      barPct = Math.min(Math.abs(val) / 0.15 * 100, 100);
    } else if (name === "lateralLean") {
      displayVal = val.toFixed(3);
      barPct = Math.min(Math.abs(val) / 0.5 * 100, 100);
    } else {
      displayVal = val.toFixed(3);
      barPct = Math.min(Math.abs(val) / 0.3 * 100, 100);
    }

    valEl.textContent = displayVal;
    barEl.style.width = `${barPct}%`;

    if (!trainedModel) {
      barEl.className = "bar-fill good";
    } else if (isGood) {
      barEl.className = barPct < 50 ? "bar-fill good" : "bar-fill warning";
    } else {
      barEl.className = barPct < 40 ? "bar-fill warning" : "bar-fill bad";
    }
  }

  // Update score
  if (score !== null) {
    const pct = (score * 100).toFixed(0);
    scoreValueEl.textContent = `${pct}%`;
    scoreValueEl.className = `score-value ${isGood ? "good" : "bad"}`;
  } else {
    scoreValueEl.textContent = "—";
    scoreValueEl.className = "score-value";
  }
}

function updateMetricLabels() {
  const profile = ANGLE_PROFILES[cameraAngle];
  for (let i = 0; i < 4; i++) {
    const labelEl = $(`metric-label-${i}`);
    labelEl.textContent = profile.metrics[i]?.name ?? "—";
  }
}

function updateTrainingUI() {
  const goodCount = trainingSamples.filter((s) => s.label === "good" && s.angle === cameraAngle).length;
  const badCount  = trainingSamples.filter((s) => s.label === "bad"  && s.angle === cameraAngle).length;
  const totalForAngle = goodCount + badCount;

  goodCountEl.textContent = `(${goodCount})`;
  badCountEl.textContent  = `(${badCount})`;

  trainClearBtn.disabled = trainingSamples.length === 0;
  trainUndoBtn.disabled  = trainingSamples.length === 0;

  // Status text
  if (trainedModel) {
    trainingStatusEl.textContent = `✅ Trained with ${goodCount} good + ${badCount} bad samples for ${ANGLE_PROFILES[cameraAngle].label}`;
    trainingStatusEl.className = "training-status trained";
  } else if (totalForAngle > 0) {
    const needGood = Math.max(0, 3 - goodCount);
    const needBad  = Math.max(0, 3 - badCount);
    trainingStatusEl.textContent = `Need ${needGood > 0 ? `${needGood} more good` : ""}${needGood > 0 && needBad > 0 ? " + " : ""}${needBad > 0 ? `${needBad} more bad` : ""} samples`;
    trainingStatusEl.className = "training-status";
  } else {
    trainingStatusEl.textContent = "Capture at least 3 good + 3 bad pose samples to enable detection.";
    trainingStatusEl.className = "training-status";
  }

  // Sample dots
  sampleListEl.innerHTML = "";
  const samplesForAngle = trainingSamples.filter((s) => s.angle === cameraAngle);
  for (const s of samplesForAngle) {
    const dot = document.createElement("div");
    dot.className = `sample-dot ${s.label}`;
    dot.title = `${s.label} — ${new Date(s.timestamp).toLocaleTimeString()}`;
    sampleListEl.appendChild(dot);
  }

  updateStatus();
}

function updateCameraAngleUI() {
  document.querySelectorAll(".cam-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.angle === cameraAngle);
  });
  updateMetricLabels();
  // Reset smoothed features when switching angle since feature dimensions change
  smoothedFeatures = null;
  // Retrain for the new angle
  retrainModel();
  updateTrainingUI();
}

function enableTrainingButtons() {
  trainGoodBtn.disabled = false;
  trainBadBtn.disabled  = false;
}

// ══════════════════════════════════════════════════════════
//  Alert System
//
//  Multiple alert channels to ensure you're notified even
//  when the tab is in the background:
//
//  1. Visual overlay — red border + text on the camera view
//  2. Web Audio — three-tone chime (needs AudioContext resumed)
//  3. Browser Notification API — works in background tabs on
//     all platforms (Windows/Mac/Linux)
//  4. Server-side system toast — via WebSocket → native
//     notification command (PowerShell / osascript / notify-send)
//  5. Title bar flash — alternates document.title so the tab
//     visibly blinks in the taskbar
// ══════════════════════════════════════════════════════════

function handleAlerts(isGood) {
  if (isGood === null) return; // no model

  const now = Date.now();
  const delay    = parseInt(alertDelayEl.value) * 1000;
  const cooldown = parseInt(alertCooldownEl.value) * 1000;

  if (!isGood) {
    if (!badPostureSince) {
      badPostureSince = now;
    } else if (now - badPostureSince >= delay && now - lastAlertTime >= cooldown) {
      fireAlert();
      lastAlertTime = now;
    }
  } else {
    badPostureSince = null;
    hideAlert();
  }
}

function fireAlert() {
  // 1 — Visual overlay
  alertOverlay.classList.remove("hidden");

  // 2 — Sound
  if (soundEnabledEl.checked) playAlertSound();

  // 3 — Notification: prefer browser notification, fall back to server toast
  //     (only one, not both — avoids duplicate alerts)
  if (notifyEnabledEl.checked) {
    if (notifPermission === "granted") {
      showBrowserNotification();
    } else if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "bad-posture", message: "Fix your posture! 🧍" }));
    }
  }

  // 4 — Title flash
  startTitleFlash();

  // Auto-dismiss visual overlay after 5s
  setTimeout(hideAlert, 5000);
}

function hideAlert() {
  alertOverlay.classList.add("hidden");
  stopTitleFlash();
}

// ── Sound via Web Audio API ────────────────────────────
//
// AudioContext must be resumed after a user gesture (browser
// autoplay policy). We resume it eagerly on first click/key
// via ensureAudioContext(), so by the time an alert fires
// the context is already running.

function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

// ══════════════════════════════════════════════════════════
//  Sound Bites — 10 selectable alert sounds
//  All synthesized with Web Audio API (no external files)
// ══════════════════════════════════════════════════════════

const SOUND_BITES = {
  chime: {
    label: "🔔 Classic Chime",
    play(ctx) {
      const tone = (freq, start, dur) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.frequency.value = freq; osc.type = "sine";
        gain.gain.setValueAtTime(0.3, ctx.currentTime + start);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
        osc.start(ctx.currentTime + start);
        osc.stop(ctx.currentTime + start + dur);
      };
      tone(523, 0, 0.15); tone(659, 0.18, 0.15); tone(784, 0.36, 0.25);
    },
  },
  airhorn: {
    label: "📯 Air Horn",
    play(ctx) {
      // Layered sawtooth waves for that MLG air horn feel
      [480, 600, 720].forEach((f) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.type = "sawtooth"; osc.frequency.value = f;
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.8);
      });
    },
  },
  bruh: {
    label: '😐 "Bruh" Drop',
    play(ctx) {
      // Low descending tone with noise burst
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(220, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(80, ctx.currentTime + 0.5);
      gain.gain.setValueAtTime(0.35, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.5);
    },
  },
  vine_boom: {
    label: "💥 Vine Boom",
    play(ctx) {
      // Deep thud + resonance
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.setValueAtTime(120, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(30, ctx.currentTime + 0.4);
      gain.gain.setValueAtTime(0.5, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.5);
      // Impact transient
      const noise = ctx.createBufferSource();
      const buf = ctx.createBuffer(1, ctx.sampleRate * 0.05, ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
      noise.buffer = buf;
      const ng = ctx.createGain();
      noise.connect(ng); ng.connect(ctx.destination);
      ng.gain.setValueAtTime(0.4, ctx.currentTime);
      ng.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
      noise.start(ctx.currentTime); noise.stop(ctx.currentTime + 0.05);
    },
  },
  sad_trombone: {
    label: "🎺 Sad Trombone",
    play(ctx) {
      // Wah wah wah waaaaah
      const notes = [392, 370, 349, 233];
      const durs = [0.25, 0.25, 0.25, 0.6];
      let t = 0;
      notes.forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.type = "triangle"; osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, ctx.currentTime + t);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + t + durs[i]);
        osc.start(ctx.currentTime + t);
        osc.stop(ctx.currentTime + t + durs[i]);
        t += durs[i] - 0.02;
      });
    },
  },
  oof: {
    label: '💀 "OOF"',
    play(ctx) {
      // Roblox-style OOF: quick descending tone
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = "square";
      osc.frequency.setValueAtTime(480, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(160, ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.2);
      // Second harmonic
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.connect(gain2); gain2.connect(ctx.destination);
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(360, ctx.currentTime);
      osc2.frequency.exponentialRampToValueAtTime(120, ctx.currentTime + 0.18);
      gain2.gain.setValueAtTime(0.15, ctx.currentTime);
      gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);
      osc2.start(ctx.currentTime); osc2.stop(ctx.currentTime + 0.22);
    },
  },
  metal_gear: {
    label: "❗ Alert (MGS)",
    play(ctx) {
      // Metal Gear Solid alert: sharp high tone
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = "square";
      osc.frequency.setValueAtTime(1200, ctx.currentTime);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.setValueAtTime(0.2, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0, ctx.currentTime + 0.13);
      gain.gain.setValueAtTime(0.25, ctx.currentTime + 0.2);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.7);
      osc.frequency.setValueAtTime(1200, ctx.currentTime + 0.2);
      osc.frequency.exponentialRampToValueAtTime(900, ctx.currentTime + 0.7);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.7);
    },
  },
  navi_hey: {
    label: '🧚 "Hey! Listen!"',
    play(ctx) {
      // Navi from Zelda: quick high pings
      [0, 0.15, 0.25].forEach((t) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.type = "sine";
        osc.frequency.setValueAtTime(1400 + t * 600, ctx.currentTime + t);
        gain.gain.setValueAtTime(0.25, ctx.currentTime + t);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + t + 0.1);
        osc.start(ctx.currentTime + t);
        osc.stop(ctx.currentTime + t + 0.1);
      });
    },
  },
  fbi_open_up: {
    label: "🚔 Siren Wail",
    play(ctx) {
      // Police siren wail up and down
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = "sawtooth";
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.0);
      osc.frequency.setValueAtTime(600, ctx.currentTime);
      osc.frequency.linearRampToValueAtTime(1200, ctx.currentTime + 0.25);
      osc.frequency.linearRampToValueAtTime(600, ctx.currentTime + 0.5);
      osc.frequency.linearRampToValueAtTime(1200, ctx.currentTime + 0.75);
      osc.frequency.linearRampToValueAtTime(600, ctx.currentTime + 1.0);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 1.0);
    },
  },
  dramatic: {
    label: "🎻 Dramatic Sting",
    play(ctx) {
      // Dramatic orchestra hit: layered low + high
      [130, 165, 196, 392, 784].forEach((f, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.type = i < 3 ? "sawtooth" : "sine";
        osc.frequency.value = f;
        gain.gain.setValueAtTime(i < 3 ? 0.1 : 0.06, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.2);
        osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 1.2);
      });
      // Impact
      const noise = ctx.createBufferSource();
      const buf = ctx.createBuffer(1, ctx.sampleRate * 0.08, ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
      noise.buffer = buf;
      const ng = ctx.createGain();
      noise.connect(ng); ng.connect(ctx.destination);
      ng.gain.setValueAtTime(0.3, ctx.currentTime);
      ng.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
      noise.start(ctx.currentTime); noise.stop(ctx.currentTime + 0.08);
    },
  },
};

function playAlertSound() {
  try {
    const ctx = ensureAudioContext();
    if (ctx.state === "suspended") {
      console.warn("[PostureCheck] AudioContext suspended — sound skipped (interact with page first)");
      return;
    }
    const sound = SOUND_BITES[selectedSound] ?? SOUND_BITES.chime;
    sound.play(ctx);
  } catch (err) {
    console.warn("[PostureCheck] Sound playback error:", err);
  }
}

// ── Browser Notification API ────────────────────────────
//
// Works when the tab is in the background on Windows, macOS,
// and Linux. Requires one-time permission grant.

async function requestNotificationPermission() {
  if (!("Notification" in window)) {
    console.warn("[PostureCheck] Browser Notifications not supported");
    updateNotifUI("unsupported");
    return;
  }
  if (Notification.permission === "granted") {
    notifPermission = "granted";
    updateNotifUI("granted");
    return;
  }
  if (Notification.permission === "denied") {
    notifPermission = "denied";
    updateNotifUI("denied");
    return;
  }
  // "default" — we can ask
  updateNotifUI("prompt");
  const perm = await Notification.requestPermission();
  notifPermission = perm;
  updateNotifUI(perm);
}

function updateNotifUI(state) {
  if (!notifStatusEl) return;
  if (state === "granted") {
    notifStatusEl.textContent = "✓ Browser notifications enabled";
    notifStatusEl.className = "notif-status granted";
    notifPermBtn?.classList.add("hidden");
  } else if (state === "denied") {
    notifStatusEl.textContent = "✗ Notifications blocked — enable in browser settings";
    notifStatusEl.className = "notif-status denied";
    notifPermBtn?.classList.add("hidden");
  } else if (state === "prompt") {
    notifStatusEl.textContent = "Browser notifications not yet permitted";
    notifStatusEl.className = "notif-status prompt";
    notifPermBtn?.classList.remove("hidden");
  } else {
    notifStatusEl.textContent = "Browser notifications not supported";
    notifStatusEl.className = "notif-status denied";
    notifPermBtn?.classList.add("hidden");
  }
}

function showBrowserNotification() {
  if (notifPermission !== "granted") return;
  try {
    const n = new Notification("PostureCheck ⚠️", {
      body: "You've been slouching — sit up straight!",
      icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='0.9em' font-size='90'>🧍</text></svg>",
      tag: "posture-alert",       // replaces previous, won't stack
      requireInteraction: false,
      silent: true,               // we handle our own sound
    });
    // Auto-close after 6s (some OS's don't auto-dismiss)
    setTimeout(() => n.close(), 6000);
  } catch (err) {
    console.warn("[PostureCheck] Browser notification error:", err);
  }
}

// ── Title Bar Flash ────────────────────────────────────
//
// Alternates the document title so the tab visually blinks
// in the taskbar / tab bar — works everywhere.

function startTitleFlash() {
  if (titleFlashInterval) return;
  let on = true;
  titleFlashInterval = setInterval(() => {
    document.title = on ? "⚠️ Fix Your Posture!" : TITLE_DEFAULT;
    on = !on;
  }, 800);
}

function stopTitleFlash() {
  if (titleFlashInterval) {
    clearInterval(titleFlashInterval);
    titleFlashInterval = null;
    document.title = TITLE_DEFAULT;
  }
}

// ══════════════════════════════════════════════════════════
//  WebSocket
// ══════════════════════════════════════════════════════════

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onclose = () => setTimeout(connectWebSocket, 3000);
  ws.onerror = () => {};
}

// ══════════════════════════════════════════════════════════
//  Persistence
// ══════════════════════════════════════════════════════════

function saveState() {
  localStorage.setItem("posture-v2", JSON.stringify({
    cameraAngle,
    trainingSamples,
    selectedSound,
    threshScore: threshScoreEl.value,
    alertDelay: alertDelayEl.value,
    alertCooldown: alertCooldownEl.value,
    sound: soundEnabledEl.checked,
    notify: notifyEnabledEl.checked,
  }));
}

function loadSavedState() {
  try {
    const raw = localStorage.getItem("posture-v2");
    if (!raw) return;
    applyConfig(JSON.parse(raw));
  } catch {
    // Ignore corrupt state
  }
}

function applyConfig(s) {
  if (s.cameraAngle) cameraAngle = s.cameraAngle;
  if (s.trainingSamples) trainingSamples = s.trainingSamples;
  if (s.selectedSound && SOUND_BITES[s.selectedSound]) {
    selectedSound = s.selectedSound;
    if (soundSelectEl) soundSelectEl.value = selectedSound;
  }
  if (s.threshScore) threshScoreEl.value = s.threshScore;
  if (s.alertDelay) alertDelayEl.value = s.alertDelay;
  if (s.alertCooldown) alertCooldownEl.value = s.alertCooldown;
  if (s.sound !== undefined) soundEnabledEl.checked = s.sound;
  if (s.notify !== undefined) notifyEnabledEl.checked = s.notify;

  // Sync labels
  $("thresh-val").textContent = threshScoreEl.value;
  $("delay-val").textContent = `${alertDelayEl.value}s`;
  $("cooldown-val").textContent = `${alertCooldownEl.value}s`;

  retrainModel();
  updateTrainingUI();
  updateCameraAngleUI();
}

function exportConfig() {
  const config = {
    _type: "PostureCheck Config",
    _version: 2,
    _exported: new Date().toISOString(),
    cameraAngle,
    trainingSamples,
    selectedSound,
    threshScore: threshScoreEl.value,
    alertDelay: alertDelayEl.value,
    alertCooldown: alertCooldownEl.value,
    sound: soundEnabledEl.checked,
    notify: notifyEnabledEl.checked,
  };
  const json = JSON.stringify(config, null, 2);
  if (configTextarea) {
    configTextarea.value = json;
    configTextarea.classList.remove("hidden");
  }
  navigator.clipboard.writeText(json).then(() => {
    showConfigStatus("✓ Config copied to clipboard!", "good");
  }).catch(() => {
    showConfigStatus("Config exported to text box below — copy it manually", "neutral");
  });
}

function importConfig() {
  const text = configTextarea?.value?.trim();
  if (!text) {
    showConfigStatus("Paste a config JSON into the text box first", "bad");
    configTextarea?.classList.remove("hidden");
    configTextarea?.focus();
    return;
  }
  try {
    const config = JSON.parse(text);
    if (config._type !== "PostureCheck Config") {
      showConfigStatus("✗ Invalid config format", "bad");
      return;
    }
    applyConfig(config);
    saveState();
    configTextarea.value = "";
    configTextarea.classList.add("hidden");
    showConfigStatus(`✓ Imported! ${config.trainingSamples?.length ?? 0} samples loaded`, "good");
  } catch (err) {
    showConfigStatus("✗ Invalid JSON — check the pasted text", "bad");
  }
}

function showConfigStatus(msg, type) {
  if (!configStatus) return;
  configStatus.textContent = msg;
  configStatus.className = `config-status ${type}`;
  setTimeout(() => { configStatus.textContent = ""; configStatus.className = "config-status"; }, 4000);
}

// ══════════════════════════════════════════════════════════
//  Event Listeners
// ══════════════════════════════════════════════════════════

function setupEventListeners() {
  // Training
  trainGoodBtn.addEventListener("click", () => captureSample("good"));
  trainBadBtn.addEventListener("click", () => captureSample("bad"));
  trainClearBtn.addEventListener("click", clearAllSamples);
  trainUndoBtn.addEventListener("click", undoLastSample);

  // Camera angle
  document.querySelectorAll(".cam-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      cameraAngle = btn.dataset.angle;
      updateCameraAngleUI();
      saveState();
    });
  });

  // Sliders
  threshScoreEl.addEventListener("input", (e) => {
    $("thresh-val").textContent = e.target.value;
  });
  alertDelayEl.addEventListener("input", (e) => {
    $("delay-val").textContent = `${e.target.value}s`;
  });
  alertCooldownEl.addEventListener("input", (e) => {
    $("cooldown-val").textContent = `${e.target.value}s`;
  });

  // Persist on change
  [threshScoreEl, alertDelayEl, alertCooldownEl].forEach((el) =>
    el.addEventListener("change", saveState)
  );
  soundEnabledEl.addEventListener("change", saveState);
  notifyEnabledEl.addEventListener("change", saveState);

  // Sound selector
  soundSelectEl?.addEventListener("change", (e) => {
    selectedSound = e.target.value;
    saveState();
  });
  soundPreviewBtn?.addEventListener("click", () => {
    playAlertSound();
  });

  // Config export/import
  exportBtn?.addEventListener("click", exportConfig);
  importBtn?.addEventListener("click", importConfig);

  // Notification permission button
  notifPermBtn?.addEventListener("click", async () => {
    await requestNotificationPermission();
  });

  // Keyboard shortcuts for quick capture
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "g" || e.key === "G") captureSample("good");
    if (e.key === "b" || e.key === "B") captureSample("bad");
    if (e.key === "z" && e.ctrlKey) undoLastSample();
  });

  // Resume AudioContext on first user interaction (browser autoplay policy)
  const resumeAudio = () => {
    ensureAudioContext();
    // Also re-request notification permission if not yet granted
    if (notifPermission !== "granted") requestNotificationPermission();
    document.removeEventListener("click", resumeAudio);
    document.removeEventListener("keydown", resumeAudio);
  };
  document.addEventListener("click", resumeAudio);
  document.addEventListener("keydown", resumeAudio);
}

// ══════════════════════════════════════════════════════════
//  Start
// ══════════════════════════════════════════════════════════

init().catch((err) => {
  console.error("PostureCheck init failed:", err);
  statusEl.textContent = "Error — see console";
  statusEl.className = "status bad";
  loadingOverlay.classList.add("hidden");
});
