# 🧍 PostureCheck

Real-time posture monitoring using your webcam. Uses **MediaPipe Pose Landmarker** to detect body landmarks and alerts you when you've been slouching for too long.

![TypeScript](https://img.shields.io/badge/TypeScript-Bun-blue) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## How It Works

1. **Webcam feed** is processed in-browser using MediaPipe's WASM-based Pose Landmarker
2. **3D WorldLandmarks** (camera-angle-invariant) are extracted for posture metrics
3. You **train** the system by capturing good and bad posture samples
4. A **centroid classifier** scores each frame and alerts you when posture degrades

## Features

- **Camera angle profiles** — Front, 45°, and side views with tuned metric weights
- **Training system** — Capture good/bad samples, ≥3 each to enable detection
- **5 alert channels** that work even when the tab is in the background:
  - Visual overlay on camera feed
  - Web Audio chime
  - Browser Notification API (cross-platform)
  - Server-side system toast (Windows/macOS/Linux)
  - Title bar flashing
- **Full persistence** — Settings, samples, and angle saved to localStorage
- **Keyboard shortcuts** — `G` = good, `B` = bad, `Ctrl+Z` = undo

## Quick Start

```bash
# Install dependencies
bun install

# Download the pose model (~4MB)
bun run setup

# Start the dev server
bun dev
```

Open **http://localhost:3000** in your browser.

## Requirements

- [Bun](https://bun.sh) runtime
- A webcam
- A modern browser (Chrome, Edge, Firefox)

## Stack

- **Runtime**: Bun (TypeScript)
- **ML**: `@mediapipe/tasks-vision` — Pose Landmarker Lite
- **Frontend**: Vanilla HTML/CSS/JS with ES modules
- **Notifications**: Web Notification API + platform-native toasts

## Usage

1. Select your **camera position** (where the camera is relative to you)
2. Sit in **good posture** → press `G` or click "Good Posture" (repeat 3+ times)
3. Sit in **bad posture** → press `B` or click "Bad Posture" (repeat 3+ times)
4. Detection activates automatically — adjust **Sensitivity** slider as needed
5. Alerts fire after sustained bad posture (configurable delay + cooldown)

## License

MIT
