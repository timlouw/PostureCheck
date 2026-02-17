import { mkdir } from "fs/promises";
import { resolve } from "path";

const MODELS_DIR = resolve(import.meta.dir, "..", "models");
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const MODEL_PATH = resolve(MODELS_DIR, "pose_landmarker_lite.task");

async function setup() {
  console.log("📦 Setting up PostureCheck...\n");

  // Create models directory
  await mkdir(MODELS_DIR, { recursive: true });

  // Download model if not already present
  const file = Bun.file(MODEL_PATH);
  if (await file.exists()) {
    console.log("✅ Model already downloaded");
  } else {
    console.log("⬇️  Downloading pose_landmarker_lite.task (~4 MB)...");
    const response = await fetch(MODEL_URL);
    if (!response.ok) {
      throw new Error(`Download failed: ${response.status} ${response.statusText}`);
    }
    await Bun.write(MODEL_PATH, response);
    console.log("✅ Model downloaded successfully");
  }

  console.log("\n🎉 Setup complete! Run `bun dev` to start the server.");
}

setup().catch((err) => {
  console.error("❌ Setup failed:", err);
  process.exit(1);
});
