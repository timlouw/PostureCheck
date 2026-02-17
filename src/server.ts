import { resolve, extname } from "path";

const PORT = 3000;
const ROOT = resolve(import.meta.dir, "..");
const PUBLIC = resolve(import.meta.dir, "public");
const MODELS = resolve(ROOT, "models");
const VISION = resolve(ROOT, "node_modules", "@mediapipe", "tasks-vision");

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".task": "application/octet-stream",
  ".data": "application/octet-stream",
};

function mime(path: string): string {
  return MIME[extname(path)] || "application/octet-stream";
}

async function serveFile(path: string): Promise<Response> {
  const file = Bun.file(path);
  if (await file.exists()) {
    return new Response(file, {
      headers: {
        "Content-Type": mime(path),
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
      },
    });
  }
  console.log(`  404: ${path}`);
  return new Response("Not Found", { status: 404 });
}

const server = Bun.serve({
  port: PORT,

  async fetch(req, server) {
    const url = new URL(req.url);
    const path = url.pathname;

    // WebSocket upgrade
    if (path === "/ws") {
      if (server.upgrade(req)) return;
      return new Response("WebSocket upgrade failed", { status: 400 });
    }

    // MediaPipe WASM runtime files
    if (path.startsWith("/wasm/")) {
      return serveFile(resolve(VISION, "wasm", path.slice(6)));
    }

    // MediaPipe JS bundle
    if (path === "/vision_bundle.mjs") {
      return serveFile(resolve(VISION, "vision_bundle.mjs"));
    }

    // Pose model files
    if (path.startsWith("/models/")) {
      return serveFile(resolve(MODELS, path.slice(8)));
    }

    // Static public files
    const file = path === "/" ? "index.html" : path.slice(1);
    return serveFile(resolve(PUBLIC, file));
  },

  websocket: {
    open(ws) {
      console.log("🔌 Client connected");
    },
    message(ws, msg) {
      try {
        const data = JSON.parse(String(msg));
        if (data.type === "bad-posture") {
          systemNotify(data.message || "Fix your posture!");
        }
      } catch {
        // ignore malformed messages
      }
    },
    close() {
      console.log("🔌 Client disconnected");
    },
  },
});

/** Fire a native OS notification — works on Windows, macOS, and Linux */
function systemNotify(message: string) {
  const platform = process.platform;

  if (platform === "win32") {
    // Windows: PowerShell toast via NotifyIcon
    const escaped = message.replace(/'/g, "''");
    Bun.spawn(
      [
        "powershell",
        "-NoProfile",
        "-Command",
        [
          "Add-Type -AssemblyName System.Windows.Forms",
          "$n = New-Object System.Windows.Forms.NotifyIcon",
          "$n.Icon = [System.Drawing.SystemIcons]::Warning",
          "$n.Visible = $true",
          `$n.BalloonTipTitle = 'PostureCheck'`,
          `$n.BalloonTipText = '${escaped}'`,
          "$n.BalloonTipIcon = 'Warning'",
          "$n.ShowBalloonTip(5000)",
          "Start-Sleep -Seconds 6",
          "$n.Dispose()",
        ].join("; "),
      ],
      { stdout: "ignore", stderr: "ignore" },
    );
  } else if (platform === "darwin") {
    // macOS: osascript notification
    const escaped = message.replace(/"/g, '\\"');
    Bun.spawn(
      [
        "osascript",
        "-e",
        `display notification "${escaped}" with title "PostureCheck" subtitle "⚠️ Posture Alert" sound name "Ping"`,
      ],
      { stdout: "ignore", stderr: "ignore" },
    );
  } else {
    // Linux: notify-send (libnotify)
    Bun.spawn(
      ["notify-send", "--urgency=critical", "--app-name=PostureCheck", "PostureCheck ⚠️", message],
      { stdout: "ignore", stderr: "ignore" },
    );
  }
}

console.log(`
🧍 PostureCheck running at http://localhost:${PORT}
   Open in your browser and keep the tab open while you work.
`);
