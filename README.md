# Real-Time Camera Optical Flow & Motion Visualizer

A high-performance browser application that captures a live webcam feed and computes dense optical flow (per-pixel motion vectors) between consecutive frames directly on the GPU using **WebGPU compute shaders**.

🌐 **Live Demo:** [https://emrecandir9.github.io/opt_flow/](https://emrecandir9.github.io/opt_flow/)

---

## 4 Interactive Visualizer Modes

1. **✨ Magical Particles (Default)** — 3,500 glowing embers pushed and swirled by motion force fields with real-time speed streaks.
2. **🌊 Fluid Streamlines** — Aerodynamic wind-tunnel flow ribbons tracing continuous motion paths.
3. **🔥 Thermal Speed Glow** — Bilinear-smoothed speed heatmap (Cyan $\rightarrow$ Magenta $\rightarrow$ Gold) for intuitive velocity visualization without color wheel ambiguity.
4. **🎯 Quiver Arrows + Compass HUD** — Directional arrow quiver plot paired with a live circular motion compass HUD displaying dominant movement angle and speed (px/frame).

---

## Key Features

- **WebGPU Compute Pipeline:** GPU-accelerated block-matching optical flow running at 480×270 working resolution.
- **Motion Threshold Slider (0.2 px – 10.0 px):** Cleanly suppresses camera sensor noise, micro-jitter, and lighting variations so only intentional gestures are shown.
- **3-Layer Compositing Stack:** High-performance overlay of video feed, WebGPU transparent heatmap shader, and 2D canvas particle/vector engine.
- **Instant Controls:** Real-time mode switching, intensity adjustment, and an animated test motion target generator.
- **Zero Build Step:** Built entirely in Vanilla JavaScript (ES modules), HTML, CSS, and WGSL shaders.

---

## Browser Requirements

- **Chrome 113+** or **Edge 113+** (WebGPU support required)
- **HTTPS or localhost** (required by browsers for camera access)
- A webcam / front-facing camera

---

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/emrecandir9/opt_flow.git
   cd opt_flow
   ```

2. Start a local HTTP server:
   ```bash
   python3 -m http.server 8080
   ```

3. Open `http://localhost:8080` in Chrome or Edge and click **"Start Camera"**.

---

## Project Structure

```text
├── index.html                     — Main UI & 3-layer canvas container
├── src/
│   ├── main.js                   — App orchestration, loop, mode switcher
│   ├── webgpu-context.js         — WebGPU device/adapter initialization
│   ├── capture.js                — Camera setup & GPU texture ingestion
│   ├── pipeline/
│   │   ├── grayscale.js          — GPU grayscale + downsample pass
│   │   ├── pyramid.js            — Gaussian image pyramid builder
│   │   ├── optical-flow.js       — Block-matching flow compute orchestration
│   │   ├── visualize-heatmap.js  — WebGPU heatmap render pass
│   │   ├── visualize-particles.js — 3,500 interactive force-field particles
│   │   └── visualize-vectors.js  — Streamline ribbons & quiver arrows
│   ├── shaders/
│   │   ├── grayscale.wgsl        — BT.601 luma conversion
│   │   ├── gaussian-blur.wgsl    — Separable 5-tap Gaussian blur
│   │   ├── downsample.wgsl       — 2× box downsampler
│   │   ├── block-match.wgsl      — GPU block-matching optical flow compute
│   │   └── heatmap.wgsl          — Bilinear thermal glow & HSV fragment shader
│   └── utils/
│       └── texture-utils.js      — GPU texture creation helpers
└── README.md
```

---

## License

MIT
