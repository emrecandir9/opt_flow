# Real-Time Camera Optical Flow & Motion Vector Canvas

A browser-based application that captures a live webcam feed and computes dense optical flow (per-pixel motion vectors) between consecutive frames on the GPU using **WebGPU compute shaders**. The result is rendered as two overlays on top of the live video:

- **Velocity Heatmap** — HSV color-wheel mapping of flow direction and magnitude
- **Vector Field** — sparse-sampled arrow quiver plot showing motion direction

## Features

- Real-time GPU-accelerated optical flow via WebGPU compute shaders
- Block-matching flow algorithm (8×8 blocks, ±8px search)
- 4-level Gaussian image pyramid with separable blur
- HSV color-wheel heatmap visualization with alpha blending
- Sparse vector arrow overlay via 2D Canvas readback
- Working resolution downsampling (480×270) for performance
- FPS counter and UI controls
- No build step — vanilla JS + ES modules

## Requirements

- **Browser:** Chrome 113+ or Edge 113+ (WebGPU support required)
- **HTTPS or localhost** (required for camera access)
- A webcam

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd opt_flow
   ```

2. Start a local HTTP server:
   ```bash
   python3 -m http.server 8080
   ```

3. Open `http://localhost:8080` in Chrome or Edge.

4. Click **"Start Camera"** and grant camera permissions.

## Architecture

```
[Webcam] → video element
   │
   ▼
[1] Import frame as GPUTexture (rgba8unorm)
   │
   ▼
[2] Grayscale conversion + downsample → R32Float (480×270)
   │
   ▼
[3] Build 4-level Gaussian image pyramid
   │
   ▼
[4] Block-matching optical flow → RG32Float flow field
   │
   ▼
[5] Visualization: HSV heatmap + vector arrows
   │
   ▼
[6] Composite: video + heatmap (alpha) + arrows (overlay)
```

## Project Structure

```
├── index.html                  — App shell + UI
├── src/
│   ├── main.js                — Entry point, animation loop
│   ├── webgpu-context.js      — WebGPU device/adapter init
│   ├── capture.js             — Camera setup, frame import
│   ├── pipeline/
│   │   ├── grayscale.js       — Grayscale + downsample pass
│   │   ├── pyramid.js         — Gaussian pyramid builder
│   │   ├── optical-flow.js    — Block-matching flow
│   │   ├── visualize-heatmap.js — HSV heatmap render
│   │   └── visualize-vectors.js — Arrow overlay
│   ├── shaders/
│   │   ├── grayscale.wgsl
│   │   ├── gaussian-blur.wgsl
│   │   ├── downsample.wgsl
│   │   ├── block-match.wgsl
│   │   ├── heatmap.wgsl
│   │   └── composite.wgsl
│   └── utils/
│       └── texture-utils.js   — Texture creation helpers
├── webgpu-optical-flow-spec.md — Full technical spec
└── README.md
```

## Performance

Target: ≥24 fps at 480×270 working resolution on a mid-range laptop GPU.

## License

MIT
