# Real-Time Camera Optical Flow & Motion Vector Canvas — Full Technical Spec

## 1. Project Summary

A browser-based application that:
1. Captures a live webcam feed.
2. Computes dense optical flow (per-pixel motion vectors) between consecutive frames on the GPU using WebGPU compute shaders.
3. Renders the result as two overlays on top of (or beside) the live video: a **directional vector field** (quiver plot / arrows) and a **velocity heatmap** (color-coded speed/direction map).

Target: runs in real time (≥ 24–30 fps) at a reduced working resolution (e.g. 320×240 or 480×270) even though the display/video is full resolution.

---

## 2. Tech Stack

- **Language:** Vanilla JavaScript (ES modules), no framework.
- **GPU API:** WebGPU (`navigator.gpu`), using **compute shaders** for the flow math and a **render pipeline** (or fragment shader pass) for visualization.
- **Shading language:** WGSL.
- **Video capture:** `navigator.mediaDevices.getUserMedia` → `<video>` element → `GPUExternalTexture` (via `device.importExternalTexture`) or copied into a regular `GPUTexture` each frame.
- **Fallback:** Optional Canvas 2D/WebGL2 fallback path if WebGPU is unavailable (flag as "nice to have," not required for v1).

---

## 3. High-Level Pipeline (per frame)

```
[Webcam] → video element
   │
   ▼
[1] Import current frame as GPUTexture (rgba8unorm)
   │
   ▼
[2] Grayscale conversion (compute or fragment pass) → R32Float texture
   │
   ▼
[3] Build image pyramid (N levels, e.g. 4) via downsampling compute pass
        (Gaussian blur + 2x2 box downsample, repeated per level)
   │
   ▼
[4] Pyramidal Lucas-Kanade optical flow:
        - Start at coarsest level with zero initial flow
        - At each level (coarse → fine):
            a. Warp previous frame's pyramid level by current flow estimate
            b. Compute per-pixel flow refinement (Lucas-Kanade window solve)
            c. Upsample flow to next finer level (×2, scale vectors ×2)
        - Output: full-resolution (or working-resolution) flow field,
          stored as RG32Float texture (vx, vy per pixel)
   │
   ▼
[5] Visualization pass(es):
        a. Velocity heatmap: fragment shader maps (vx,vy) → HSV/viridis color
        b. Vector field: sparse-sampled arrows drawn via instanced geometry
           or a 2D canvas overlay, drawn every N pixels (e.g. every 12px)
   │
   ▼
[6] Composite: draw video frame + heatmap (alpha blended) + vector overlay
    to the visible <canvas>
   │
   ▼
[7] Store current frame/pyramid as "previous" for next frame, loop
```

---

## 4. Detailed Stage-by-Stage Design

### 4.1 Camera Capture
- Request camera via `getUserMedia({ video: { width, height, frameRate } })`.
- Play into a hidden `<video>` element (`autoplay`, `muted`, `playsinline`).
- Each animation frame (`requestAnimationFrame` or `requestVideoFrameCallback` — **prefer `requestVideoFrameCallback`** for accurate per-frame timing instead of rAF, since it fires exactly when a new video frame is ready).
- Import the frame:
  - Option A (preferred, cheapest): `device.importExternalTexture({ source: video })` — zero-copy, but the resulting `GPUExternalTexture` is **only valid for the current submit** call and can't be stored across frames. Use it immediately in the grayscale-conversion pass and copy the result into a persistent `GPUTexture`.
  - Option B: draw video to an offscreen canvas, then `copyExternalImageToTexture`. More overhead but simpler texture lifetime management. Fine for v1, optimize to Option A later.

### 4.2 Working Resolution
- Do NOT run optical flow at full camera resolution (e.g. 1280×720). Downsample to a working resolution — **480×270 or 320×180** — at the very first grayscale-conversion step. This is the single biggest performance lever.
- Keep the full-resolution frame separately only for final display compositing.

### 4.3 Grayscale Conversion
- Compute or fragment shader: `luma = dot(rgb, vec3(0.299, 0.587, 0.114))`.
- Output format: `r32float` (need signed/precise values for gradient math later; unorm8 loses too much precision for derivatives).
- This pass also does the resolution downsample (sample source texture at working-resolution grid using a simple box filter or bilinear sample).

### 4.4 Image Pyramid Construction
- Number of levels: 3–4 (configurable). Level 0 = working resolution, each subsequent level halved.
- Each level built from the previous via:
  1. Separable Gaussian blur (5-tap, sigma ≈1) — two passes (horizontal, vertical) or a single 2D pass if simplicity > perf.
  2. 2× downsample (nearest or bilinear sample at half-res grid).
- Store all levels as either:
  - An array of separate `GPUTexture`s (simplest, recommended for v1), or
  - A single texture with mip levels (`mipLevelCount`), using `textureStore`/`textureLoad` per mip — more advanced, optimize later.
- Need this pyramid for **both** the current and previous frame (previous frame's pyramid was already computed last frame — cache it, don't recompute).

### 4.5 Pyramidal Lucas-Kanade Optical Flow (the core algorithm)

**Why pyramidal:** plain LK only handles small (sub-pixel to a few pixel) displacements because it linearizes brightness change. Fast hand/camera motion produces displacements of 10–50+ px. The pyramid lets you estimate large motion coarsely then refine.

**Per-level algorithm (coarse → fine), for each pixel p and window size w (e.g. 15×15 or 21×21):**

1. **Initialize:** at the coarsest level, flow estimate `d = (0,0)` everywhere. At finer levels, take the flow from the coarser level, upsample (bilinear) and multiply by 2 (since spatial scale doubled).
2. **Warp:** sample the *previous* frame's image at this level at position `(x,y) + d` (this is the predicted correspondence).
3. **Compute per-pixel gradients** `Ix, Iy` of the *current* frame level via central differences (Sobel or simple `[-1,0,1]` kernel).
4. **Compute temporal difference** `It = I_current(x,y) − I_previous_warped(x,y)`.
5. **Structure tensor over the window** (sum over window pixels, can precompute via a box-filter/separable sum for speed):
   ```
   Sxx = Σ Ix²      Sxy = Σ IxIy      Syy = Σ Iy²
   Sxt = Σ IxIt      Syt = Σ IyIt
   ```
6. **Solve 2×2 linear system** for the flow increment `(du, dv)`:
   ```
   [Sxx Sxy] [du]   [-Sxt]
   [Sxy Syy] [dv] = [-Syt]
   ```
   via direct inverse (determinant = Sxx·Syy − Sxy²; if determinant is near zero — low-texture/aperture-problem region — skip update or damp it, e.g. Tikhonov regularization: add small ε to Sxx, Syy diagonal).
7. **Update flow:** `d = d + (du, dv)`. Optionally iterate steps 2–7 a few times per level (2–4 Newton iterations) for better convergence — this is what makes it "iterative LK" rather than one-shot.
8. **Pass d to next finer level** (upsample ×2 as in step 1).
9. At the finest level, the resulting `d` field **is the final per-pixel optical flow** for this frame pair.

**Implementation notes for WebGPU:**
- This is naturally expressed as a compute shader with one invocation per pixel (workgroup size e.g. 8×8).
- The window-sum step (Sxx, Sxy, etc.) is the expensive part — implement as a separable box filter (horizontal pass + vertical pass) over a per-pixel product texture (Ix·Ix, Ix·Iy, Iy·Iy, Ix·It, Iy·It — 5 textures), rather than a naive nested loop per pixel, or you'll pay O(w²) per pixel and kill performance. This is the single most important optimization in the whole project.
- Store flow as `rg32float` texture: R = vx, G = vy (in pixels/frame, in *working-resolution* pixel units — remember to rescale when mapping to display resolution).
- Ping-pong between two flow textures per pyramid level (can't read and write the same texture in one pass).

**Fallback / easier v1 alternative:** if pyramidal LK proves too much to implement first pass, implement **single-scale block matching** (for each block, e.g. 8×8, search a small window, e.g. ±8px, in the previous frame for best SAD/SSD match) as a compute shader — much simpler logic, coarser/blockier output, still gives a convincing demo, and can be upgraded to LK later without changing the surrounding pipeline.

### 4.6 Visualization

**A. Velocity heatmap (fragment shader, full-screen quad sampling the flow texture):**
- Convert (vx, vy) to (magnitude, angle).
- Map angle → hue, magnitude → saturation/value (HSV colorwheel — this is the classic Middlebury optical-flow visualization), or magnitude-only → a perceptually uniform colormap (viridis/magma) if you want a simpler "heat" look.
- Alpha-blend this over the video, e.g. `alpha = clamp(magnitude / maxSpeed, 0, 0.7)` so static areas stay transparent (showing the video) and fast-moving areas glow with color.

**B. Vector field (arrows / quiver plot):**
- Don't draw one arrow per pixel — subsample on a grid (e.g. every 10–16 px in working resolution → maps to a coarser grid on the full-res canvas).
- Two implementation options:
  1. **WebGPU instanced draw:** one small arrow/line mesh (a few vertices), instanced once per grid cell, with per-instance data (position, vx, vy) read from the flow texture in the vertex shader via `textureLoad`. Rotate/scale the arrow mesh by the vector's angle/magnitude in the vertex shader.
  2. **2D Canvas overlay (simpler):** read back the flow texture to CPU (`GPUBuffer` copy + `mapAsync`) at a coarse grid resolution only (e.g. 40×30 samples — cheap), then draw arrows with Canvas 2D `strokeLine`/small triangle heads on a transparent `<canvas>` layered on top of the WebGPU canvas. Much easier to get right first; the GPU readback of a *small* grid is cheap enough not to stall the pipeline meaningfully. Recommended for v1.

### 4.7 Compositing
- Layer order (bottom → top): live video (drawn to canvas or as WebGPU texture sample) → heatmap (alpha blended) → vector arrows (opaque, on top).
- If using the 2D-canvas-overlay approach for arrows, this is literally two stacked `<canvas>` elements (`position: absolute`) — WebGPU canvas below, 2D canvas above with `pointer-events: none`.

### 4.8 Frame Loop / State Management
- Maintain: `previousFramePyramid[levels]`, `currentFramePyramid[levels]`, ping-ponged each frame (swap references, don't recopy).
- On the very first frame, skip flow computation (no previous frame yet) — just display video.
- Frame timing: use `requestVideoFrameCallback` to drive the loop so flow is computed once per actual new camera frame, not once per display refresh (avoids wasted recomputation if camera fps < display refresh rate).

---

## 5. File / Module Structure (suggested)

```
/src
  main.js                 — app entry, sets up canvas, video, animation loop
  webgpu-context.js       — device/adapter init, canvas context config
  capture.js              — camera setup, video element, frame import
  pipeline/
    grayscale.js          — grayscale + downsample compute pass
    pyramid.js            — Gaussian blur + downsample pyramid builder
    optical-flow.js        — pyramidal LK compute passes (per-level orchestration)
    visualize-heatmap.js  — fragment shader pass for velocity heatmap
    visualize-vectors.js  — arrow overlay (instanced draw or 2D canvas readback)
  shaders/
    grayscale.wgsl
    gaussian-blur.wgsl
    downsample.wgsl
    lk-gradients.wgsl      — compute Ix, Iy, It and products
    lk-boxsum.wgsl         — separable box filter for structure tensor sums
    lk-solve.wgsl          — solve 2x2 system, update flow
    upsample-flow.wgsl     — upsample + scale flow between pyramid levels
    heatmap.wgsl
    vector-arrow.wgsl       — (if instanced-draw approach)
  utils/
    texture-utils.js       — helpers for creating/ping-ponging textures
    colormap.js             — HSV/viridis colormap helper (JS-side constants, or inline in WGSL)
index.html
```

---

## 6. Key Data Structures

**Flow texture:** `rg32float`, working resolution, one per pyramid level during computation, final one at level 0 used for visualization.

**Pyramid level texture:** `r32float`, halved resolution per level.

**Uniforms buffer** (per-pass, small struct), e.g.:
```wgsl
struct FlowParams {
  levelWidth: u32,
  levelHeight: u32,
  windowRadius: i32,     // e.g. 7 for a 15x15 window
  regularizationEps: f32,
  iterationsPerLevel: u32,
};
```

---

## 7. Performance Targets & Levers

- Target: 480×270 working resolution, 4 pyramid levels, 15×15 window, 2 iterations/level, on a mid-range laptop GPU → aim for real-time (20–30fps).
- Levers if too slow, in order of impact:
  1. Lower working resolution further (320×180).
  2. Fewer pyramid levels (3) or fewer iterations per level (1).
  3. Smaller LK window (11×11).
  4. Reduce arrow-overlay readback frequency (e.g. every 2nd frame) — visualization doesn't need to match flow computation rate exactly.
  5. Skip full recompute every frame — compute flow at e.g. 15fps while displaying video at 30fps, interpolating/holding the last flow field.

---

## 8. Known Pitfalls / Edge Cases

- **Aperture problem:** Lucas-Kanade fails on low-texture regions (blank walls) — the structure tensor becomes near-singular. Regularize (add epsilon to diagonal) and/or mask out low-confidence regions (low determinant) in the visualization (fade them out).
- **Large uniform motion (whole camera pan):** should be handled fine by the pyramid — verify with a deliberate pan test.
- **Lighting flicker:** camera auto-exposure changes can look like motion. Consider a mild temporal smoothing filter on brightness, or ignore — acceptable for a demo project.
- **`GPUExternalTexture` lifetime:** only valid within the `GPUCommandEncoder`'s current submission — never cache it across frames; always copy to a persistent texture immediately if you need to keep the data.
- **Texture read/write hazards:** never read and write the same `GPUTexture` in one dispatch — always ping-pong between two textures per stage.
- **Browser support:** WebGPU is not universal yet (Safari support is newer/partial). Feature-detect `navigator.gpu` and show a fallback message.
- **Permissions:** camera access requires HTTPS (or localhost) and a user gesture in many browsers — trigger `getUserMedia` from a button click, not on page load automatically.

---

## 9. Suggested Build Order (milestones)

1. **M1 — Camera passthrough:** get webcam feed rendering to a WebGPU canvas, no processing. Validates capture + import pipeline.
2. **M2 — Grayscale + downsample:** confirm working-resolution grayscale texture looks correct (debug by rendering it directly to canvas).
3. **M3 — Pyramid:** build and visually verify each pyramid level (render each level to a corner of the screen for debugging).
4. **M4 — Block-matching flow (fallback algorithm):** get *something* moving-vectors-shaped working end-to-end fast, even if crude. Validates the whole downstream visualization pipeline before tackling LK math.
5. **M5 — Heatmap visualization:** wire block-matching output into the HSV/viridis heatmap overlay.
6. **M6 — Vector arrow overlay:** 2D canvas readback + arrow drawing.
7. **M7 — Replace block matching with single-scale Lucas-Kanade:** verify against block matching output on the same input.
8. **M8 — Add pyramid levels to LK (full pyramidal LK):** test with fast hand motion to confirm large displacements are now tracked.
9. **M9 — Performance pass:** profile, apply levers from Section 7.
10. **M10 — Polish:** UI controls (window size, pyramid levels, colormap toggle, pause/reset), on-screen fps counter.

---

## 10. Validation / Testing Approach

- **Static scene test:** camera pointed at a still scene — flow field should be ~zero everywhere (sanity check for noise floor).
- **Uniform pan test:** move the whole camera left/right steadily — flow vectors should point uniformly in one direction with consistent magnitude.
- **Local motion test:** wave a hand in front of a static background — flow should be localized to the hand region, background near-zero.
- **Fast motion test:** quick hand swipe — confirms pyramid is actually helping (compare pyramidal vs single-scale output side by side).
- **Occlusion/edge test:** object moving in front of another — expect visible flow discontinuity at the boundary (this is inherent to LK, not a bug — worth understanding/explaining, not "fixing").

---

## 11. Stretch Goals (post-v1)

- Switch structure-tensor computation to use mip-mapped textures instead of manually-managed level array for cleaner code.
- Add a "confidence mask" visualization (from structure tensor determinant) to show where flow estimates are trustworthy.
- Export flow field as data (CSV/JSON) for a given frame, for offline analysis.
- Replace hand-rolled LK with a compute-shader port of Farneback dense flow for smoother, less blocky fields (harder, but denser and more robust than LK to aperture-problem gaps).
