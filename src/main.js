/**
 * Main entry point — sets up WebGPU, camera, pipeline stages, and animation loop.
 *
 * Rendering architecture (3 layers):
 *   Layer 0 (bottom): video-canvas  — 2D context drawing camera feed
 *   Layer 1 (middle): gpu-canvas    — WebGPU transparent canvas for heatmap overlay
 *   Layer 2 (top):    arrow-canvas  — 2D context drawing vector arrows
 */

import { initWebGPU } from './webgpu-context.js';
import { startCapture, importFrame, stopCapture } from './capture.js';
import { GrayscalePass } from './pipeline/grayscale.js';
import { PyramidBuilder } from './pipeline/pyramid.js';
import { OpticalFlowPass } from './pipeline/optical-flow.js';
import { HeatmapPass } from './pipeline/visualize-heatmap.js';
import { VectorOverlay } from './pipeline/visualize-vectors.js';

// ── Configuration ──────────────────────────────────────────────────
const WORKING_WIDTH = 480;
const WORKING_HEIGHT = 270;
const PYRAMID_LEVELS = 4;
const BLOCK_SIZE = 8;
const SEARCH_RADIUS = 8;
const ARROW_GRID_SPACING = 12;

// ── DOM Elements ───────────────────────────────────────────────────
const videoCanvas = document.getElementById('video-canvas');
const gpuCanvas = document.getElementById('gpu-canvas');
const arrowCanvas = document.getElementById('arrow-canvas');
const video = document.getElementById('camera-video');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const errorMsg = document.getElementById('error-msg');
const controls = document.getElementById('controls');
const fpsValue = document.getElementById('fps-value');
const flowTimeEl = document.getElementById('flow-time');
const resValue = document.getElementById('res-value');
const heatmapOpacityInput = document.getElementById('heatmap-opacity');
const arrowsToggle = document.getElementById('arrows-toggle');
const resolutionSelect = document.getElementById('resolution-select');

// ── 2D contexts ────────────────────────────────────────────────────
const videoCtx = videoCanvas.getContext('2d');

// ── State ──────────────────────────────────────────────────────────
let device, context, canvasFormat;
let capture;
let grayscalePass, pyramidBuilder, opticalFlowPass, heatmapPass, vectorOverlay;
let frameCount = 0;
let hasFirstFrame = false;
let isReadingBack = false;
let isRunning = false;
let lastFpsTime = performance.now();
let fpsFrameCount = 0;

// ── Error Display ──────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.style.display = 'block';
  startBtn.disabled = true;
}

// ── Canvas Sizing ──────────────────────────────────────────────────
function resizeCanvases() {
  const container = gpuCanvas.parentElement;
  const w = container.clientWidth;
  const h = container.clientHeight;

  videoCanvas.width = w;
  videoCanvas.height = h;
  gpuCanvas.width = w;
  gpuCanvas.height = h;
  arrowCanvas.width = w;
  arrowCanvas.height = h;

  if (vectorOverlay) {
    vectorOverlay.resize(w, h);
  }
}

// ── Main Init ──────────────────────────────────────────────────────
async function init() {
  try {
    // Init WebGPU (only once)
    if (!device) {
      const gpuInit = await initWebGPU(gpuCanvas);
      device = gpuInit.device;
      context = gpuInit.context;
      canvasFormat = gpuInit.format;
    }

    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);

    // Start camera
    capture = await startCapture(video, device);
    console.log(`Camera: ${capture.cameraWidth}×${capture.cameraHeight}`);

    // Init pipeline stages
    grayscalePass = new GrayscalePass(
      device,
      capture.cameraWidth,
      capture.cameraHeight,
      WORKING_WIDTH,
      WORKING_HEIGHT
    );

    pyramidBuilder = new PyramidBuilder(device, WORKING_WIDTH, WORKING_HEIGHT, PYRAMID_LEVELS);

    opticalFlowPass = new OpticalFlowPass(
      device,
      WORKING_WIDTH,
      WORKING_HEIGHT,
      BLOCK_SIZE,
      SEARCH_RADIUS
    );

    heatmapPass = new HeatmapPass(device, canvasFormat);

    vectorOverlay = new VectorOverlay(
      arrowCanvas,
      WORKING_WIDTH,
      WORKING_HEIGHT,
      ARROW_GRID_SPACING
    );
    vectorOverlay.resize(arrowCanvas.width, arrowCanvas.height);

    // Update UI
    resValue.textContent = `${WORKING_WIDTH}×${WORKING_HEIGHT}`;
    controls.style.display = 'flex';
    startBtn.style.display = 'none';
    stopBtn.style.display = 'inline-block';

    // Bind UI controls
    heatmapOpacityInput.addEventListener('input', () => {
      const opacity = parseInt(heatmapOpacityInput.value) / 100;
      heatmapPass.setParams(10.0, opacity);
    });

    arrowsToggle.addEventListener('change', () => {
      vectorOverlay.enabled = arrowsToggle.value === 'on';
      if (!vectorOverlay.enabled) {
        vectorOverlay.draw(); // Clear arrows
      }
    });

    // Wait a bit for async pipeline compilation
    await new Promise((r) => setTimeout(r, 300));

    // Start frame loop
    isRunning = true;
    hasFirstFrame = false;
    frameCount = 0;
    requestFrame();
  } catch (err) {
    showError(err.message);
    console.error(err);
  }
}

// ── Stop Camera ────────────────────────────────────────────────────
function stop() {
  isRunning = false;

  if (capture) {
    stopCapture(capture);
    capture = null;
  }

  // Clear canvases
  videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
  if (vectorOverlay) {
    vectorOverlay.destroy();
  }

  // Reset UI
  startBtn.style.display = 'inline-block';
  startBtn.disabled = false;
  startBtn.textContent = 'Start Camera';
  stopBtn.style.display = 'none';
  controls.style.display = 'none';
  fpsValue.textContent = '—';
  flowTimeEl.textContent = '—';
  resValue.textContent = '—';
  hasFirstFrame = false;
  isReadingBack = false;
}

// ── Frame Loop ─────────────────────────────────────────────────────
function requestFrame() {
  if (!isRunning) return;

  // Prefer requestVideoFrameCallback for accurate per-frame timing
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(onFrame);
  } else {
    requestAnimationFrame(onFrame);
  }
}

function onFrame() {
  if (!isRunning || !capture) return;

  // Schedule next frame immediately
  requestFrame();

  const frameStart = performance.now();

  // ── Layer 0: Draw video to the video canvas (2D) ──────────────
  videoCtx.drawImage(capture.video, 0, 0, videoCanvas.width, videoCanvas.height);

  // ── GPU Pipeline: grayscale → pyramid → flow ──────────────────
  // Only run GPU pipeline if all stages are ready
  const gpuReady =
    grayscalePass?.ready &&
    pyramidBuilder?.ready &&
    opticalFlowPass?.ready &&
    heatmapPass?.ready;

  if (gpuReady) {
    // Import frame to GPU (for compute pipeline)
    importFrame(device, capture);

    const encoder = device.createCommandEncoder({ label: 'frame-encoder' });

    // Grayscale + downsample
    grayscalePass.encode(encoder, capture.frameTexture);

    // Build pyramid
    pyramidBuilder.encode(encoder, grayscalePass.outputTexture);

    // Optical flow (if we have a previous frame)
    if (hasFirstFrame) {
      opticalFlowPass.encode(
        encoder,
        pyramidBuilder.currentPyramid[0],
        pyramidBuilder.previousPyramid[0]
      );

      // Copy flow for readback (for vector arrows)
      if (!isReadingBack) {
        opticalFlowPass.encodeCopyForReadback(encoder);
      }
    }

    // ── Layer 1: Heatmap on transparent WebGPU canvas ──────────
    const canvasView = context.getCurrentTexture().createView();

    const renderPass = encoder.beginRenderPass({
      label: 'heatmap-pass',
      colorAttachments: [
        {
          view: canvasView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 }, // Transparent clear
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    // Draw heatmap overlay (alpha blended, transparent background)
    if (hasFirstFrame) {
      heatmapPass.encodeInPass(renderPass, opticalFlowPass.flowTexture);
    }

    renderPass.end();

    // Submit
    device.queue.submit([encoder.finish()]);

    // Swap pyramids for next frame
    pyramidBuilder.swapPyramids();
    hasFirstFrame = true;

    // ── Layer 2: Read back flow for vector arrows ──────────────
    if (hasFirstFrame && !isReadingBack && vectorOverlay.enabled) {
      isReadingBack = true;
      opticalFlowPass
        .readFlowData()
        .then((flowData) => {
          vectorOverlay.setFlowData(flowData);
          vectorOverlay.draw();
          isReadingBack = false;
        })
        .catch(() => {
          isReadingBack = false;
        });
    }
  }

  // ── FPS counter ──────────────────────────────────────────────────
  fpsFrameCount++;
  const now = performance.now();
  const elapsed = now - lastFpsTime;
  if (elapsed >= 1000) {
    const fps = Math.round((fpsFrameCount * 1000) / elapsed);
    fpsValue.textContent = `${fps}`;
    flowTimeEl.textContent = `${(now - frameStart).toFixed(1)}ms`;
    fpsFrameCount = 0;
    lastFpsTime = now;
  }

  frameCount++;
}

// ── Button Handlers ────────────────────────────────────────────────
startBtn.addEventListener('click', () => {
  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';
  errorMsg.style.display = 'none';
  init();
});

stopBtn.addEventListener('click', () => {
  stop();
});

// Feature detection on load
if (!navigator.gpu) {
  showError(
    'WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+ with WebGPU enabled.'
  );
}
