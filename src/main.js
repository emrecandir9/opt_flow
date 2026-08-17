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
import { createGrayscaleTexture } from './utils/texture-utils.js';

// ── Configuration ──────────────────────────────────────────────────
const WORKING_WIDTH = 480;
const WORKING_HEIGHT = 270;
const PYRAMID_LEVELS = 4;
const BLOCK_SIZE = 8;
const SEARCH_RADIUS = 12;
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
const sensitivitySelect = document.getElementById('sensitivity-select');
const simBtn = document.getElementById('sim-btn');

// ── 2D contexts ────────────────────────────────────────────────────
const videoCtx = videoCanvas.getContext('2d');

// ── State ──────────────────────────────────────────────────────────
let device, context, canvasFormat;
let capture;
let grayscalePass, prevGrayscaleTexture, pyramidBuilder, opticalFlowPass, heatmapPass, vectorOverlay;
let frameCount = 0;
let hasFirstFrame = false;
let isReadingBack = false;
let isRunning = false;
let simMode = false;
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
  const w = container.clientWidth || 960;
  const h = container.clientHeight || 540;

  videoCanvas.width = w;
  videoCanvas.height = h;
  gpuCanvas.width = w;
  gpuCanvas.height = h;
  arrowCanvas.width = w;
  arrowCanvas.height = h;

  if (context && device) {
    context.configure({
      device,
      format: canvasFormat,
      alphaMode: 'premultiplied',
    });
  }

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

    prevGrayscaleTexture = createGrayscaleTexture(
      device,
      WORKING_WIDTH,
      WORKING_HEIGHT,
      'prev-grayscale'
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
    document.getElementById('heatmap-opacity')?.addEventListener('input', (e) => {
      const opacity = parseInt(e.target.value) / 100;
      heatmapPass?.setParams(6.0, opacity);
    });

    document.getElementById('arrows-toggle')?.addEventListener('change', (e) => {
      if (vectorOverlay) {
        vectorOverlay.enabled = e.target.value === 'on';
        if (!vectorOverlay.enabled) {
          vectorOverlay.draw(); // Clear arrows
        }
      }
    });

    const simBtnEl = document.getElementById('sim-btn');
    simBtnEl?.addEventListener('click', () => {
      simMode = !simMode;
      simBtnEl.textContent = simMode ? 'Stop Test Motion' : 'Test Motion Overlay';
      simBtnEl.style.background = simMode ? '#4f46e5' : '#374151';
    });

    // Wait for async pipeline compilation
    await new Promise((r) => setTimeout(r, 500));

    console.log('Pipeline ready:', {
      grayscale: grayscalePass.ready,
      opticalFlow: opticalFlowPass.ready,
      heatmap: heatmapPass.ready,
    });

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
  simMode = false;
  if (simBtn) {
    simBtn.textContent = 'Test Motion Overlay';
    simBtn.style.background = '#374151';
  }
}

// ── Frame Loop ─────────────────────────────────────────────────────
function requestFrame() {
  if (!isRunning) return;

  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(onFrame);
  } else {
    requestAnimationFrame(onFrame);
  }
}

function onFrame() {
  if (!isRunning || !capture) return;

  // Schedule next frame
  requestFrame();

  const frameStart = performance.now();

  // ── Layer 0: Draw video to the video canvas (2D) ──────────────
  videoCtx.drawImage(capture.video, 0, 0, videoCanvas.width, videoCanvas.height);

  // If simulation mode is active, draw an animated moving target onto canvas
  if (simMode) {
    const t = performance.now() * 0.0006;
    const sx = (Math.sin(t * 1.5) * 0.35 + 0.5) * videoCanvas.width;
    const sy = (Math.cos(t * 1.2) * 0.3 + 0.5) * videoCanvas.height;

    videoCtx.save();
    videoCtx.fillStyle = '#ffffff';
    videoCtx.beginPath();
    videoCtx.arc(sx, sy, 40, 0, Math.PI * 2);
    videoCtx.fill();
    videoCtx.fillStyle = '#000000';
    videoCtx.fillRect(sx - 20, sy - 20, 20, 20);
    videoCtx.fillRect(sx, sy, 20, 20);
    videoCtx.restore();

    // Also draw onto capture canvas so GPU sees it
    const capX = (sx / videoCanvas.width) * capture.cameraWidth;
    const capY = (sy / videoCanvas.height) * capture.cameraHeight;
    capture.captureCtx.drawImage(capture.video, 0, 0, capture.cameraWidth, capture.cameraHeight);
    capture.captureCtx.save();
    capture.captureCtx.fillStyle = '#ffffff';
    capture.captureCtx.beginPath();
    capture.captureCtx.arc(capX, capY, (40 / videoCanvas.width) * capture.cameraWidth, 0, Math.PI * 2);
    capture.captureCtx.fill();
    capture.captureCtx.fillStyle = '#000000';
    const sSize = (20 / videoCanvas.width) * capture.cameraWidth;
    capture.captureCtx.fillRect(capX - sSize, capY - sSize, sSize, sSize);
    capture.captureCtx.fillRect(capX, capY, sSize, sSize);
    capture.captureCtx.restore();
  }

  // ── GPU Pipeline: grayscale → flow ────────────────────────────
  const gpuReady =
    grayscalePass?.ready &&
    opticalFlowPass?.ready &&
    heatmapPass?.ready;

  if (gpuReady) {
    // Import frame to GPU
    if (!simMode) {
      importFrame(device, capture);
    } else {
      // In sim mode, captureCtx already has the combined frame
      const imgData = capture.captureCtx.getImageData(0, 0, capture.cameraWidth, capture.cameraHeight);
      device.queue.writeTexture(
        { texture: capture.frameTexture },
        imgData.data,
        { bytesPerRow: capture.cameraWidth * 4, rowsPerImage: capture.cameraHeight },
        [capture.cameraWidth, capture.cameraHeight]
      );
    }

    const encoder = device.createCommandEncoder({ label: 'frame-encoder' });

    // 1. Grayscale + downsample
    grayscalePass.encode(encoder, capture.frameTexture);

    // 2. Optical flow (compare current crisp grayscale with previous frame grayscale)
    if (hasFirstFrame) {
      opticalFlowPass.encode(
        encoder,
        grayscalePass.outputTexture,
        prevGrayscaleTexture
      );

      // Copy flow for readback (for vector arrows)
      if (!isReadingBack) {
        opticalFlowPass.encodeCopyForReadback(encoder);
      }
    }

    // 3. ── Layer 1: Heatmap on transparent WebGPU canvas ──────────
    const canvasView = context.getCurrentTexture().createView();

    const renderPass = encoder.beginRenderPass({
      label: 'heatmap-pass',
      colorAttachments: [
        {
          view: canvasView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    if (hasFirstFrame) {
      heatmapPass.encodeInPass(renderPass, opticalFlowPass.flowTexture);
    }

    renderPass.end();

    // 4. Save current grayscale into prevGrayscaleTexture for next frame
    encoder.copyTextureToTexture(
      { texture: grayscalePass.outputTexture },
      { texture: prevGrayscaleTexture },
      [WORKING_WIDTH, WORKING_HEIGHT]
    );

    device.queue.submit([encoder.finish()]);
    hasFirstFrame = true;

    // 5. ── Layer 2: Read back flow for vector arrows ──────────────
    if (hasFirstFrame && !isReadingBack && vectorOverlay.enabled) {
      isReadingBack = true;
      opticalFlowPass
        .readFlowData()
        .then((flowData) => {
          vectorOverlay.setFlowData(flowData);
          vectorOverlay.draw();
          isReadingBack = false;
        })
        .catch((err) => {
          console.error('Flow readback error:', err);
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
