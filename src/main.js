/**
 * Main entry point — sets up WebGPU, camera, pipeline stages, and multi-mode interactive visualizers.
 *
 * Visualizer Modes:
 *   1. ✨ particles   — 3,500 glowing embers pushed by motion force-fields
 *   2. 🌊 streamlines — Aerodynamic wind-tunnel flow ribbons
 *   3. 🔥 thermal     — Smooth non-blocky speed heatmap silhouette
 *   4. 🎯 arrows      — Sleek quiver plot with circular direction compass
 */

import { initWebGPU } from './webgpu-context.js';
import { startCapture, importFrame, stopCapture } from './capture.js';
import { GrayscalePass } from './pipeline/grayscale.js';
import { OpticalFlowPass } from './pipeline/optical-flow.js';
import { HeatmapPass } from './pipeline/visualize-heatmap.js';
import { VectorOverlay } from './pipeline/visualize-vectors.js';
import { ParticleSystem } from './pipeline/visualize-particles.js';
import { createGrayscaleTexture } from './utils/texture-utils.js';

// ── Configuration ──────────────────────────────────────────────────
const WORKING_WIDTH = 480;
const WORKING_HEIGHT = 270;
const BLOCK_SIZE = 8;
const SEARCH_RADIUS = 9;
const ARROW_GRID_SPACING = 14;

// ── DOM Elements ───────────────────────────────────────────────────
const videoCanvas = document.getElementById('video-canvas');
const gpuCanvas = document.getElementById('gpu-canvas');
const arrowCanvas = document.getElementById('arrow-canvas');
const video = document.getElementById('camera-video');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const errorMsg = document.getElementById('error-msg');
const modeBar = document.getElementById('mode-bar');
const controls = document.getElementById('controls');
const fpsValue = document.getElementById('fps-value');
const flowTimeEl = document.getElementById('flow-time');
const resValue = document.getElementById('res-value');
const compassNeedle = document.getElementById('compass-needle');
const compassSpeed = document.getElementById('compass-speed');
const compassLabel = document.getElementById('compass-label');
const intensitySlider = document.getElementById('intensity-slider');
const simBtn = document.getElementById('sim-btn');

// ── 2D contexts ────────────────────────────────────────────────────
const videoCtx = videoCanvas.getContext('2d');

// ── State ──────────────────────────────────────────────────────────
let device, context, canvasFormat;
let capture;
let grayscalePass, prevGrayscaleTexture, opticalFlowPass, heatmapPass, vectorOverlay, particleSystem;
let currentMode = 'particles'; // 'particles' | 'streamlines' | 'thermal' | 'arrows'
let currentIntensity = 0.8;
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

  if (vectorOverlay) vectorOverlay.resize(w, h);
  if (particleSystem) particleSystem.resize(w, h);
}

// ── Mode Switching ─────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;

  // Update tab buttons
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  if (!heatmapPass || !vectorOverlay || !particleSystem) return;

  // Configure sub-systems
  if (mode === 'particles') {
    particleSystem.enabled = true;
    vectorOverlay.enabled = false;
    heatmapPass.enabled = false;
    compassLabel.textContent = 'Force Field';
  } else if (mode === 'streamlines') {
    vectorOverlay.enabled = true;
    vectorOverlay.mode = 'streamlines';
    particleSystem.enabled = false;
    heatmapPass.enabled = false;
    compassLabel.textContent = 'Wind Flow';
  } else if (mode === 'thermal') {
    heatmapPass.enabled = true;
    heatmapPass.setParams(6.0, currentIntensity, 0.0); // Thermal Speed Glow
    vectorOverlay.enabled = false;
    particleSystem.enabled = false;
    compassLabel.textContent = 'Speed Heatmap';
  } else if (mode === 'arrows') {
    vectorOverlay.enabled = true;
    vectorOverlay.mode = 'arrows';
    heatmapPass.enabled = true;
    heatmapPass.setParams(6.0, currentIntensity * 0.4, 1.0); // Subtle HSV
    particleSystem.enabled = false;
    compassLabel.textContent = 'Quiver Vectors';
  }

  // Clear 2D overlay on switch
  const overlayCtx = arrowCanvas.getContext('2d');
  overlayCtx.clearRect(0, 0, arrowCanvas.width, arrowCanvas.height);
}

// ── Main Init ──────────────────────────────────────────────────────
async function init() {
  try {
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

    particleSystem = new ParticleSystem(
      arrowCanvas,
      WORKING_WIDTH,
      WORKING_HEIGHT,
      3500
    );
    particleSystem.resize(arrowCanvas.width, arrowCanvas.height);

    // Update UI
    resValue.textContent = `${WORKING_WIDTH}×${WORKING_HEIGHT}`;
    modeBar.style.display = 'flex';
    controls.style.display = 'flex';
    startBtn.style.display = 'none';
    stopBtn.style.display = 'inline-block';

    // Set initial mode
    setMode('particles');

    // Bind Mode Switcher tabs
    document.querySelectorAll('.mode-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        setMode(btn.dataset.mode);
      });
    });

    // Bind Controls
    intensitySlider?.addEventListener('input', (e) => {
      currentIntensity = parseInt(e.target.value) / 100;
      if (currentMode === 'thermal') {
        heatmapPass?.setParams(6.0, currentIntensity, 0.0);
      } else if (currentMode === 'arrows') {
        heatmapPass?.setParams(6.0, currentIntensity * 0.4, 1.0);
      }
    });

    simBtn?.addEventListener('click', () => {
      simMode = !simMode;
      simBtn.textContent = simMode ? 'Stop Motion Target' : 'Test Motion Target';
      simBtn.style.background = simMode ? 'rgba(99, 102, 241, 0.3)' : 'rgba(255, 255, 255, 0.05)';
      simBtn.style.borderColor = simMode ? '#818cf8' : 'rgba(255, 255, 255, 0.12)';
    });

    // Wait for async pipeline compilation
    await new Promise((r) => setTimeout(r, 400));

    console.log('Pipelines ready.');

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

  videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
  if (vectorOverlay) vectorOverlay.destroy();
  if (particleSystem) particleSystem.destroy();

  startBtn.style.display = 'inline-block';
  startBtn.disabled = false;
  startBtn.textContent = 'Start Camera';
  stopBtn.style.display = 'none';
  modeBar.style.display = 'none';
  controls.style.display = 'none';
  fpsValue.textContent = '—';
  flowTimeEl.textContent = '—';
  resValue.textContent = '—';
  compassSpeed.textContent = '0.0 px/f';
  compassNeedle.style.transform = 'translate(-50%, -100%) rotate(0deg)';
  hasFirstFrame = false;
  isReadingBack = false;
  simMode = false;
  if (simBtn) {
    simBtn.textContent = 'Test Motion Target';
    simBtn.style.background = 'rgba(255, 255, 255, 0.05)';
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

  // ── Layer 0: Draw video to 2D canvas ──────────────────────────
  videoCtx.drawImage(capture.video, 0, 0, videoCanvas.width, videoCanvas.height);

  // If simulation mode is active, draw animated test target
  if (simMode) {
    const t = performance.now() * 0.0007;
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

  // ── GPU Pipeline: Grayscale → Optical Flow → Heatmap ──────────
  const gpuReady =
    grayscalePass?.ready &&
    opticalFlowPass?.ready &&
    heatmapPass?.ready;

  if (gpuReady) {
    if (!simMode) {
      importFrame(device, capture);
    } else {
      const imgData = capture.captureCtx.getImageData(0, 0, capture.cameraWidth, capture.cameraHeight);
      device.queue.writeTexture(
        { texture: capture.frameTexture },
        imgData.data,
        { bytesPerRow: capture.cameraWidth * 4, rowsPerImage: capture.cameraHeight },
        [capture.cameraWidth, capture.cameraHeight]
      );
    }

    const encoder = device.createCommandEncoder({ label: 'frame-encoder' });

    // 1. Grayscale conversion
    grayscalePass.encode(encoder, capture.frameTexture);

    // 2. Optical flow computation
    if (hasFirstFrame) {
      opticalFlowPass.encode(
        encoder,
        grayscalePass.outputTexture,
        prevGrayscaleTexture
      );

      if (!isReadingBack) {
        opticalFlowPass.encodeCopyForReadback(encoder);
      }
    }

    // 3. Layer 1: Heatmap pass on transparent WebGPU canvas
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

    if (hasFirstFrame && heatmapPass.enabled) {
      heatmapPass.encodeInPass(renderPass, opticalFlowPass.flowTexture);
    }

    renderPass.end();

    // 4. Save current grayscale into prevGrayscaleTexture
    encoder.copyTextureToTexture(
      { texture: grayscalePass.outputTexture },
      { texture: prevGrayscaleTexture },
      [WORKING_WIDTH, WORKING_HEIGHT]
    );

    device.queue.submit([encoder.finish()]);
    hasFirstFrame = true;

    // 5. Layer 2: Readback for Particles, Streamlines, and Arrows ──
    if (hasFirstFrame && !isReadingBack) {
      isReadingBack = true;
      opticalFlowPass
        .readFlowData()
        .then((flowData) => {
          // Update Particle System
          if (currentMode === 'particles') {
            particleSystem.setFlowData(flowData);
            particleSystem.update(1.0);
            particleSystem.draw();
          } else if (currentMode === 'streamlines' || currentMode === 'arrows') {
            vectorOverlay.setFlowData(flowData);
            vectorOverlay.draw();
          }

          // Update Motion Direction Compass HUD Widget
          updateCompassHUD(flowData);

          isReadingBack = false;
        })
        .catch(() => {
          isReadingBack = false;
        });
    } else if (currentMode === 'particles') {
      // Keep particles drifting smoothly even on non-readback frames
      particleSystem.update(1.0);
      particleSystem.draw();
    }
  }

  // ── FPS Counter ──────────────────────────────────────────────────
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

// ── Compass HUD Updater ────────────────────────────────────────────
function updateCompassHUD(flowData) {
  if (!flowData) return;

  let sumVx = 0, sumVy = 0, count = 0, maxSpeed = 0;

  for (let i = 0; i < flowData.length; i += 8) {
    const vx = flowData[i];
    const vy = flowData[i + 1];
    const speed = Math.sqrt(vx * vx + vy * vy);
    if (speed > 0.3) {
      sumVx += vx;
      sumVy += vy;
      count++;
      if (speed > maxSpeed) maxSpeed = speed;
    }
  }

  if (count > 5) {
    const avgVx = sumVx / count;
    const avgVy = sumVy / count;
    const angleRad = Math.atan2(avgVy, avgVx);
    const angleDeg = (angleRad * 180) / Math.PI;

    compassNeedle.style.transform = `translate(-50%, -100%) rotate(${angleDeg + 90}deg)`;
    compassSpeed.textContent = `${maxSpeed.toFixed(1)} px/f`;
  } else {
    compassSpeed.textContent = '0.0 px/f';
  }
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

if (!navigator.gpu) {
  showError(
    'WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+ with WebGPU enabled.'
  );
}
