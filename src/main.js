/**
 * Main entry point — sets up WebGPU, camera, pipeline stages, and animation loop.
 */

import { initWebGPU } from './webgpu-context.js';
import { startCapture, importFrame } from './capture.js';
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
const gpuCanvas = document.getElementById('gpu-canvas');
const arrowCanvas = document.getElementById('arrow-canvas');
const video = document.getElementById('camera-video');
const startBtn = document.getElementById('start-btn');
const errorMsg = document.getElementById('error-msg');
const controls = document.getElementById('controls');
const fpsValue = document.getElementById('fps-value');
const flowTimeEl = document.getElementById('flow-time');
const resValue = document.getElementById('res-value');
const heatmapOpacityInput = document.getElementById('heatmap-opacity');
const arrowsToggle = document.getElementById('arrows-toggle');
const resolutionSelect = document.getElementById('resolution-select');

// ── State ──────────────────────────────────────────────────────────
let device, context, canvasFormat;
let capture;
let grayscalePass, pyramidBuilder, opticalFlowPass, heatmapPass, vectorOverlay;
let compositePipeline, compositeSampler;
let frameCount = 0;
let hasFirstFrame = false;
let isReadingBack = false;
let lastFpsTime = performance.now();
let fpsFrameCount = 0;

// ── Error Display ──────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.style.display = 'block';
  startBtn.disabled = true;
}

// ── Composite Pipeline Init ────────────────────────────────────────
async function initCompositePipeline() {
  const shaderCode = await fetch('src/shaders/composite.wgsl').then((r) => r.text());
  const shaderModule = device.createShaderModule({
    label: 'composite-shader',
    code: shaderCode,
  });

  compositeSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  compositePipeline = device.createRenderPipeline({
    label: 'composite-pipeline',
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format: canvasFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });
}

// ── Canvas Sizing ──────────────────────────────────────────────────
function resizeCanvases() {
  const container = gpuCanvas.parentElement;
  const w = container.clientWidth;
  const h = container.clientHeight;

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
    // Init WebGPU
    const gpuInit = await initWebGPU(gpuCanvas);
    device = gpuInit.device;
    context = gpuInit.context;
    canvasFormat = gpuInit.format;

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

    await initCompositePipeline();

    // Update UI
    resValue.textContent = `${WORKING_WIDTH}×${WORKING_HEIGHT}`;
    controls.style.display = 'flex';
    startBtn.style.display = 'none';

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
    await new Promise((r) => setTimeout(r, 200));

    // Start frame loop
    requestFrame();
  } catch (err) {
    showError(err.message);
    console.error(err);
  }
}

// ── Frame Loop ─────────────────────────────────────────────────────
function requestFrame() {
  // Prefer requestVideoFrameCallback for accurate per-frame timing
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(onFrame);
  } else {
    requestAnimationFrame(onFrame);
  }
}

async function onFrame() {
  // Schedule next frame immediately
  requestFrame();

  // Wait for all pipelines to be ready
  if (
    !grayscalePass.ready ||
    !pyramidBuilder.ready ||
    !opticalFlowPass.ready ||
    !heatmapPass.ready ||
    !compositePipeline
  ) {
    return;
  }

  const frameStart = performance.now();

  // 1. Import current camera frame to GPU texture
  importFrame(device, capture);

  // 2. Create command encoder
  const encoder = device.createCommandEncoder({ label: 'frame-encoder' });

  // 3. Grayscale + downsample
  grayscalePass.encode(encoder, capture.frameTexture);

  // 4. Build pyramid
  pyramidBuilder.encode(encoder, grayscalePass.outputTexture);

  // 5. Optical flow (if we have a previous frame)
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

  // 6. Render: composite video + heatmap
  const canvasView = context.getCurrentTexture().createView();

  // First render pass: draw video
  const compositeBindGroup = device.createBindGroup({
    layout: compositePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: capture.frameTexture.createView() },
      { binding: 1, resource: compositeSampler },
    ],
  });

  // Use a single render pass with load: 'clear' for video, then heatmap on top
  const renderPass = encoder.beginRenderPass({
    label: 'composite-pass',
    colorAttachments: [
      {
        view: canvasView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });

  // Draw video
  renderPass.setPipeline(compositePipeline);
  renderPass.setBindGroup(0, compositeBindGroup);
  renderPass.draw(3);

  // Draw heatmap overlay (alpha blended on top)
  if (hasFirstFrame) {
    heatmapPass.encodeInPass(renderPass, opticalFlowPass.flowTexture);
  }

  renderPass.end();

  // 7. Submit
  device.queue.submit([encoder.finish()]);

  // 8. Swap pyramids for next frame
  pyramidBuilder.swapPyramids();
  hasFirstFrame = true;

  // 9. Read back flow for vector arrows (async, non-blocking)
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

  // 10. FPS counter
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

// ── Start Button Handler ───────────────────────────────────────────
startBtn.addEventListener('click', () => {
  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';
  init();
});

// Feature detection on load
if (!navigator.gpu) {
  showError(
    'WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+ with WebGPU enabled.'
  );
}
