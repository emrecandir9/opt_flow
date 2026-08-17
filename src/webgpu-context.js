/**
 * WebGPU context initialization.
 * Handles adapter/device request and canvas configuration.
 */

/**
 * Initialize WebGPU: request adapter, device, and configure canvas context.
 * @param {HTMLCanvasElement} canvas
 * @returns {{ device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat }}
 */
export async function initWebGPU(canvas) {
  // Feature detection
  if (!navigator.gpu) {
    throw new Error(
      'WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.'
    );
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) {
    throw new Error(
      'Failed to get GPU adapter. Your device may not support WebGPU.'
    );
  }

  // Request device
  const device = await adapter.requestDevice({
    requiredFeatures: [],
    requiredLimits: {},
  });

  // Handle device loss
  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.reason}`, info.message);
    if (info.reason !== 'destroyed') {
      // Could attempt re-init here
      console.warn('Device lost unexpectedly. Refresh the page to recover.');
    }
  });

  // Configure canvas context
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: 'premultiplied',
  });

  return { device, context, format };
}
