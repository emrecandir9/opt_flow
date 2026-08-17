/**
 * Camera capture module.
 * Handles getUserMedia, video element setup, and frame import to GPU texture.
 * Uses canvas getImageData + device.queue.writeTexture for maximum reliability.
 */

import { createRGBATexture } from './utils/texture-utils.js';

/**
 * @typedef {Object} CaptureState
 * @property {HTMLVideoElement} video
 * @property {HTMLCanvasElement} captureCanvas - Canvas for video frame extraction
 * @property {CanvasRenderingContext2D} captureCtx
 * @property {GPUTexture} frameTexture - Current camera frame as GPU texture
 * @property {number} cameraWidth - Actual camera resolution width
 * @property {number} cameraHeight - Actual camera resolution height
 * @property {MediaStream} stream - The camera media stream
 */

/**
 * Start camera capture.
 * @param {HTMLVideoElement} video
 * @param {GPUDevice} device
 * @returns {Promise<CaptureState>}
 */
export async function startCapture(video, device) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
      facingMode: 'user',
    },
    audio: false,
  });

  video.srcObject = stream;

  // Wait for video to be ready
  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });

  // Wait for actual dimensions and frame data to be available
  await new Promise((resolve) => {
    const check = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) {
        resolve();
      } else {
        requestAnimationFrame(check);
      }
    };
    check();
  });

  const cameraWidth = video.videoWidth;
  const cameraHeight = video.videoHeight;

  // Create a capture canvas at camera resolution for pixel data extraction
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = cameraWidth;
  captureCanvas.height = cameraHeight;
  const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });

  // Create the GPU texture to receive video frames
  const frameTexture = createRGBATexture(device, cameraWidth, cameraHeight, 'camera-frame');

  return {
    video,
    captureCanvas,
    captureCtx,
    frameTexture,
    cameraWidth,
    cameraHeight,
    stream,
  };
}

/**
 * Copy the current video frame into the GPU texture.
 * Uses canvas drawImage + getImageData + writeTexture for maximum reliability.
 * This bypasses copyExternalImageToTexture entirely.
 * @param {GPUDevice} device
 * @param {CaptureState} capture
 */
export function importFrame(device, capture) {
  const { captureCtx, captureCanvas, video, frameTexture, cameraWidth, cameraHeight } = capture;

  // Draw video to capture canvas
  captureCtx.drawImage(video, 0, 0, cameraWidth, cameraHeight);

  // Extract raw pixel data
  const imageData = captureCtx.getImageData(0, 0, cameraWidth, cameraHeight);

  // Write pixel data directly to GPU texture
  device.queue.writeTexture(
    { texture: frameTexture },
    imageData.data,
    {
      bytesPerRow: cameraWidth * 4,
      rowsPerImage: cameraHeight,
    },
    [cameraWidth, cameraHeight]
  );
}

/**
 * Stop the camera and release resources.
 * @param {CaptureState} capture
 */
export function stopCapture(capture) {
  if (capture.stream) {
    capture.stream.getTracks().forEach((track) => track.stop());
  }
  capture.video.srcObject = null;
  capture.frameTexture.destroy();
}
