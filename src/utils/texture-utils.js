/**
 * Texture utility helpers for creating/managing GPU textures.
 */

/**
 * Create a grayscale (r32float) texture at the given resolution.
 */
export function createGrayscaleTexture(device, width, height, label = 'grayscale') {
  return device.createTexture({
    label,
    size: [width, height],
    format: 'r32float',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
}

/**
 * Create an RGBA texture for video frame capture.
 */
export function createRGBATexture(device, width, height, label = 'rgba') {
  return device.createTexture({
    label,
    size: [width, height],
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

/**
 * Create a flow texture (rg32float) for motion vectors.
 */
export function createFlowTexture(device, width, height, label = 'flow') {
  return device.createTexture({
    label,
    size: [width, height],
    format: 'rg32float',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
}

/**
 * Create a pair of textures for ping-pong rendering.
 * Returns [textureA, textureB].
 */
export function createPingPongTextures(device, width, height, format, label = 'pingpong') {
  const usage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST;

  return [
    device.createTexture({ label: `${label}_A`, size: [width, height], format, usage }),
    device.createTexture({ label: `${label}_B`, size: [width, height], format, usage }),
  ];
}

/**
 * Create pyramid of grayscale textures at halving resolutions.
 * Level 0 = full working resolution, level N = (width >> N, height >> N).
 */
export function createPyramidTextures(device, width, height, levels, label = 'pyramid') {
  const textures = [];
  for (let i = 0; i < levels; i++) {
    const w = Math.max(1, width >> i);
    const h = Math.max(1, height >> i);
    textures.push(createGrayscaleTexture(device, w, h, `${label}_L${i}`));
  }
  return textures;
}
