/**
 * Grayscale conversion + downsample pipeline stage.
 * Converts RGBA camera frame to R32Float grayscale at working resolution.
 */

import { createGrayscaleTexture } from '../utils/texture-utils.js';

export class GrayscalePass {
  /**
   * @param {GPUDevice} device
   * @param {number} srcWidth  - Camera resolution width
   * @param {number} srcHeight - Camera resolution height
   * @param {number} dstWidth  - Working resolution width
   * @param {number} dstHeight - Working resolution height
   */
  constructor(device, srcWidth, srcHeight, dstWidth, dstHeight) {
    this.device = device;
    this.dstWidth = dstWidth;
    this.dstHeight = dstHeight;

    // Output texture
    this.outputTexture = createGrayscaleTexture(device, dstWidth, dstHeight, 'grayscale-output');

    // Uniform buffer
    this.uniformBuffer = device.createBuffer({
      size: 16, // 4 x u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Uint32Array([srcWidth, srcHeight, dstWidth, dstHeight])
    );

    // Pipeline (created asynchronously)
    this.pipeline = null;
    this.bindGroup = null;

    this._initPipeline();
  }

  async _initPipeline() {
    const shaderCode = await fetch('src/shaders/grayscale.wgsl').then((r) => r.text());
    const shaderModule = this.device.createShaderModule({
      label: 'grayscale-shader',
      code: shaderCode,
    });

    this.pipeline = this.device.createComputePipeline({
      label: 'grayscale-pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  /**
   * Create bind group for this pass.
   * Must be called after pipeline is ready and when input texture changes.
   * @param {GPUTexture} inputTexture - The camera frame RGBA texture
   */
  createBindGroup(inputTexture) {
    if (!this.pipeline) return;

    this.bindGroup = this.device.createBindGroup({
      label: 'grayscale-bind-group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: inputTexture.createView() },
        { binding: 1, resource: this.outputTexture.createView() },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });
  }

  /**
   * Encode the grayscale compute pass.
   * @param {GPUCommandEncoder} encoder
   * @param {GPUTexture} inputTexture
   */
  encode(encoder, inputTexture) {
    if (!this.pipeline) return;

    // Recreate bind group each frame since input texture may differ
    this.createBindGroup(inputTexture);
    if (!this.bindGroup) return;

    const pass = encoder.beginComputePass({ label: 'grayscale-pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.dstWidth / 8),
      Math.ceil(this.dstHeight / 8)
    );
    pass.end();
  }

  /**
   * Check if pipeline is ready.
   */
  get ready() {
    return this.pipeline !== null;
  }

  destroy() {
    this.outputTexture.destroy();
    this.uniformBuffer.destroy();
  }
}
