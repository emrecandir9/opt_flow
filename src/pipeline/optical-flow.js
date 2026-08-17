/**
 * Optical flow computation via block matching.
 * Processes current vs previous grayscale frames to produce motion vectors.
 */

import { createFlowTexture } from '../utils/texture-utils.js';

export class OpticalFlowPass {
  /**
   * @param {GPUDevice} device
   * @param {number} width  - Working resolution width
   * @param {number} height - Working resolution height
   * @param {number} blockSize - Block size for matching (default 8)
   * @param {number} searchRadius - Search radius in pixels (default 8)
   */
  constructor(device, width, height, blockSize = 8, searchRadius = 8) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.blockSize = blockSize;
    this.searchRadius = searchRadius;

    // Output flow texture (rg32float)
    this.flowTexture = createFlowTexture(device, width, height, 'flow-output');

    // Readback buffer for vector visualization
    // Flow is rg32float = 8 bytes per pixel
    this.readbackBuffer = device.createBuffer({
      size: width * height * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Uniform buffer
    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Uint32Array([width, height, blockSize, searchRadius])
    );

    this.pipeline = null;
    this._initPipeline();
  }

  async _initPipeline() {
    const shaderCode = await fetch('src/shaders/block-match.wgsl').then((r) => r.text());
    const shaderModule = this.device.createShaderModule({
      label: 'block-match-shader',
      code: shaderCode,
    });

    this.pipeline = this.device.createComputePipeline({
      label: 'block-match-pipeline',
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  get ready() {
    return this.pipeline !== null;
  }

  /**
   * Encode the block-matching compute pass.
   * @param {GPUCommandEncoder} encoder
   * @param {GPUTexture} currentFrame  - Current grayscale frame (R32Float, level 0)
   * @param {GPUTexture} previousFrame - Previous grayscale frame (R32Float, level 0)
   */
  encode(encoder, currentFrame, previousFrame) {
    if (!this.pipeline) return;

    const bindGroup = this.device.createBindGroup({
      label: 'block-match-bind-group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: currentFrame.createView() },
        { binding: 1, resource: previousFrame.createView() },
        { binding: 2, resource: this.flowTexture.createView() },
        { binding: 3, resource: { buffer: this.uniformBuffer } },
      ],
    });

    const blocksX = Math.ceil(this.width / this.blockSize);
    const blocksY = Math.ceil(this.height / this.blockSize);

    const pass = encoder.beginComputePass({ label: 'block-match-pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(blocksX / 8), Math.ceil(blocksY / 8));
    pass.end();
  }

  /**
   * Encode a copy from the flow texture to the readback buffer.
   * Call this after encode() in the same command encoder.
   * @param {GPUCommandEncoder} encoder
   */
  encodeCopyForReadback(encoder) {
    encoder.copyTextureToBuffer(
      { texture: this.flowTexture },
      {
        buffer: this.readbackBuffer,
        bytesPerRow: this.width * 8, // rg32float = 8 bytes
        rowsPerImage: this.height,
      },
      [this.width, this.height]
    );
  }

  /**
   * Read back the flow data from GPU.
   * Must be called after the command buffer with encodeCopyForReadback has been submitted.
   * @returns {Promise<Float32Array>} Flow data as [vx, vy, vx, vy, ...] per pixel
   */
  async readFlowData() {
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
    this.readbackBuffer.unmap();
    return data;
  }

  destroy() {
    this.flowTexture.destroy();
    this.readbackBuffer.destroy();
    this.uniformBuffer.destroy();
  }
}
