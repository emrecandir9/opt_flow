/**
 * Heatmap visualization pass.
 * Renders optical flow as a Thermal Speed Glow or HSV color-wheel heatmap overlaid on the video.
 */

export class HeatmapPass {
  /**
   * @param {GPUDevice} device
   * @param {GPUTextureFormat} canvasFormat - The canvas preferred format
   */
  constructor(device, canvasFormat) {
    this.device = device;
    this.canvasFormat = canvasFormat;
    this.enabled = true;

    // Heatmap params uniform: maxSpeed, opacity, colorMode, _pad (16 bytes)
    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.setParams(6.0, 0.7, 0.0); // Default to Thermal Speed Glow

    this.pipeline = null;
    this._initPipeline();
  }

  async _initPipeline() {
    const shaderCode = await fetch('src/shaders/heatmap.wgsl').then((r) => r.text());
    const shaderModule = this.device.createShaderModule({
      label: 'heatmap-shader',
      code: shaderCode,
    });

    this.pipeline = this.device.createRenderPipeline({
      label: 'heatmap-pipeline',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: this.canvasFormat,
            blend: {
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  /**
   * Update heatmap parameters.
   * @param {number} maxSpeed
   * @param {number} opacity
   * @param {number} colorMode - 0.0 for Thermal Speed Glow, 1.0 for HSV Wheel
   */
  setParams(maxSpeed, opacity, colorMode = 0.0) {
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Float32Array([maxSpeed, opacity, colorMode, 0.0])
    );
  }

  get ready() {
    return this.pipeline !== null;
  }

  /**
   * Encode the heatmap render pass.
   * @param {GPURenderPassEncoder} renderPass - An active render pass
   * @param {GPUTexture} flowTexture - The flow field texture (rg32float)
   */
  encodeInPass(renderPass, flowTexture) {
    if (!this.pipeline || !this.enabled) return;

    const bindGroup = this.device.createBindGroup({
      label: 'heatmap-bind-group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: flowTexture.createView() },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3); // Full-screen triangle
  }

  destroy() {
    this.uniformBuffer.destroy();
  }
}
