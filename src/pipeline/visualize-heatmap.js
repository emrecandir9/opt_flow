/**
 * Heatmap visualization pass.
 * Renders optical flow as an HSV color-wheel heatmap overlaid on the video.
 */

export class HeatmapPass {
  /**
   * @param {GPUDevice} device
   * @param {GPUTextureFormat} canvasFormat - The canvas preferred format
   */
  constructor(device, canvasFormat) {
    this.device = device;
    this.canvasFormat = canvasFormat;

    // Heatmap params uniform: maxSpeed, opacity
    this.uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.setParams(10.0, 0.6);

    // Sampler for flow texture
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

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
                srcFactor: 'src-alpha',
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
   */
  setParams(maxSpeed, opacity) {
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Float32Array([maxSpeed, opacity])
    );
  }

  get ready() {
    return this.pipeline !== null;
  }

  /**
   * Encode the heatmap render pass (draws on top of existing canvas content).
   * @param {GPURenderPassEncoder} renderPass - An active render pass
   * @param {GPUTexture} flowTexture - The flow field texture (rg32float)
   */
  encodeInPass(renderPass, flowTexture) {
    if (!this.pipeline) return;

    const bindGroup = this.device.createBindGroup({
      label: 'heatmap-bind-group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: flowTexture.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
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
