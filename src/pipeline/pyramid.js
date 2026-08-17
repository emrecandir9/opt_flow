/**
 * Gaussian image pyramid builder.
 * Builds N-level pyramid from a grayscale R32Float input.
 * Each level: Gaussian blur (horizontal + vertical) then 2x downsample.
 */

import { createGrayscaleTexture } from '../utils/texture-utils.js';

export class PyramidBuilder {
  /**
   * @param {GPUDevice} device
   * @param {number} baseWidth  - Working resolution width (level 0)
   * @param {number} baseHeight - Working resolution height (level 0)
   * @param {number} levels     - Number of pyramid levels (including base)
   */
  constructor(device, baseWidth, baseHeight, levels = 4) {
    this.device = device;
    this.levels = levels;
    this.baseWidth = baseWidth;
    this.baseHeight = baseHeight;

    // Create pyramid textures for current and previous frame
    this.currentPyramid = this._createPyramid('curr-pyramid');
    this.previousPyramid = this._createPyramid('prev-pyramid');

    // Temp texture for separable blur intermediate
    this.blurTemp = [];
    for (let i = 0; i < levels; i++) {
      const w = Math.max(1, baseWidth >> i);
      const h = Math.max(1, baseHeight >> i);
      this.blurTemp.push(createGrayscaleTexture(device, w, h, `blur-temp-L${i}`));
    }

    this.blurPipeline = null;
    this.downsamplePipeline = null;
    this.blurUniforms = [];
    this.downsampleUniforms = [];

    this._initPipelines();
  }

  _createPyramid(label) {
    const textures = [];
    for (let i = 0; i < this.levels; i++) {
      const w = Math.max(1, this.baseWidth >> i);
      const h = Math.max(1, this.baseHeight >> i);
      textures.push(createGrayscaleTexture(this.device, w, h, `${label}-L${i}`));
    }
    return textures;
  }

  async _initPipelines() {
    const [blurCode, downsampleCode] = await Promise.all([
      fetch('src/shaders/gaussian-blur.wgsl').then((r) => r.text()),
      fetch('src/shaders/downsample.wgsl').then((r) => r.text()),
    ]);

    const blurModule = this.device.createShaderModule({
      label: 'gaussian-blur-shader',
      code: blurCode,
    });

    const downsampleModule = this.device.createShaderModule({
      label: 'downsample-shader',
      code: downsampleCode,
    });

    this.blurPipeline = this.device.createComputePipeline({
      label: 'blur-pipeline',
      layout: 'auto',
      compute: { module: blurModule, entryPoint: 'main' },
    });

    this.downsamplePipeline = this.device.createComputePipeline({
      label: 'downsample-pipeline',
      layout: 'auto',
      compute: { module: downsampleModule, entryPoint: 'main' },
    });

    // Create uniform buffers for each level
    for (let i = 0; i < this.levels; i++) {
      const w = Math.max(1, this.baseWidth >> i);
      const h = Math.max(1, this.baseHeight >> i);

      // Blur uniforms: width, height, direction, pad (2 buffers: horizontal & vertical)
      const blurH = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blurH, 0, new Uint32Array([w, h, 0, 0]));

      const blurV = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blurV, 0, new Uint32Array([w, h, 1, 0]));

      this.blurUniforms.push({ horizontal: blurH, vertical: blurV });

      // Downsample uniforms (for the level below, so skip the last level)
      if (i < this.levels - 1) {
        const dw = Math.max(1, this.baseWidth >> (i + 1));
        const dh = Math.max(1, this.baseHeight >> (i + 1));
        const dsUniform = this.device.createBuffer({
          size: 8,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(dsUniform, 0, new Uint32Array([dw, dh]));
        this.downsampleUniforms.push(dsUniform);
      }
    }
  }

  get ready() {
    return this.blurPipeline !== null && this.downsamplePipeline !== null;
  }

  /**
   * Build pyramid from the grayscale input (level 0).
   * Copies input into currentPyramid[0], blurs and downsamples to subsequent levels.
   * @param {GPUCommandEncoder} encoder
   * @param {GPUTexture} grayscaleInput - R32Float working-res texture
   */
  encode(encoder, grayscaleInput) {
    if (!this.ready) return;

    // Copy grayscale input into level 0 of current pyramid
    const w0 = Math.max(1, this.baseWidth);
    const h0 = Math.max(1, this.baseHeight);
    encoder.copyTextureToTexture(
      { texture: grayscaleInput },
      { texture: this.currentPyramid[0] },
      [w0, h0]
    );

    // Build each subsequent level
    for (let i = 0; i < this.levels - 1; i++) {
      const w = Math.max(1, this.baseWidth >> i);
      const h = Math.max(1, this.baseHeight >> i);
      const dw = Math.max(1, this.baseWidth >> (i + 1));
      const dh = Math.max(1, this.baseHeight >> (i + 1));

      // Horizontal blur: currentPyramid[i] → blurTemp[i]
      const blurHGroup = this.device.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.currentPyramid[i].createView() },
          { binding: 1, resource: this.blurTemp[i].createView() },
          { binding: 2, resource: { buffer: this.blurUniforms[i].horizontal } },
        ],
      });

      const passH = encoder.beginComputePass({ label: `blur-h-L${i}` });
      passH.setPipeline(this.blurPipeline);
      passH.setBindGroup(0, blurHGroup);
      passH.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      passH.end();

      // Vertical blur: blurTemp[i] → currentPyramid[i] (reuse as blurred version)
      const blurVGroup = this.device.createBindGroup({
        layout: this.blurPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.blurTemp[i].createView() },
          { binding: 1, resource: this.currentPyramid[i].createView() },
          { binding: 2, resource: { buffer: this.blurUniforms[i].vertical } },
        ],
      });

      const passV = encoder.beginComputePass({ label: `blur-v-L${i}` });
      passV.setPipeline(this.blurPipeline);
      passV.setBindGroup(0, blurVGroup);
      passV.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      passV.end();

      // Downsample: currentPyramid[i] → currentPyramid[i+1]
      const dsGroup = this.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.currentPyramid[i].createView() },
          { binding: 1, resource: this.currentPyramid[i + 1].createView() },
          { binding: 2, resource: { buffer: this.downsampleUniforms[i] } },
        ],
      });

      const passDS = encoder.beginComputePass({ label: `downsample-L${i}→L${i + 1}` });
      passDS.setPipeline(this.downsamplePipeline);
      passDS.setBindGroup(0, dsGroup);
      passDS.dispatchWorkgroups(Math.ceil(dw / 8), Math.ceil(dh / 8));
      passDS.end();
    }
  }

  /**
   * Swap current and previous pyramids (call at end of frame).
   */
  swapPyramids() {
    const temp = this.currentPyramid;
    this.currentPyramid = this.previousPyramid;
    this.previousPyramid = temp;
  }

  destroy() {
    for (const t of this.currentPyramid) t.destroy();
    for (const t of this.previousPyramid) t.destroy();
    for (const t of this.blurTemp) t.destroy();
    for (const u of this.blurUniforms) {
      u.horizontal.destroy();
      u.vertical.destroy();
    }
    for (const u of this.downsampleUniforms) u.destroy();
  }
}
