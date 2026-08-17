/**
 * Interactive Particle System driven by Optical Flow.
 * 
 * Particles float over the video feed like luminous embers/dust.
 * Motion vectors act as a fluid velocity field (wind / force field),
 * accelerating particles in the direction of moving hands, gestures, or objects.
 */

export class ParticleSystem {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} flowWidth
   * @param {number} flowHeight
   * @param {number} particleCount
   */
  constructor(canvas, flowWidth, flowHeight, particleCount = 3500) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.flowWidth = flowWidth;
    this.flowHeight = flowHeight;
    this.count = particleCount;
    this.enabled = true;

    // Particle state arrays (SoA layout for cache efficiency)
    this.x = new Float32Array(particleCount);
    this.y = new Float32Array(particleCount);
    this.vx = new Float32Array(particleCount);
    this.vy = new Float32Array(particleCount);
    this.life = new Float32Array(particleCount);
    this.maxLife = new Float32Array(particleCount);
    this.size = new Float32Array(particleCount);
    this.hue = new Float32Array(particleCount);

    this.lastFlowData = null;
    this._initParticles();
  }

  _initParticles() {
    const w = this.canvas.width || 960;
    const h = this.canvas.height || 540;

    for (let i = 0; i < this.count; i++) {
      this._spawnParticle(i, w, h, true);
    }
  }

  _spawnParticle(i, w, h, randomLife = false) {
    this.x[i] = Math.random() * w;
    this.y[i] = Math.random() * h;
    // Ambient gentle drift
    this.vx[i] = (Math.random() - 0.5) * 0.4;
    this.vy[i] = (Math.random() - 0.5) * 0.4;

    const maxL = 80 + Math.random() * 120;
    this.maxLife[i] = maxL;
    this.life[i] = randomLife ? Math.random() * maxL : maxL;
    this.size[i] = 1.2 + Math.random() * 2.2;
    // Neon electric color palette: Cyan (180), Violet (270), Magenta (320), Gold (45)
    const hues = [185, 205, 275, 310, 45];
    this.hue[i] = hues[Math.floor(Math.random() * hues.length)] + (Math.random() - 0.5) * 20;
  }

  setFlowData(flowData) {
    this.lastFlowData = flowData;
  }

  /**
   * Update particle physics based on optical flow velocity field.
   * @param {number} dt
   */
  update(dt = 1.0) {
    if (!this.enabled) return;

    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const flowData = this.lastFlowData;
    const fw = this.flowWidth;
    const fh = this.flowHeight;
    const scaleX = fw / cw;
    const scaleY = fh / ch;

    for (let i = 0; i < this.count; i++) {
      this.life[i] -= dt;

      if (this.life[i] <= 0) {
        this._spawnParticle(i, cw, ch, false);
        continue;
      }

      // Sample flow vector at particle position
      const px = this.x[i];
      const py = this.y[i];

      const fx = Math.floor(px * scaleX);
      const fy = Math.floor(py * scaleY);

      if (flowData && fx >= 0 && fx < fw && fy >= 0 && fy < fh) {
        const idx = (fy * fw + fx) * 2;
        const flowVx = flowData[idx];
        const flowVy = flowData[idx + 1];

        const speed = Math.sqrt(flowVx * flowVx + flowVy * flowVy);

        if (speed > 0.3) {
          // Force field impulse: push particle in motion direction
          const impulse = Math.min(speed * 1.6, 12.0);
          this.vx[i] += flowVx * 0.45;
          this.vy[i] += flowVy * 0.45;

          // Shift hue dynamically towards warm fire / gold on high speed
          if (speed > 3.0) {
            this.hue[i] = (this.hue[i] * 0.8 + 45 * 0.2); // Warm gold
          }
        }
      }

      // Apply drag / friction (fluid dampening)
      this.vx[i] *= 0.93;
      this.vy[i] *= 0.93;

      // Position integration
      this.x[i] += this.vx[i];
      this.y[i] += this.vy[i];

      // Screen wrapping
      if (this.x[i] < 0) this.x[i] += cw;
      if (this.x[i] >= cw) this.x[i] -= cw;
      if (this.y[i] < 0) this.y[i] += ch;
      if (this.y[i] >= ch) this.y[i] -= ch;
    }
  }

  /**
   * Render particles with soft glowing trails.
   */
  draw() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    // Semi-transparent clear creates smooth luminous motion trails
    ctx.clearRect(0, 0, cw, ch);

    if (!this.enabled) return;

    ctx.save();
    ctx.globalCompositeOperation = 'lighter'; // Additive blending for gorgeous glow

    for (let i = 0; i < this.count; i++) {
      const lifeRatio = this.life[i] / this.maxLife[i];
      const speed = Math.sqrt(this.vx[i] * this.vx[i] + this.vy[i] * this.vy[i]);

      // Brightness and alpha scale with life and motion velocity
      const alpha = Math.min(1.0, (1.0 - Math.abs(lifeRatio - 0.5) * 2) * 1.5) * (0.35 + Math.min(speed * 0.25, 0.65));
      const r = this.size[i] * (1.0 + Math.min(speed * 0.3, 1.8));

      ctx.fillStyle = `hsla(${this.hue[i]}, 95%, ${60 + Math.min(speed * 8, 30)}%, ${alpha})`;

      ctx.beginPath();
      ctx.arc(this.x[i], this.y[i], r, 0, Math.PI * 2);
      ctx.fill();

      // For fast particles, draw a tapered motion streak
      if (speed > 1.5) {
        ctx.lineWidth = r * 0.8;
        ctx.strokeStyle = ctx.fillStyle;
        ctx.beginPath();
        ctx.moveTo(this.x[i], this.y[i]);
        ctx.lineTo(this.x[i] - this.vx[i] * 3.0, this.y[i] - this.vy[i] * 3.0);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
  }

  destroy() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
