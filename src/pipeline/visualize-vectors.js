/**
 * Vector and Streamline visualization overlay.
 * Reads back flow data and renders clean, glowing motion vectors or aerodynamic streamlines.
 */

export class VectorOverlay {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} flowWidth
   * @param {number} flowHeight
   * @param {number} gridSpacing
   */
  constructor(canvas, flowWidth, flowHeight, gridSpacing = 14) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.flowWidth = flowWidth;
    this.flowHeight = flowHeight;
    this.gridSpacing = gridSpacing;
    this.mode = 'streamlines'; // 'streamlines' | 'arrows' | 'off'
    this.enabled = true;
    this.lastFlowData = null;

    // Streamline particles for animated wind-tunnel visualization
    this.streamlines = [];
    this._initStreamlines(80);
  }

  _initStreamlines(count) {
    this.streamlines = [];
    for (let i = 0; i < count; i++) {
      this.streamlines.push({
        x: Math.random() * (this.canvas.width || 960),
        y: Math.random() * (this.canvas.height || 540),
        history: [],
        life: Math.random() * 50 + 20,
        maxLife: 60,
      });
    }
  }

  setFlowData(flowData) {
    this.lastFlowData = flowData;
  }

  draw() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);

    if (!this.enabled || this.mode === 'off' || !this.lastFlowData) return;

    if (this.mode === 'streamlines') {
      this._drawStreamlines();
    } else {
      this._drawArrows();
    }
  }

  _drawStreamlines() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const data = this.lastFlowData;
    const scaleX = this.flowWidth / cw;
    const scaleY = this.flowHeight / ch;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let s of this.streamlines) {
      s.life -= 1;

      // Sample velocity
      const fx = Math.floor(s.x * scaleX);
      const fy = Math.floor(s.y * scaleY);

      let vx = 0, vy = 0;
      if (fx >= 0 && fx < this.flowWidth && fy >= 0 && fy < this.flowHeight) {
        const idx = (fy * this.flowWidth + fx) * 2;
        vx = data[idx] * 2.5;
        vy = data[idx + 1] * 2.5;
      }

      const speed = Math.sqrt(vx * vx + vy * vy);

      if (speed > 0.4) {
        s.x += vx;
        s.y += vy;
        s.history.push({ x: s.x, y: s.y, speed });
        if (s.history.length > 8) s.history.shift();
      } else {
        s.history.shift();
      }

      // Respawn if dead or out of bounds
      if (s.life <= 0 || s.x < 0 || s.x > cw || s.y < 0 || s.y > ch || (speed < 0.2 && s.history.length === 0)) {
        s.x = Math.random() * cw;
        s.y = Math.random() * ch;
        s.history = [];
        s.maxLife = Math.random() * 40 + 30;
        s.life = s.maxLife;
      }

      // Draw smooth curving ribbon
      if (s.history.length >= 2) {
        for (let i = 1; i < s.history.length; i++) {
          const p0 = s.history[i - 1];
          const p1 = s.history[i];
          const t = i / s.history.length;

          const angle = Math.atan2(p1.y - p0.y, p1.x - p0.x);
          const hue = ((angle / Math.PI + 1) * 180) % 360;

          ctx.lineWidth = t * 3.0 + 1.0;
          ctx.strokeStyle = `hsla(${hue}, 95%, 65%, ${t * 0.85})`;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }
      }
    }

    ctx.restore();
  }

  _drawArrows() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const data = this.lastFlowData;
    const scaleX = cw / this.flowWidth;
    const scaleY = ch / this.flowHeight;
    const arrowScale = Math.max(scaleX, scaleY) * 2.2;
    const minMagnitude = 0.5;

    ctx.lineWidth = 2.0;
    ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 4;

    for (let y = 6; y < this.flowHeight; y += this.gridSpacing) {
      for (let x = 6; x < this.flowWidth; x += this.gridSpacing) {
        const idx = (y * this.flowWidth + x) * 2;
        const vx = data[idx];
        const vy = data[idx + 1];

        const magnitude = Math.sqrt(vx * vx + vy * vy);
        if (magnitude < minMagnitude) continue;

        const px = (x + 0.5) * scaleX;
        const py = (y + 0.5) * scaleY;
        const ex = px + vx * arrowScale;
        const ey = py + vy * arrowScale;

        const angle = Math.atan2(vy, vx);
        const hue = ((angle / Math.PI + 1) * 180) % 360;
        const normMag = Math.min(magnitude / 8.0, 1.0);

        const color = `hsl(${hue}, 95%, ${60 + normMag * 20}%)`;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        const headLen = Math.max(6, 6 * (1 + normMag * 0.5));
        const headAngle = Math.atan2(ey - py, ex - px);

        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(headAngle - 0.45), ey - headLen * Math.sin(headAngle - 0.45));
        ctx.lineTo(ex - headLen * Math.cos(headAngle + 0.45), ey - headLen * Math.sin(headAngle + 0.45));
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this._initStreamlines(80);
  }

  destroy() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
