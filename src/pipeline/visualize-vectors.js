/**
 * Vector arrow visualization overlay.
 * Reads back flow data at a sparse grid and draws arrows on a 2D Canvas.
 */

export class VectorOverlay {
  /**
   * @param {HTMLCanvasElement} canvas - The 2D overlay canvas
   * @param {number} flowWidth - Working resolution width
   * @param {number} flowHeight - Working resolution height
   * @param {number} gridSpacing - Pixels between arrow samples (in working res)
   */
  constructor(canvas, flowWidth, flowHeight, gridSpacing = 12) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.flowWidth = flowWidth;
    this.flowHeight = flowHeight;
    this.gridSpacing = gridSpacing;
    this.enabled = true;
    this.lastFlowData = null;
  }

  /**
   * Update the flow data to visualize.
   * @param {Float32Array} flowData - [vx, vy, ...] per pixel, row-major
   */
  setFlowData(flowData) {
    this.lastFlowData = flowData;
  }

  /**
   * Draw arrows on the canvas.
   */
  draw() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);

    if (!this.enabled || !this.lastFlowData) return;

    const data = this.lastFlowData;
    const scaleX = cw / this.flowWidth;
    const scaleY = ch / this.flowHeight;
    const arrowScale = Math.max(scaleX, scaleY) * 2.2;
    const minMagnitude = 0.4; // Minimum flow magnitude to draw an arrow

    ctx.lineWidth = 2.0;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 4;

    for (let y = 4; y < this.flowHeight; y += this.gridSpacing) {
      for (let x = 4; x < this.flowWidth; x += this.gridSpacing) {
        const idx = (y * this.flowWidth + x) * 2;
        const vx = data[idx];
        const vy = data[idx + 1];

        const magnitude = Math.sqrt(vx * vx + vy * vy);
        if (magnitude < minMagnitude) continue;

        // Position on display canvas
        const px = (x + 0.5) * scaleX;
        const py = (y + 0.5) * scaleY;

        // Arrow endpoint
        const ex = px + vx * arrowScale;
        const ey = py + vy * arrowScale;

        // Color from flow direction (HSV wheel)
        const angle = Math.atan2(vy, vx);
        const hue = ((angle / Math.PI + 1) * 180) % 360;
        const normMag = Math.min(magnitude / 8.0, 1.0);

        const color = `hsl(${hue}, 95%, ${60 + normMag * 20}%)`;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        // Draw arrow line
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Draw arrowhead
        const headLen = Math.max(6, 6 * (1 + normMag * 0.5));
        const headAngle = Math.atan2(ey - py, ex - px);

        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(
          ex - headLen * Math.cos(headAngle - 0.45),
          ey - headLen * Math.sin(headAngle - 0.45)
        );
        ctx.lineTo(
          ex - headLen * Math.cos(headAngle + 0.45),
          ey - headLen * Math.sin(headAngle + 0.45)
        );
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  /**
   * Resize the canvas to match container dimensions.
   * @param {number} width
   * @param {number} height
   */
  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  destroy() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
