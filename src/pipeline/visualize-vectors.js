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
    const arrowScale = Math.max(scaleX, scaleY) * 1.5;
    const minMagnitude = 0.5; // Minimum flow magnitude to draw an arrow

    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';

    for (let y = 0; y < this.flowHeight; y += this.gridSpacing) {
      for (let x = 0; x < this.flowWidth; x += this.gridSpacing) {
        const idx = (y * this.flowWidth + x) * 2;
        const vx = data[idx];
        const vy = data[idx + 1];

        const magnitude = Math.sqrt(vx * vx + vy * vy);
        if (magnitude < minMagnitude) continue;

        // Position on display canvas
        const px = x * scaleX;
        const py = y * scaleY;

        // Arrow endpoint
        const ex = px + vx * arrowScale;
        const ey = py + vy * arrowScale;

        // Color from flow direction (HSV wheel)
        const angle = Math.atan2(vy, vx);
        const hue = ((angle / Math.PI + 1) * 180) % 360;
        const normMag = Math.min(magnitude / 10, 1);

        ctx.strokeStyle = `hsla(${hue}, ${70 + normMag * 30}%, ${50 + normMag * 20}%, ${0.4 + normMag * 0.5})`;
        ctx.fillStyle = ctx.strokeStyle;

        // Draw arrow line
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Draw arrowhead
        const headLen = 4 * (1 + normMag);
        const headAngle = Math.atan2(ey - py, ex - px);

        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(
          ex - headLen * Math.cos(headAngle - 0.4),
          ey - headLen * Math.sin(headAngle - 0.4)
        );
        ctx.lineTo(
          ex - headLen * Math.cos(headAngle + 0.4),
          ey - headLen * Math.sin(headAngle + 0.4)
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
