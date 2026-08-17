// Heatmap visualization — vertex + fragment shader
// Renders a full-screen quad sampling the flow texture.
// Maps (vx, vy) → HSV color wheel, alpha ∝ magnitude.

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

// Full-screen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );

  var out: VertexOutput;
  out.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  out.uv = (pos[vertexIndex] + 1.0) * 0.5;
  out.uv.y = 1.0 - out.uv.y; // Flip Y for texture coordinates
  return out;
}

struct HeatmapParams {
  maxSpeed:     f32,  // Maximum expected speed for normalization
  opacity:      f32,  // Overall opacity multiplier
};

@group(0) @binding(0) var flowTexture: texture_2d<f32>;
@group(0) @binding(1) var flowSampler: sampler;
@group(0) @binding(2) var<uniform> params: HeatmapParams;

// HSV to RGB conversion
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3f {
  let c = v * s;
  let hp = h / 60.0;
  let x = c * (1.0 - abs(hp % 2.0 - 1.0));
  let m = v - c;

  var rgb: vec3f;
  if (hp < 1.0) {
    rgb = vec3f(c, x, 0.0);
  } else if (hp < 2.0) {
    rgb = vec3f(x, c, 0.0);
  } else if (hp < 3.0) {
    rgb = vec3f(0.0, c, x);
  } else if (hp < 4.0) {
    rgb = vec3f(0.0, x, c);
  } else if (hp < 5.0) {
    rgb = vec3f(x, 0.0, c);
  } else {
    rgb = vec3f(c, 0.0, x);
  }

  return rgb + m;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let flow = textureSample(flowTexture, flowSampler, in.uv).rg;
  let vx = flow.x;
  let vy = flow.y;

  let magnitude = length(vec2f(vx, vy));
  let angle = atan2(vy, vx); // -π to π

  // Map angle to hue (0-360)
  let hue = (angle / 3.14159265 + 1.0) * 180.0;

  // Normalize magnitude
  let normMag = clamp(magnitude / params.maxSpeed, 0.0, 1.0);

  // HSV: hue from direction, saturation & value from magnitude
  let rgb = hsv2rgb(hue, normMag * 0.8 + 0.2, normMag * 0.9 + 0.1);

  // Alpha based on magnitude — static areas stay transparent
  let alpha = clamp(normMag * 1.5, 0.0, 0.85) * params.opacity;

  return vec4f(rgb, alpha);
}
