// Heatmap visualization — vertex + fragment shader
// Renders full-screen quad with bilinear-filtered optical flow.
// Supports Thermal Speed Glow and HSV Direction Wheel color modes with adjustable threshold.

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

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
  colorMode:    f32,  // 0.0 = Thermal Speed Glow, 1.0 = HSV Direction Wheel
  minSpeed:     f32,  // Minimum speed threshold to show motion
};

@group(0) @binding(0) var flowTexture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: HeatmapParams;

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

// Thermal Speed Glow (Cyberpunk Inferno: Transparent -> Deep Cyan -> Vivid Magenta -> Bright Gold)
fn thermalGlow(t: f32) -> vec3f {
  let c0 = vec3f(0.0, 0.8, 1.0);  // Cyan
  let c1 = vec3f(0.85, 0.1, 0.9); // Vivid Magenta
  let c2 = vec3f(1.0, 0.85, 0.2); // Glowing Gold

  if (t < 0.5) {
    return mix(c0, c1, t * 2.0);
  } else {
    return mix(c1, c2, (t - 0.5) * 2.0);
  }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(flowTexture));
  
  // Bilinear interpolation for smooth, silky non-blocky flow contours
  let samplePos = in.uv * dims - 0.5;
  let i0 = vec2u(clamp(vec2i(floor(samplePos)), vec2i(0), vec2i(dims) - 1));
  let i1 = vec2u(clamp(vec2i(ceil(samplePos)), vec2i(0), vec2i(dims) - 1));
  let f = fract(samplePos);

  let f00 = textureLoad(flowTexture, vec2u(i0.x, i0.y), 0).rg;
  let f10 = textureLoad(flowTexture, vec2u(i1.x, i0.y), 0).rg;
  let f01 = textureLoad(flowTexture, vec2u(i0.x, i1.y), 0).rg;
  let f11 = textureLoad(flowTexture, vec2u(i1.x, i1.y), 0).rg;

  let flow = mix(mix(f00, f10, f.x), mix(f01, f11, f.x), f.y);

  let vx = flow.x;
  let vy = flow.y;
  let magnitude = length(vec2f(vx, vy));

  // Sensitivity threshold check: ignore subtle noise below minSpeed
  if (magnitude < params.minSpeed) {
    return vec4f(0.0, 0.0, 0.0, 0.0);
  }

  let span = max(params.maxSpeed - params.minSpeed, 0.1);
  let normMag = clamp((magnitude - params.minSpeed) / span, 0.0, 1.0);
  var rgb: vec3f;

  if (params.colorMode < 0.5) {
    // Mode 0: Thermal Speed Glow
    rgb = thermalGlow(normMag);
  } else {
    // Mode 1: HSV Direction Wheel
    let angle = atan2(vy, vx);
    let hue = (angle / 3.14159265 + 1.0) * 180.0;
    rgb = hsv2rgb(hue, 0.95, 1.0);
  }

  let alpha = clamp((0.35 + 0.65 * normMag) * params.opacity, 0.0, 0.95);

  // Premultiplied alpha output
  return vec4f(rgb * alpha, alpha);
}
