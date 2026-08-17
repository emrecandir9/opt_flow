// Composite shader — draws the camera video frame to the canvas
// The heatmap is blended on top via the render pipeline's alpha blending.

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
  out.uv.y = 1.0 - out.uv.y;
  return out;
}

@group(0) @binding(0) var videoTexture: texture_2d<f32>;
@group(0) @binding(1) var videoSampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let color = textureSample(videoTexture, videoSampler, in.uv);
  return vec4f(color.rgb, 1.0);
}
