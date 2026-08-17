// Grayscale conversion + downsample compute shader
// Reads rgba8unorm source at full camera resolution,
// writes r32float grayscale at working resolution.

struct Params {
  srcWidth:  u32,
  srcHeight: u32,
  dstWidth:  u32,
  dstHeight: u32,
};

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var dstTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dstCoord = gid.xy;
  if (dstCoord.x >= params.dstWidth || dstCoord.y >= params.dstHeight) {
    return;
  }

  // Map destination pixel to source coordinates (nearest-neighbor downsample)
  let scaleX = f32(params.srcWidth) / f32(params.dstWidth);
  let scaleY = f32(params.srcHeight) / f32(params.dstHeight);

  // 2x2 box filter for anti-aliased downsample
  let srcX = f32(dstCoord.x) * scaleX;
  let srcY = f32(dstCoord.y) * scaleY;

  let x0 = u32(srcX);
  let y0 = u32(srcY);
  let x1 = min(x0 + 1u, params.srcWidth - 1u);
  let y1 = min(y0 + 1u, params.srcHeight - 1u);

  let p00 = textureLoad(srcTexture, vec2u(x0, y0), 0);
  let p10 = textureLoad(srcTexture, vec2u(x1, y0), 0);
  let p01 = textureLoad(srcTexture, vec2u(x0, y1), 0);
  let p11 = textureLoad(srcTexture, vec2u(x1, y1), 0);

  // Bilinear interpolation weights
  let fx = fract(srcX);
  let fy = fract(srcY);

  let rgb = mix(mix(p00.rgb, p10.rgb, fx), mix(p01.rgb, p11.rgb, fx), fy);

  // BT.601 luma conversion
  let luma = dot(rgb, vec3f(0.299, 0.587, 0.114));

  textureStore(dstTexture, dstCoord, vec4f(luma, 0.0, 0.0, 1.0));
}
