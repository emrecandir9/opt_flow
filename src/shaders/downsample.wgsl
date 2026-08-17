// 2×2 box downsample compute shader
// Reads r32float source, outputs r32float at half resolution

struct Params {
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

  let srcX = dstCoord.x * 2u;
  let srcY = dstCoord.y * 2u;

  let p00 = textureLoad(srcTexture, vec2u(srcX,     srcY),     0).r;
  let p10 = textureLoad(srcTexture, vec2u(srcX + 1u, srcY),     0).r;
  let p01 = textureLoad(srcTexture, vec2u(srcX,     srcY + 1u), 0).r;
  let p11 = textureLoad(srcTexture, vec2u(srcX + 1u, srcY + 1u), 0).r;

  let avg = (p00 + p10 + p01 + p11) * 0.25;

  textureStore(dstTexture, dstCoord, vec4f(avg, 0.0, 0.0, 1.0));
}
