// Separable 5-tap Gaussian blur compute shader
// direction: 0 = horizontal, 1 = vertical
// Kernel weights for sigma ≈ 1.0: [0.0625, 0.25, 0.375, 0.25, 0.0625]

struct Params {
  width:     u32,
  height:    u32,
  direction: u32,  // 0 = horizontal, 1 = vertical
  _pad:      u32,
};

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var dstTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

const KERNEL = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let coord = gid.xy;
  if (coord.x >= params.width || coord.y >= params.height) {
    return;
  }

  var sum = 0.0;

  for (var i = 0; i < 5; i++) {
    let offset = i - 2;
    var sampleCoord: vec2i;

    if (params.direction == 0u) {
      // Horizontal blur
      sampleCoord = vec2i(
        clamp(i32(coord.x) + offset, 0, i32(params.width) - 1),
        i32(coord.y)
      );
    } else {
      // Vertical blur
      sampleCoord = vec2i(
        i32(coord.x),
        clamp(i32(coord.y) + offset, 0, i32(params.height) - 1)
      );
    }

    let val = textureLoad(srcTexture, vec2u(sampleCoord), 0).r;
    sum += val * KERNEL[i];
  }

  textureStore(dstTexture, coord, vec4f(sum, 0.0, 0.0, 1.0));
}
