// Block-matching optical flow compute shader
// Each invocation processes one 8×8 block.
// Searches ±searchRadius pixels in the previous frame using SAD.
// Output: rg32float texture with per-block motion vector (vx, vy).

struct Params {
  width:        u32,  // Working resolution width
  height:       u32,  // Working resolution height
  blockSize:    u32,  // Block size (8)
  searchRadius: u32,  // Search radius (8)
};

@group(0) @binding(0) var currFrame: texture_2d<f32>;
@group(0) @binding(1) var prevFrame: texture_2d<f32>;
@group(0) @binding(2) var flowOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  // Each invocation = one block
  let blocksX = (params.width + params.blockSize - 1u) / params.blockSize;
  let blocksY = (params.height + params.blockSize - 1u) / params.blockSize;

  if (gid.x >= blocksX || gid.y >= blocksY) {
    return;
  }

  let blockX = gid.x * params.blockSize;
  let blockY = gid.y * params.blockSize;

  var bestSAD = 1e10;
  var bestDx = 0;
  var bestDy = 0;

  let sr = i32(params.searchRadius);

  // Exhaustive search over the search window
  for (var dy = -sr; dy <= sr; dy++) {
    for (var dx = -sr; dx <= sr; dx++) {
      var sad = 0.0;

      // Compute SAD for this candidate displacement
      for (var by = 0u; by < params.blockSize; by++) {
        for (var bx = 0u; bx < params.blockSize; bx++) {
          let cx = blockX + bx;
          let cy = blockY + by;

          if (cx >= params.width || cy >= params.height) {
            continue;
          }

          let px = i32(cx) + dx;
          let py = i32(cy) + dy;

          // Clamp to image bounds
          let cpx = u32(clamp(px, 0, i32(params.width) - 1));
          let cpy = u32(clamp(py, 0, i32(params.height) - 1));

          let currVal = textureLoad(currFrame, vec2u(cx, cy), 0).r;
          let prevVal = textureLoad(prevFrame, vec2u(cpx, cpy), 0).r;

          sad += abs(currVal - prevVal);
        }
      }

      if (sad < bestSAD) {
        bestSAD = sad;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  // Write the motion vector for each pixel in this block
  for (var by = 0u; by < params.blockSize; by++) {
    for (var bx = 0u; bx < params.blockSize; bx++) {
      let wx = blockX + bx;
      let wy = blockY + by;
      if (wx < params.width && wy < params.height) {
        textureStore(flowOut, vec2u(wx, wy), vec4f(f32(bestDx), f32(bestDy), 0.0, 0.0));
      }
    }
  }
}
