// Block-matching optical flow compute shader
// Each invocation processes one 8×8 block.
// Searches ±searchRadius pixels in the previous frame using SAD.
// Output: rg32float texture with per-block motion vector (vx, vy).

struct Params {
  width:        u32,  // Working resolution width
  height:       u32,  // Working resolution height
  blockSize:    u32,  // Block size (8)
  searchRadius: u32,  // Search radius (12)
};

@group(0) @binding(0) var currFrame: texture_2d<f32>;
@group(0) @binding(1) var prevFrame: texture_2d<f32>;
@group(0) @binding(2) var flowOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let blocksX = (params.width + params.blockSize - 1u) / params.blockSize;
  let blocksY = (params.height + params.blockSize - 1u) / params.blockSize;

  if (gid.x >= blocksX || gid.y >= blocksY) {
    return;
  }

  let blockX = gid.x * params.blockSize;
  let blockY = gid.y * params.blockSize;

  // 1. Measure block contrast and compute zero-displacement SAD
  var minLuma = 1.0;
  var maxLuma = 0.0;
  var sadZero = 0.0;

  for (var by = 0u; by < params.blockSize; by++) {
    for (var bx = 0u; bx < params.blockSize; bx++) {
      let cx = blockX + bx;
      let cy = blockY + by;
      if (cx < params.width && cy < params.height) {
        let currVal = textureLoad(currFrame, vec2u(cx, cy), 0).r;
        let prevVal = textureLoad(prevFrame, vec2u(cx, cy), 0).r;
        minLuma = min(minLuma, currVal);
        maxLuma = max(maxLuma, currVal);
        sadZero += abs(currVal - prevVal);
      }
    }
  }

  // 2. Search over candidate displacements
  var bestCost = sadZero;
  var bestDx = 0;
  var bestDy = 0;

  let sr = i32(params.searchRadius);

  for (var dy = -sr; dy <= sr; dy++) {
    for (var dx = -sr; dx <= sr; dx++) {
      if (dx == 0 && dy == 0) {
        continue;
      }

      var sad = 0.0;

      for (var by = 0u; by < params.blockSize; by++) {
        for (var bx = 0u; bx < params.blockSize; bx++) {
          let cx = blockX + bx;
          let cy = blockY + by;

          if (cx >= params.width || cy >= params.height) {
            continue;
          }

          let px = i32(cx) + dx;
          let py = i32(cy) + dy;

          let cpx = u32(clamp(px, 0, i32(params.width) - 1));
          let cpy = u32(clamp(py, 0, i32(params.height) - 1));

          let currVal = textureLoad(currFrame, vec2u(cx, cy), 0).r;
          let prevVal = textureLoad(prevFrame, vec2u(cpx, cpy), 0).r;

          sad += abs(currVal - prevVal);
        }
      }

      // Smooth distance penalty
      let distCost = f32(dx * dx + dy * dy) * 0.0004;
      let cost = sad + distCost;

      if (cost < bestCost) {
        bestCost = cost;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  // If the best displacement is not at least 5% better than zero displacement,
  // or if the block has virtually zero contrast, stay at (0, 0)
  let contrast = maxLuma - minLuma;
  if (bestCost >= sadZero * 0.95 || contrast < 0.005) {
    bestDx = 0;
    bestDy = 0;
  }

  // 3. Write flow vector to all pixels in this block
  let outVec = vec4f(f32(bestDx), f32(bestDy), 0.0, 0.0);
  for (var by = 0u; by < params.blockSize; by++) {
    for (var bx = 0u; bx < params.blockSize; bx++) {
      let wx = blockX + bx;
      let wy = blockY + by;
      if (wx < params.width && wy < params.height) {
        textureStore(flowOut, vec2u(wx, wy), outVec);
      }
    }
  }
}
