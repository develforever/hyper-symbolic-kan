/**
 * WebGPU WGSL Kernels for Hyper-Symbolic KAN
 * Zero-dependency pure compute shaders for batch evaluation and analytical gradients.
 */

export const KAN_COMPUTE_WGSL = /* wgsl */ `
struct KANParams {
  rank: u32,
  degree: u32,
  spatial_dim: u32,
  _padding0: u32,
  lambdas: array<vec4<f32>, 4>, // 16 values
  factors_x: array<vec4<f32>, 24>, // 16 * 6 = 96 floats
  factors_y: array<vec4<f32>, 24>,
  factors_z: array<vec4<f32>, 24>,
};

struct BatchUniforms {
  num_points: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> kan: KANParams;
@group(0) @binding(1) var<storage, read> in_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_gradients: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> uniforms: BatchUniforms;

fn compute_chebyshev_3d(
  x_in: vec3<f32>,
  deg: u32,
  Tx: ptr<function, array<f32, 6>>,
  Ty: ptr<function, array<f32, 6>>,
  Tz: ptr<function, array<f32, 6>>,
  dTx: ptr<function, array<f32, 6>>,
  dTy: ptr<function, array<f32, 6>>,
  dTz: ptr<function, array<f32, 6>>
) {
  let x = clamp(x_in.x, -1.0, 1.0);
  let y = clamp(x_in.y, -1.0, 1.0);
  let z = clamp(x_in.z, -1.0, 1.0);

  (*Tx)[0] = 1.0; (*Ty)[0] = 1.0; (*Tz)[0] = 1.0;
  (*dTx)[0] = 0.0; (*dTy)[0] = 0.0; (*dTz)[0] = 0.0;

  if (deg >= 1u) {
    (*Tx)[1] = x; (*Ty)[1] = y; (*Tz)[1] = z;
    (*dTx)[1] = 1.0; (*dTy)[1] = 1.0; (*dTz)[1] = 1.0;
  }

  for (var k = 1u; k < deg; k = k + 1u) {
    (*Tx)[k + 1u] = 2.0 * x * (*Tx)[k] - (*Tx)[k - 1u];
    (*Ty)[k + 1u] = 2.0 * y * (*Ty)[k] - (*Ty)[k - 1u];
    (*Tz)[k + 1u] = 2.0 * z * (*Tz)[k] - (*Tz)[k - 1u];

    (*dTx)[k + 1u] = 2.0 * (*Tx)[k] + 2.0 * x * (*dTx)[k] - (*dTx)[k - 1u];
    (*dTy)[k + 1u] = 2.0 * (*Ty)[k] + 2.0 * y * (*dTy)[k] - (*dTy)[k - 1u];
    (*dTz)[k + 1u] = 2.0 * (*Tz)[k] + 2.0 * z * (*dTz)[k] - (*dTz)[k - 1u];
  }
}

fn get_lambda(r: u32) -> f32 {
  let vec_idx = r / 4u;
  let comp_idx = r % 4u;
  return kan.lambdas[vec_idx][comp_idx];
}

fn get_factor(dim: u32, r: u32, k: u32) -> f32 {
  let flat_idx = r * 6u + k;
  let vec_idx = flat_idx / 4u;
  let comp_idx = flat_idx % 4u;
  if (dim == 0u) {
    return kan.factors_x[vec_idx][comp_idx];
  } else if (dim == 1u) {
    return kan.factors_y[vec_idx][comp_idx];
  } else {
    return kan.factors_z[vec_idx][comp_idx];
  }
}

@compute @workgroup_size(64)
fn evaluate_kan_batch(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= uniforms.num_points) {
    return;
  }

  let pos = in_points[idx].xyz;
  var Tx: array<f32, 6>;
  var Ty: array<f32, 6>;
  var Tz: array<f32, 6>;
  var dTx: array<f32, 6>;
  var dTy: array<f32, 6>;
  var dTz: array<f32, 6>;

  compute_chebyshev_3d(pos, kan.degree, &Tx, &Ty, &Tz, &dTx, &dTy, &dTz);

  var total_val = 0.0;
  var grad = vec3<f32>(0.0, 0.0, 0.0);

  for (var r = 0u; r < kan.rank; r = r + 1u) {
    var phi_x = 0.0;
    var phi_y = 0.0;
    var phi_z = 0.0;
    var dphi_x = 0.0;
    var dphi_y = 0.0;
    var dphi_z = 0.0;

    for (var k = 0u; k <= kan.degree; k = k + 1u) {
      let fx = get_factor(0u, r, k);
      let fy = get_factor(1u, r, k);
      let fz = get_factor(2u, r, k);

      phi_x += fx * Tx[k];
      dphi_x += fx * dTx[k];

      phi_y += fy * Ty[k];
      dphi_y += fy * dTy[k];

      phi_z += fz * Tz[k];
      dphi_z += fz * dTz[k];
    }

    let lam = get_lambda(r);
    total_val += lam * phi_x * phi_y * phi_z;
    grad.x += lam * dphi_x * phi_y * phi_z;
    grad.y += lam * phi_x * dphi_y * phi_z;
    grad.z += lam * phi_x * phi_y * dphi_z;
  }

  out_values[idx] = total_val;
  out_gradients[idx] = vec4<f32>(grad, 0.0);
}
`;
