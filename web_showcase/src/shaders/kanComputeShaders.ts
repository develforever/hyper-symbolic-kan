/**
 * Hyper-Symbolic KAN: WebGPU WGSL Compute & Render Shaders
 * Realizuje analityczną ewaluację gradientu tensora KAN i integrację cząstek w 100% na GPU (Zero-Copy).
 */

export const KanComputeWGSL = /* wgsl */ `
struct Particle {
    pos: vec4<f32>, // xyz: position, w: speed
    vel: vec4<f32>, // xyz: velocity, w: isViolated
};

struct KanParams {
    lambdas: array<vec4<f32>, 2>,
    factorsX: array<vec4<f32>, 12>,
    factorsY: array<vec4<f32>, 12>,
    factorsZ: array<vec4<f32>, 12>,
};

struct SimUniforms {
    viewProj: mat4x4<f32>,
    obstaclePos: vec4<f32>,
    params: vec4<f32>,  // x: flowSpeed, y: noiseAmount, z: dt, w: time
    config: vec4<u32>,  // x: numAgents, y: safetyGuardActive, z: colorScheme, w: resetFlag
};

struct SimStats {
    violations: atomic<u32>,
    activeAgents: atomic<u32>,
    dummy1: u32,
    dummy2: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> kanParams: KanParams;
@group(0) @binding(2) var<uniform> simUniforms: SimUniforms;
@group(0) @binding(3) var<storage, read_write> simStats: SimStats;

fn getFactorX(r: u32, k: u32) -> f32 {
    let idx = r * 6u + k;
    let v = kanParams.factorsX[idx / 4u];
    let comp = idx % 4u;
    if (comp == 0u) { return v.x; }
    if (comp == 1u) { return v.y; }
    if (comp == 2u) { return v.z; }
    return v.w;
}

fn getFactorY(r: u32, k: u32) -> f32 {
    let idx = r * 6u + k;
    let v = kanParams.factorsY[idx / 4u];
    let comp = idx % 4u;
    if (comp == 0u) { return v.x; }
    if (comp == 1u) { return v.y; }
    if (comp == 2u) { return v.z; }
    return v.w;
}

fn getFactorZ(r: u32, k: u32) -> f32 {
    let idx = r * 6u + k;
    let v = kanParams.factorsZ[idx / 4u];
    let comp = idx % 4u;
    if (comp == 0u) { return v.x; }
    if (comp == 1u) { return v.y; }
    if (comp == 2u) { return v.z; }
    return v.w;
}

fn getLambda(r: u32) -> f32 {
    let v = kanParams.lambdas[r / 4u];
    let comp = r % 4u;
    if (comp == 0u) { return v.x; }
    if (comp == 1u) { return v.y; }
    if (comp == 2u) { return v.z; }
    return v.w;
}

fn evaluateKanGrad(p: vec3<f32>) -> vec3<f32> {
    let c = clamp(p, vec3<f32>(-1.0), vec3<f32>(1.0));
    
    var Tx: array<f32, 6>;
    var Ty: array<f32, 6>;
    var Tz: array<f32, 6>;
    
    var dTx: array<f32, 6>;
    var dTy: array<f32, 6>;
    var dTz: array<f32, 6>;
    
    Tx[0] = 1.0; dTx[0] = 0.0;
    Tx[1] = c.x; dTx[1] = 1.0;
    
    Ty[0] = 1.0; dTy[0] = 0.0;
    Ty[1] = c.y; dTy[1] = 1.0;
    
    Tz[0] = 1.0; dTz[0] = 0.0;
    Tz[1] = c.z; dTz[1] = 1.0;
    
    for (var k: u32 = 1u; k < 5u; k = k + 1u) {
        Tx[k+1u] = 2.0 * c.x * Tx[k] - Tx[k-1u];
        dTx[k+1u] = 2.0 * Tx[k] + 2.0 * c.x * dTx[k] - dTx[k-1u];
        
        Ty[k+1u] = 2.0 * c.y * Ty[k] - Ty[k-1u];
        dTy[k+1u] = 2.0 * Ty[k] + 2.0 * c.y * dTy[k] - dTy[k-1u];
        
        Tz[k+1u] = 2.0 * c.z * Tz[k] - Tz[k-1u];
        dTz[k+1u] = 2.0 * Tz[k] + 2.0 * c.z * dTz[k] - dTz[k-1u];
    }
    
    var grad = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var r: u32 = 0u; r < 8u; r = r + 1u) {
        var phiX: f32 = 0.0; var dphiX: f32 = 0.0;
        var phiY: f32 = 0.0; var dphiY: f32 = 0.0;
        var phiZ: f32 = 0.0; var dphiZ: f32 = 0.0;
        
        for (var k: u32 = 0u; k < 6u; k = k + 1u) {
            let fx = getFactorX(r, k);
            let fy = getFactorY(r, k);
            let fz = getFactorZ(r, k);
            
            phiX += fx * Tx[k];
            dphiX += fx * dTx[k];
            
            phiY += fy * Ty[k];
            dphiY += fy * dTy[k];
            
            phiZ += fz * Tz[k];
            dphiZ += fz * dTz[k];
        }
        
        let lam = getLambda(r);
        grad.x += lam * dphiX * phiY * phiZ;
        grad.y += lam * phiX * dphiY * phiZ;
        grad.z += lam * phiX * phiY * dphiZ;
    }
    
    return grad;
}

fn hash11(p: f32) -> f32 {
    var p3 = fract(vec3<f32>(p * 0.1031, p * 0.1030, p * 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p3_in: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p3_in * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= simUniforms.config.x) {
        return;
    }
    
    var p = particles[index].pos.xyz;
    var v = particles[index].vel.xyz;
    let seed = f32(index) * 0.123 + simUniforms.params.w;
    
    let flowSpeed = simUniforms.params.x;
    let noiseAmount = simUniforms.params.y;
    let dt = min(simUniforms.params.z, 0.05);
    let safetyActive = simUniforms.config.y == 1u;
    let resetFlag = simUniforms.config.w == 1u;
    
    if (resetFlag) {
        let rnd = (hash33(vec3<f32>(f32(index), seed, seed * 1.5)) - 0.5) * 1.6;
        let rndV = (hash33(vec3<f32>(seed * 2.1, f32(index) * 1.3, seed)) - 0.5) * 0.04;
        particles[index].pos = vec4<f32>(rnd, 0.0);
        particles[index].vel = vec4<f32>(rndV, 0.0);
        return;
    }
    
    // 1. Obliczenie analitycznego gradientu pola KAN grad f(p)
    let grad = evaluateKanGrad(p);
    
    // Dynamiczny szum stochastyczny i przyspieszenie gradientowe
    let noise = (hash33(p * 12.0 + vec3<f32>(seed, seed * 1.3, seed * 0.7)) - 0.5) * noiseAmount;
    let acc = -grad * flowSpeed * 1.5 + noise;
    
    // Całkowanie Eulera-Chromera
    v = v * 0.94 + acc * dt;
    var nextP = p + v * (dt * 5.0);
    
    // Geometria ograniczeń i No-Fly Zone
    let BOUND_LIMIT = 0.92;
    let NO_FLY_CENTER = vec3<f32>(-0.35, 0.2, 0.0);
    let NO_FLY_RADIUS = 0.35;
    
    let toNoFly = nextP - NO_FLY_CENTER;
    let distNoFly = length(toNoFly);
    
    let isBoundViolated = abs(nextP.x) > BOUND_LIMIT || abs(nextP.y) > BOUND_LIMIT || abs(nextP.z) > BOUND_LIMIT;
    let isNoFlyViolated = distNoFly < NO_FLY_RADIUS;
    
    if (isBoundViolated || isNoFlyViolated) {
        atomicAdd(&simStats.violations, 1u);
    }
    
    // 2. Kategorialny Guard Bezpieczeństwa (MCT-NSE)
    if (safetyActive) {
        // Projekcja barierowa na granice sześcianu [-0.92, 0.92]^3
        if (nextP.x > BOUND_LIMIT) { nextP.x = BOUND_LIMIT; v.x = -abs(v.x) * 0.5; }
        if (nextP.x < -BOUND_LIMIT) { nextP.x = -BOUND_LIMIT; v.x = abs(v.x) * 0.5; }
        if (nextP.y > BOUND_LIMIT) { nextP.y = BOUND_LIMIT; v.y = -abs(v.y) * 0.5; }
        if (nextP.y < -BOUND_LIMIT) { nextP.y = -BOUND_LIMIT; v.y = abs(v.y) * 0.5; }
        if (nextP.z > BOUND_LIMIT) { nextP.z = BOUND_LIMIT; v.z = -abs(v.z) * 0.5; }
        if (nextP.z < -BOUND_LIMIT) { nextP.z = -BOUND_LIMIT; v.z = abs(v.z) * 0.5; }
        
        // Projekcja barierowa na sferę No-Fly Zone
        if (distNoFly < NO_FLY_RADIUS && distNoFly > 1e-4) {
            let n = normalize(toNoFly);
            nextP = NO_FLY_CENTER + n * NO_FLY_RADIUS;
            let vDotN = dot(v, n);
            v = (v - 1.8 * vDotN * n) * 0.6;
        }
    } else {
        // Bez filtra: cząstki wylatujące za przestrzeń są resetowane losowo
        if (abs(nextP.x) > 1.3 || abs(nextP.y) > 1.3 || abs(nextP.z) > 1.3) {
            nextP = (hash33(vec3<f32>(f32(index), seed, seed * 2.0)) - 0.5) * 1.5;
            v = (hash33(vec3<f32>(seed, f32(index), seed * 3.0)) - 0.5) * 0.04;
        }
    }
    
    var speed = length(v);
    
    // Ciągła dynamika roju: respawn uśpionych cząstek
    if (speed < 0.002 || hash11(seed + f32(index) * 0.01) < 0.003) {
        nextP = (hash33(vec3<f32>(f32(index) * 1.1, seed * 1.2, seed * 0.7)) - 0.5) * 1.6;
        v = (hash33(vec3<f32>(seed * 0.3, f32(index) * 1.7, seed)) - 0.5) * 0.04;
        speed = length(v);
    }
    
    let violationFlag = select(0.0, 1.0, isNoFlyViolated && !safetyActive);
    particles[index].pos = vec4<f32>(nextP, speed);
    particles[index].vel = vec4<f32>(v, violationFlag);
}
`;

export const KanRenderWGSL = /* wgsl */ `
struct Particle {
    pos: vec4<f32>, // xyz: pos, w: speed
    vel: vec4<f32>, // xyz: vel, w: isViolated
};

struct SimUniforms {
    viewProj: mat4x4<f32>,
    obstaclePos: vec4<f32>,
    params: vec4<f32>,  // x: flowSpeed, y: noiseAmount, z: dt, w: time
    config: vec4<u32>,  // x: numAgents, y: safetyGuardActive, z: colorScheme, w: resetFlag
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> simUniforms: SimUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) speed: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) i_idx: u32
) -> VertexOutput {
    var out: VertexOutput;
    let particle = particles[i_idx];
    let pos = particle.pos.xyz;
    let speed = particle.pos.w;
    let isViolated = particle.vel.w > 0.5;
    
    // Quad billboard z 2 trójkątów (6 wierzchołków)
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    let corner = corners[v_idx];
    out.uv = corner;
    out.speed = speed;
    
    // Dynamiczny rozmiar cząstki skalowany prędkością
    let size = clamp(0.010 + speed * 0.035, 0.007, 0.022);
    
    // Ekstrakcja wektorów kamery Right i Up z macierzy ViewProj
    let camRight = vec3<f32>(simUniforms.viewProj[0][0], simUniforms.viewProj[1][0], simUniforms.viewProj[2][0]);
    let camUp    = vec3<f32>(simUniforms.viewProj[0][1], simUniforms.viewProj[1][1], simUniforms.viewProj[2][1]);
    
    let worldPos = pos + (camRight * corner.x + camUp * corner.y) * size;
    out.position = simUniforms.viewProj * vec4<f32>(worldPos, 1.0);
    
    let colorScheme = simUniforms.config.z;
    if (isViolated) {
        out.color = vec4<f32>(1.0, 0.15, 0.25, 0.95);
    } else {
        let t = clamp(speed * 12.0, 0.0, 1.0);
        if (colorScheme == 0u) {
            // Plasma
            let c = mix(vec3<f32>(0.2, 0.1, 0.75), vec3<f32>(0.95, 0.45, 0.1), t) + vec3<f32>(0.1, 0.8, 1.0) * pow(t, 2.0);
            out.color = vec4<f32>(c, 0.9);
        } else if (colorScheme == 1u) {
            // Cyan Steel (High-Tech)
            let c = mix(vec3<f32>(0.05, 0.45, 0.85), vec3<f32>(0.15, 0.95, 1.0), t) + vec3<f32>(0.4, 0.7, 1.0) * pow(t, 3.0);
            out.color = vec4<f32>(c, 0.9);
        } else if (colorScheme == 2u) {
            // Amber Energy
            let c = mix(vec3<f32>(0.75, 0.25, 0.05), vec3<f32>(1.0, 0.85, 0.2), t) + vec3<f32>(1.0, 0.9, 0.5) * pow(t, 3.0);
            out.color = vec4<f32>(c, 0.9);
        } else {
            // Emerald Matrix
            let c = mix(vec3<f32>(0.05, 0.55, 0.3), vec3<f32>(0.25, 1.0, 0.6), t) + vec3<f32>(0.5, 1.0, 0.7) * pow(t, 3.0);
            out.color = vec4<f32>(c, 0.9);
        }
    }
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let distSq = dot(in.uv, in.uv);
    if (distSq > 1.0) {
        discard;
    }
    
    // Miękkie wygaszanie krawędzi cząstki i świecące jądro
    let alpha = smoothstep(1.0, 0.25, distSq);
    let core = exp(-distSq * 3.5) * 0.45;
    let rgb = in.color.rgb + vec3<f32>(core);
    
    return vec4<f32>(rgb, alpha * in.color.a);
}
`;
