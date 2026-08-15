import * as THREE from "three";

export const KanVolumeVertexShader = `
varying vec3 vOrigin;
varying vec3 vDirection;
varying vec3 vLocalPosition;

void main() {
    vLocalPosition = position;
    
    // Obliczanie wektora promienia kamery w lokalnym układzie sześcianu
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vec4 cameraLocal = inverse(modelMatrix) * vec4(cameraPosition, 1.0);
    
    vOrigin = cameraLocal.xyz;
    vDirection = position - cameraLocal.xyz;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const KanVolumeFragmentShader = `
precision highp float;

varying vec3 vOrigin;
varying vec3 vDirection;
varying vec3 vLocalPosition;

// Hiperparametry KAN: R=8, K=5 (K+1 = 6 bazowych wielomianów)
// Rozmiar macierzy wag dla każdego wymiaru: 8 * 6 = 48 floatów
uniform float u_lambdas[8];
uniform float u_factorsX[48];
uniform float u_factorsY[48];
uniform float u_factorsZ[48];

uniform float u_isoLevel;
uniform float u_density;
uniform int u_colorScheme;
uniform float u_time;
uniform vec3 u_lightPos;
uniform bool u_showWireGrid;

// Obliczenie wartości pola KAN w punkcie p in [-1, 1]^3
float evaluateKan(vec3 p) {
    vec3 c = clamp(p, -1.0, 1.0);
    
    // 1. Rekurencja Czebyszewa T_0..T_5 na GPU
    float Tx[6];
    float Ty[6];
    float Tz[6];
    
    Tx[0] = 1.0; Tx[1] = c.x;
    Ty[0] = 1.0; Ty[1] = c.y;
    Tz[0] = 1.0; Tz[1] = c.z;
    
    for (int k = 1; k < 5; k++) {
        Tx[k+1] = 2.0 * c.x * Tx[k] - Tx[k-1];
        Ty[k+1] = 2.0 * c.y * Ty[k] - Ty[k-1];
        Tz[k+1] = 2.0 * c.z * Tz[k] - Tz[k-1];
    }
    
    float total = 0.0;
    
    // 2. Kontrakcja tensorowa CP
    for (int r = 0; r < 8; r++) {
        int offset = r * 6;
        float phiX = 0.0;
        float phiY = 0.0;
        float phiZ = 0.0;
        
        for (int k = 0; k < 6; k++) {
            phiX += u_factorsX[offset + k] * Tx[k];
            phiY += u_factorsY[offset + k] * Ty[k];
            phiZ += u_factorsZ[offset + k] * Tz[k];
        }
        
        total += u_lambdas[r] * phiX * phiY * phiZ;
    }
    
    return total;
}

// Analytical Gradient grad(f) computed inside shader without finite differences
vec3 evaluateKanGradient(vec3 p) {
    vec3 c = clamp(p, -1.0, 1.0);
    
    float Tx[6]; float dTx[6];
    float Ty[6]; float dTy[6];
    float Tz[6]; float dTz[6];
    
    Tx[0] = 1.0; dTx[0] = 0.0;
    Tx[1] = c.x; dTx[1] = 1.0;
    
    Ty[0] = 1.0; dTy[0] = 0.0;
    Ty[1] = c.y; dTy[1] = 1.0;
    
    Tz[0] = 1.0; dTz[0] = 0.0;
    Tz[1] = c.z; dTz[1] = 1.0;
    
    for (int k = 1; k < 5; k++) {
        Tx[k+1] = 2.0 * c.x * Tx[k] - Tx[k-1];
        dTx[k+1] = 2.0 * Tx[k] + 2.0 * c.x * dTx[k] - dTx[k-1];
        
        Ty[k+1] = 2.0 * c.y * Ty[k] - Ty[k-1];
        dTy[k+1] = 2.0 * Ty[k] + 2.0 * c.y * dTy[k] - dTy[k-1];
        
        Tz[k+1] = 2.0 * c.z * Tz[k] - Tz[k-1];
        dTz[k+1] = 2.0 * Tz[k] + 2.0 * c.z * dTz[k] - dTz[k-1];
    }
    
    vec3 grad = vec3(0.0);
    
    for (int r = 0; r < 8; r++) {
        int offset = r * 6;
        float phiX = 0.0; float dphiX = 0.0;
        float phiY = 0.0; float dphiY = 0.0;
        float phiZ = 0.0; float dphiZ = 0.0;
        
        for (int k = 0; k < 6; k++) {
            phiX += u_factorsX[offset + k] * Tx[k];
            dphiX += u_factorsX[offset + k] * dTx[k];
            
            phiY += u_factorsY[offset + k] * Ty[k];
            dphiY += u_factorsY[offset + k] * dTy[k];
            
            phiZ += u_factorsZ[offset + k] * Tz[k];
            dphiZ += u_factorsZ[offset + k] * dTz[k];
        }
        
        float lam = u_lambdas[r];
        grad.x += lam * dphiX * phiY * phiZ;
        grad.y += lam * phiX * dphiY * phiZ;
        grad.z += lam * phiX * phiY * dphiZ;
    }
    
    return grad;
}

// Mapowanie kolorów palety
vec3 getColorMap(float val, int scheme) {
    float t = clamp(val, 0.0, 1.0);
    
    if (scheme == 0) {
        // Plasma / Electric
        return mix(vec3(0.05, 0.1, 0.25), vec3(0.9, 0.4, 0.1), t) + vec3(0.1, 0.8, 0.9) * pow(t, 3.0);
    } else if (scheme == 1) {
        // Cyan / Blue Steel High-Tech
        return mix(vec3(0.02, 0.08, 0.15), vec3(0.0, 0.75, 0.85), t) + vec3(0.9, 0.95, 1.0) * pow(t, 4.0);
    } else if (scheme == 2) {
        // Amber / Gold Energy
        return mix(vec3(0.1, 0.05, 0.02), vec3(0.95, 0.65, 0.1), t) + vec3(1.0, 0.9, 0.7) * pow(t, 3.0);
    } else {
        // Emerald Matrix Green
        return mix(vec3(0.01, 0.08, 0.04), vec3(0.0, 0.9, 0.45), t) + vec3(0.8, 1.0, 0.8) * pow(t, 4.0);
    }
}

// Przecięcie promienia z sześcianem [-1, 1]^3
vec2 intersectAABB(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax) {
    vec3 tMin = (boxMin - ro) / rd;
    vec3 tMax = (boxMax - ro) / rd;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

void main() {
    vec3 ro = vOrigin;
    vec3 rd = normalize(vDirection);
    
    vec2 tBox = intersectAABB(ro, rd, vec3(-1.0), vec3(1.0));
    if (tBox.x > tBox.y || tBox.y < 0.0) {
        discard;
    }
    
    float tStart = max(0.0, tBox.x);
    float tEnd = tBox.y;
    
    // Parametry wolumetrycznego marszu promienia KAN
    int maxSteps = 80;
    float dt = (tEnd - tStart) / float(maxSteps);
    float t = tStart;
    
    vec4 accumulatedColor = vec4(0.0);
    
    for (int i = 0; i < 80; i++) {
        if (t > tEnd || accumulatedColor.a > 0.98) break;
        
        vec3 p = ro + rd * t;
        float val = evaluateKan(p);
        
        if (val > u_isoLevel) {
            float excess = (val - u_isoLevel) * u_density;
            vec3 grad = evaluateKanGradient(p);
            vec3 normal = -normalize(grad + vec3(1e-6));
            
            // Oświetlenie Phonga
            vec3 lightDir = normalize(u_lightPos - p);
            float diff = max(0.2, dot(normal, lightDir));
            float spec = pow(max(0.0, dot(reflect(-lightDir, normal), -rd)), 16.0) * 0.4;
            
            vec3 baseCol = getColorMap((val - u_isoLevel) / (1.5 - u_isoLevel), u_colorScheme);
            vec3 finalRgb = baseCol * diff + vec3(spec);
            
            float alpha = clamp(excess * 0.15, 0.0, 1.0);
            accumulatedColor.rgb += (1.0 - accumulatedColor.a) * finalRgb * alpha;
            accumulatedColor.a += (1.0 - accumulatedColor.a) * alpha;
        }
        
        t += dt;
    }
    
    if (accumulatedColor.a < 0.02) {
        discard;
    }
    
    gl_FragColor = accumulatedColor;
}
`;

export function createKanShaderMaterial(data: {
  lambdas: number[];
  factors: number[][][]; // (D, R, K+1)
  rank: number;
  degree: number;
}) {
  const K1 = data.degree + 1;
  const factorsX = new Float32Array(48);
  const factorsY = new Float32Array(48);
  const factorsZ = new Float32Array(48);
  const lambdas = new Float32Array(8);

  for (let r = 0; r < Math.min(8, data.rank); r++) {
    lambdas[r] = data.lambdas[r] || 1.0;
    for (let k = 0; k < Math.min(6, K1); k++) {
      factorsX[r * 6 + k] = data.factors[0][r][k] || 0.0;
      factorsY[r * 6 + k] = data.factors[1][r][k] || 0.0;
      factorsZ[r * 6 + k] = data.factors[2][r][k] || 0.0;
    }
  }

  return new THREE.ShaderMaterial({
    vertexShader: KanVolumeVertexShader,
    fragmentShader: KanVolumeFragmentShader,
    uniforms: {
      u_lambdas: { value: Array.from(lambdas) },
      u_factorsX: { value: Array.from(factorsX) },
      u_factorsY: { value: Array.from(factorsY) },
      u_factorsZ: { value: Array.from(factorsZ) },
      u_isoLevel: { value: 0.10 },
      u_density: { value: 3.5 },
      u_colorScheme: { value: 1 }, // 1 = Cyan/Steel High-Tech
      u_time: { value: 0.0 },
      u_lightPos: { value: new THREE.Vector3(2.5, 3.5, 2.5) },
      u_showWireGrid: { value: true },
    },
    transparent: true,
    side: THREE.BackSide,
    depthWrite: false,
  });
}
