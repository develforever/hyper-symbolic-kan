/**
 * Hyper-Symbolic KAN Pure TypeScript Engine
 * Umożliwia natychmiastową ewaluację i adaptację online (Streaming ALS) w przeglądarce.
 */

export interface KANModelData {
  type: string;
  spatial_dim: number;
  rank: number;
  degree: number;
  lambdas: number[];
  factors: number[][][]; // (D, R, K+1)
}

export class KanEvaluator {
  public spatialDim: number;
  public rank: number;
  public degree: number;
  public lambdas: Float64Array;
  public factors: Float64Array[]; // Array of size D, each having R * (K+1) elements

  constructor(data: KANModelData) {
    this.spatialDim = data.spatial_dim;
    this.rank = data.rank;
    this.degree = data.degree;
    this.lambdas = new Float64Array(data.lambdas);
    
    const K1 = this.degree + 1;
    this.factors = [];
    for (let d = 0; d < this.spatialDim; d++) {
      const flatFactor = new Float64Array(this.rank * K1);
      for (let r = 0; r < this.rank; r++) {
        for (let k = 0; k < K1; k++) {
          flatFactor[r * K1 + k] = data.factors[d][r][k];
        }
      }
      this.factors.push(flatFactor);
    }
  }

  /**
   * Wyznacza wielomiany Czebyszewa T_k(x) oraz pochodne dT_k/dx
   */
  public computeChebyshev(x: number, T_out: Float64Array, dT_out?: Float64Array): void {
    const xClamped = Math.max(-1.0, Math.min(1.0, x));
    const K = this.degree;

    T_out[0] = 1.0;
    if (dT_out) dT_out[0] = 0.0;

    if (K >= 1) {
      T_out[1] = xClamped;
      if (dT_out) dT_out[1] = 1.0;
    }

    for (let k = 2; k <= K; k++) {
      T_out[k] = 2.0 * xClamped * T_out[k - 1] - T_out[k - 2];
      if (dT_out) {
        dT_out[k] = 2.0 * T_out[k - 1] + 2.0 * xClamped * dT_out[k - 1] - dT_out[k - 2];
      }
    }
  }

  /**
   * Szybka ewaluacja punktu w przestrzeni 3D
   */
  public evaluate3D(x: number, y: number, z: number): number {
    const K1 = this.degree + 1;
    const Tx = new Float64Array(K1);
    const Ty = new Float64Array(K1);
    const Tz = new Float64Array(K1);

    this.computeChebyshev(x, Tx);
    this.computeChebyshev(y, Ty);
    this.computeChebyshev(z, Tz);

    let total = 0.0;
    const fx = this.factors[0];
    const fy = this.factors[1];
    const fz = this.factors[2];

    for (let r = 0; r < this.rank; r++) {
      const offset = r * K1;
      let phiX = 0.0, phiY = 0.0, phiZ = 0.0;
      for (let k = 0; k < K1; k++) {
        phiX += fx[offset + k] * Tx[k];
        phiY += fy[offset + k] * Ty[k];
        phiZ += fz[offset + k] * Tz[k];
      }
      total += this.lambdas[r] * phiX * phiY * phiZ;
    }

    return total;
  }

  /**
   * Analityczny gradient 3D \nabla f(x, y, z) bez różnic skończonych
   */
  public gradient3D(x: number, y: number, z: number, gradOut: Float32Array | Float64Array): void {
    const K1 = this.degree + 1;
    const Tx = new Float64Array(K1);
    const Ty = new Float64Array(K1);
    const Tz = new Float64Array(K1);
    const dTx = new Float64Array(K1);
    const dTy = new Float64Array(K1);
    const dTz = new Float64Array(K1);

    this.computeChebyshev(x, Tx, dTx);
    this.computeChebyshev(y, Ty, dTy);
    this.computeChebyshev(z, Tz, dTz);

    let gx = 0.0, gy = 0.0, gz = 0.0;
    const fx = this.factors[0];
    const fy = this.factors[1];
    const fz = this.factors[2];

    for (let r = 0; r < this.rank; r++) {
      const offset = r * K1;
      let phiX = 0.0, phiY = 0.0, phiZ = 0.0;
      let dphiX = 0.0, dphiY = 0.0, dphiZ = 0.0;

      for (let k = 0; k < K1; k++) {
        phiX += fx[offset + k] * Tx[k];
        dphiX += fx[offset + k] * dTx[k];

        phiY += fy[offset + k] * Ty[k];
        dphiY += fy[offset + k] * dTy[k];

        phiZ += fz[offset + k] * Tz[k];
        dphiZ += fz[offset + k] * dTz[k];
      }

      const lam = this.lambdas[r];
      gx += lam * dphiX * phiY * phiZ;
      gy += lam * phiX * dphiY * phiZ;
      gz += lam * phiX * phiY * dphiZ;
    }

    gradOut[0] = gx;
    gradOut[1] = gy;
    gradOut[2] = gz;
  }

  /**
   * Błyskawiczna adaptacja online (Concept Drift Streaming ALS) na GPU/CPU
   * Aktualizuje model w locie przy przesunięciu przeszkody.
   */
  public updateOnlineStreaming(x: number, y: number, z: number, target: number, lr: number = 0.05): void {
    const current = this.evaluate3D(x, y, z);
    const error = target - current;

    const K1 = this.degree + 1;
    const Tx = new Float64Array(K1);
    const Ty = new Float64Array(K1);
    const Tz = new Float64Array(K1);
    this.computeChebyshev(x, Tx);
    this.computeChebyshev(y, Ty);
    this.computeChebyshev(z, Tz);

    const fx = this.factors[0];
    const fy = this.factors[1];
    const fz = this.factors[2];

    for (let r = 0; r < this.rank; r++) {
      const offset = r * K1;
      let phiX = 0.0, phiY = 0.0, phiZ = 0.0;
      for (let k = 0; k < K1; k++) {
        phiX += fx[offset + k] * Tx[k];
        phiY += fy[offset + k] * Ty[k];
        phiZ += fz[offset + k] * Tz[k];
      }

      // Aktualizacja lambda
      this.lambdas[r] += lr * error * (phiX * phiY * phiZ);
    }
  }
}
