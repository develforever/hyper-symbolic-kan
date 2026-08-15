/**
 * @hyper-kan/webgpu
 * Standalone WebGPU Compute Pipeline and Zero-Dependency TypeScript Evaluator
 * for Continuous Polyadic & Tensor Train Functional Fields.
 */

export * from "./evaluator.js";
export * from "./pipeline.js";
export * from "./shaders.js";

import { KanEvaluator, type KANModelData } from "./evaluator.js";

/**
 * Creates a default spherical KAN obstacle field model for initialization and testing.
 */
export function createDefaultSphereModel(radius: number = 0.5, rank: number = 8, degree: number = 4): KANModelData {
  const K1 = degree + 1;
  const factors: number[][][] = [];

  for (let d = 0; d < 3; d++) {
    const dimFactors: number[][] = [];
    for (let r = 0; r < rank; r++) {
      const row: number[] = new Array(K1).fill(0);
      row[0] = 0.5;
      if (degree >= 2) {
        row[2] = 0.25; // x^2 term in Chebyshev basis: T_2(x) = 2x^2 - 1 => x^2 = (T_2 + 1)/2
      }
      dimFactors.push(row);
    }
    factors.push(dimFactors);
  }

  const lambdas = new Array(rank).fill(1.0 / rank);

  return {
    type: "TDFFNet_CP_KAN",
    spatial_dim: 3,
    rank,
    degree,
    lambdas,
    factors,
  };
}
