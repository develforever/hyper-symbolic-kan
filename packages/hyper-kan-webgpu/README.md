# @hyper-kan/webgpu

> **Standalone High-Performance WebGPU Compute Pipeline & Zero-Dependency TypeScript Engine for Hyper-Symbolic KAN (Continuous Polyadic & Tensor Train Functional Fields).**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![WebGPU Ready](https://img.shields.io/badge/WebGPU-Compute_Pipelines-green.svg)](https://w3.org/TR/webgpu/)
[![TypeScript Strict](https://img.shields.io/badge/TypeScript-5.4+-blue.svg)](https://www.typescriptlang.org/)

---

## 1. Overview

`@hyper-kan/webgpu` is a modular, zero-runtime-dependency TypeScript library providing:
- **Direct WebGPU Compute Kernels (WGSL)**: Parallel batch evaluation of continuous fields $f(\mathbf{x})$ and exact analytical gradients $\nabla f(\mathbf{x})$ directly inside VRAM (`StorageBuffer` Zero-Copy).
- **Standalone CPU / Memory Evaluator**: Highly optimized pure TypeScript implementation of Chebyshev polynomial basis recurrence and CP-KAN tensor contractions.
- **Concept Drift Streaming Updates**: Online real-time weight adaptation (Streaming ALS / SGD) without full retraining.
- **Zero Framework Coupling**: Operates independently of Three.js, React, or DOM UI rendering loops.

---

## 2. Installation

```bash
npm install @hyper-kan/webgpu
```

---

## 3. Quick Start

### 3.1 Pure TypeScript / CPU Evaluation

```typescript
import { KanEvaluator, createDefaultSphereModel } from "@hyper-kan/webgpu";

// 1. Initialize from model weights (or create default)
const modelData = createDefaultSphereModel(0.5, 8, 4);
const evaluator = new KanEvaluator(modelData);

// 2. Evaluate single point in 3D
const val = evaluator.evaluate3D(0.2, -0.4, 0.1);
console.log("Field value:", val);

// 3. Compute exact analytical gradient \nabla f(x, y, z)
const grad = new Float32Array(3);
evaluator.gradient3D(0.2, -0.4, 0.1, grad);
console.log("Analytical gradient:", grad);

// 4. Online streaming adaptation
evaluator.updateOnlineStreaming(0.2, -0.4, 0.1, 0.0 /* target */, 0.05 /* lr */);
```

### 3.2 High-Throughput WebGPU Compute Pipeline

```typescript
import { WebGPUKanPipeline, createDefaultSphereModel } from "@hyper-kan/webgpu";

async function runWebGPU() {
  if (!WebGPUKanPipeline.isSupported()) {
    console.warn("WebGPU is not supported in this browser.");
    return;
  }

  const pipeline = await WebGPUKanPipeline.create();
  if (!pipeline) return;

  const modelData = createDefaultSphereModel();
  pipeline.updateKanWeights(modelData);

  // Batch evaluation of 100,000 points
  const numPoints = 100000;
  const points = new Float32Array(numPoints * 3);
  for (let i = 0; i < points.length; i++) {
    points[i] = (Math.random() - 0.5) * 2.0;
  }

  const result = await pipeline.evaluateBatch(points);
  console.log(`Evaluated ${result.values.length} points on GPU.`);
  console.log("First point value:", result.values[0]);
  console.log("First point gradient:", result.gradients.subarray(0, 3));
}
```

---

## 4. Architecture & Memory Layout

### WebGPU Uniform Buffer Layout (608 bytes, 16-byte aligned)
| Offset (Bytes) | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| `0..15` | `u32[4]` | `rank`, `degree`, `spatial_dim`, `_pad` | Structural configuration |
| `16..79` | `vec4<f32>[4]` | `lambdas[16]` | Component scaling weights $\lambda_r$ |
| `80..271` | `vec4<f32>[24]`| `factors_x[16 * 6]` | Chebyshev weights for dimension $X$ |
| `272..463`| `vec4<f32>[24]`| `factors_y[16 * 6]` | Chebyshev weights for dimension $Y$ |
| `464..655`| `vec4<f32>[24]`| `factors_z[16 * 6]` | Chebyshev weights for dimension $Z$ |

---

## 5. License
MIT License.
