/**
 * WebGPU Compute Pipeline for Hyper-Symbolic KAN
 * Zero-copy VRAM evaluation and analytical gradient computation.
 */

import { KAN_COMPUTE_WGSL } from "./shaders.js";
import type { KANModelData } from "./evaluator.js";

export interface ComputeBatchResult {
  values: Float32Array;
  gradients: Float32Array; // [gx0, gy0, gz0, 0, gx1, gy1, gz1, 0, ...]
}

export class WebGPUKanPipeline {
  private device: GPUDevice;
  private computePipeline: GPUComputePipeline | null = null;
  private kanParamsBuffer: GPUBuffer | null = null;
  private uniformsBuffer: GPUBuffer | null = null;

  private pointsBuffer: GPUBuffer | null = null;
  private outValuesBuffer: GPUBuffer | null = null;
  private outGradientsBuffer: GPUBuffer | null = null;
  private stagingValuesBuffer: GPUBuffer | null = null;
  private stagingGradientsBuffer: GPUBuffer | null = null;

  private bindGroup: GPUBindGroup | null = null;
  private currentCapacity: number = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  public static isSupported(): boolean {
    return typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;
  }

  public static async create(): Promise<WebGPUKanPipeline | null> {
    if (!WebGPUKanPipeline.isSupported()) {
      return null;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const pipeline = new WebGPUKanPipeline(device);
    await pipeline.init();
    return pipeline;
  }

  public async init(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Hyper-Symbolic KAN WGSL Shader",
      code: KAN_COMPUTE_WGSL,
    });

    this.computePipeline = await this.device.createComputePipelineAsync({
      label: "Hyper-Symbolic KAN Compute Pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "evaluate_kan_batch",
      },
    });

    // 16-byte aligned Uniform Buffer for KAN parameters (608 bytes)
    this.kanParamsBuffer = this.device.createBuffer({
      label: "KAN Parameters Uniform Buffer",
      size: 608,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Batch Uniforms (16 bytes)
    this.uniformsBuffer = this.device.createBuffer({
      label: "KAN Batch Uniforms Buffer",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Updates KAN weights inside VRAM uniform buffer.
   */
  public updateKanWeights(model: KANModelData): void {
    if (!this.kanParamsBuffer) return;

    const data = new ArrayBuffer(608);
    const u32View = new Uint32Array(data, 0, 4);
    u32View[0] = model.rank;
    u32View[1] = model.degree;
    u32View[2] = model.spatial_dim;
    u32View[3] = 0;

    const f32View = new Float32Array(data);

    // Lambdas offset: byte 16 (float index 4)
    for (let r = 0; r < model.rank && r < 16; r++) {
      f32View[4 + r] = model.lambdas[r];
    }

    // Factors offset: byte 80 (float index 20)
    // factors_x: 20 .. 115 (96 floats)
    // factors_y: 116 .. 211 (96 floats)
    // factors_z: 212 .. 307 (96 floats)
    const K1 = model.degree + 1;
    for (let d = 0; d < 3 && d < model.spatial_dim; d++) {
      const dimOffset = 20 + d * 96;
      for (let r = 0; r < model.rank && r < 16; r++) {
        for (let k = 0; k < K1 && k < 6; k++) {
          f32View[dimOffset + r * 6 + k] = model.factors[d][r][k];
        }
      }
    }

    this.device.queue.writeBuffer(this.kanParamsBuffer, 0, data);
  }

  /**
   * Resizes internal storage buffers to fit numPoints.
   */
  private ensureCapacity(numPoints: number): void {
    if (numPoints <= this.currentCapacity && this.pointsBuffer) {
      return;
    }

    const capacity = Math.max(numPoints, 1024);

    if (this.pointsBuffer) this.pointsBuffer.destroy();
    if (this.outValuesBuffer) this.outValuesBuffer.destroy();
    if (this.outGradientsBuffer) this.outGradientsBuffer.destroy();
    if (this.stagingValuesBuffer) this.stagingValuesBuffer.destroy();
    if (this.stagingGradientsBuffer) this.stagingGradientsBuffer.destroy();

    this.pointsBuffer = this.device.createBuffer({
      label: "KAN Input Points Buffer",
      size: capacity * 16, // vec4<f32>
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.outValuesBuffer = this.device.createBuffer({
      label: "KAN Output Values Buffer",
      size: capacity * 4, // f32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.outGradientsBuffer = this.device.createBuffer({
      label: "KAN Output Gradients Buffer",
      size: capacity * 16, // vec4<f32>
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.stagingValuesBuffer = this.device.createBuffer({
      label: "KAN Staging Values Buffer",
      size: capacity * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.stagingGradientsBuffer = this.device.createBuffer({
      label: "KAN Staging Gradients Buffer",
      size: capacity * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.bindGroup = this.device.createBindGroup({
      label: "KAN Compute Bind Group",
      layout: this.computePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.kanParamsBuffer! } },
        { binding: 1, resource: { buffer: this.pointsBuffer } },
        { binding: 2, resource: { buffer: this.outValuesBuffer } },
        { binding: 3, resource: { buffer: this.outGradientsBuffer } },
        { binding: 4, resource: { buffer: this.uniformsBuffer! } },
      ],
    });

    this.currentCapacity = capacity;
  }

  /**
   * Evaluates a batch of 3D points on the GPU and reads back the computed field values and gradients.
   * points: [x0, y0, z0, x1, y1, z1, ...]
   */
  public async evaluateBatch(points: Float32Array): Promise<ComputeBatchResult> {
    const numPoints = Math.floor(points.length / 3);
    if (numPoints === 0 || !this.computePipeline || !this.kanParamsBuffer || !this.uniformsBuffer) {
      return { values: new Float32Array(0), gradients: new Float32Array(0) };
    }

    this.ensureCapacity(numPoints);

    // Pack input points into vec4<f32>
    const pointsVec4 = new Float32Array(numPoints * 4);
    for (let i = 0; i < numPoints; i++) {
      pointsVec4[i * 4] = points[i * 3];
      pointsVec4[i * 4 + 1] = points[i * 3 + 1];
      pointsVec4[i * 4 + 2] = points[i * 3 + 2];
      pointsVec4[i * 4 + 3] = 1.0;
    }
    this.device.queue.writeBuffer(this.pointsBuffer!, 0, pointsVec4);

    // Write batch uniforms
    const batchUniforms = new Uint32Array([numPoints, 0, 0, 0]);
    this.device.queue.writeBuffer(this.uniformsBuffer, 0, batchUniforms);

    // Encode compute pass
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, this.bindGroup!);
    const workgroups = Math.ceil(numPoints / 64);
    passEncoder.dispatchWorkgroups(workgroups);
    passEncoder.end();

    // Copy to staging buffers
    commandEncoder.copyBufferToBuffer(this.outValuesBuffer!, 0, this.stagingValuesBuffer!, 0, numPoints * 4);
    commandEncoder.copyBufferToBuffer(this.outGradientsBuffer!, 0, this.stagingGradientsBuffer!, 0, numPoints * 16);

    this.device.queue.submit([commandEncoder.finish()]);

    // Map staging buffers
    await Promise.all([
      this.stagingValuesBuffer!.mapAsync(GPUMapMode.READ),
      this.stagingGradientsBuffer!.mapAsync(GPUMapMode.READ),
    ]);

    const valuesCopy = new Float32Array(this.stagingValuesBuffer!.getMappedRange(0, numPoints * 4)).slice();
    const gradientsCopy = new Float32Array(this.stagingGradientsBuffer!.getMappedRange(0, numPoints * 16)).slice();

    this.stagingValuesBuffer!.unmap();
    this.stagingGradientsBuffer!.unmap();

    return { values: valuesCopy, gradients: gradientsCopy };
  }

  /**
   * Dispatches zero-copy compute directly within an existing command encoder.
   */
  public dispatchZeroCopy(
    passEncoder: GPUComputePassEncoder,
    bindGroup: GPUBindGroup,
    numPoints: number
  ): void {
    if (!this.computePipeline) return;
    passEncoder.setPipeline(this.computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numPoints / 64));
  }
}
