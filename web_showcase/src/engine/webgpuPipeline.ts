/**
 * WebGPU Swarm Engine Pipeline
 * Zarządza alokacją buforów VRAM (Zero-Copy), dyspaczem Compute Shaderów oraz bezpośrednim renderowaniem 500k agentów.
 */

import { KanComputeWGSL, KanRenderWGSL } from "../shaders/kanComputeShaders";
import type { KANModelData } from "./kanEvaluator";

export interface SimStateInput {
  numAgents: number;
  flowSpeed: number;
  noiseAmount: number;
  safetyGuardActive: boolean;
  colorScheme: number;
  obstaclePos: [number, number, number];
  dt: number;
  time: number;
  viewProjMatrix: Float32Array; // 16 elements
  resetRequested?: boolean;
}

export interface SwarmTelemetry {
  fps: number;
  computeTimeUs: number;
  gpuThroughputMpts: number;
  violations: number;
  isWebGPU: boolean;
  agentCount: number;
}

export class WebGPUSwarmEngine {
  public maxAgents: number = 500000;
  private canvas: HTMLCanvasElement;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private presentationFormat: GPUTextureFormat = "bgra8unorm";

  // Bufory GPU
  private particleBuffer: GPUBuffer | null = null;
  private kanParamsBuffer: GPUBuffer | null = null;
  private simUniformsBuffer: GPUBuffer | null = null;
  private statsBuffer: GPUBuffer | null = null;
  private statsStagingBuffer: GPUBuffer | null = null;

  // Potoki
  private computePipeline: GPUComputePipeline | null = null;
  private renderPipeline: GPURenderPipeline | null = null;
  private computeBindGroup: GPUBindGroup | null = null;
  private renderBindGroup: GPUBindGroup | null = null;

  // Stany pomocnicze
  private isStagingMapped: boolean = false;
  private lastViolationCount: number = 0;
  private frameCount: number = 0;
  private isInitialized: boolean = false;

  constructor(canvas: HTMLCanvasElement, maxAgents: number = 500000) {
    this.canvas = canvas;
    this.maxAgents = maxAgents;
  }

  public static isSupported(): boolean {
    return typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;
  }

  public async init(initialModel: KANModelData): Promise<boolean> {
    if (!WebGPUSwarmEngine.isSupported()) {
      console.warn("[WebGPU] navigator.gpu nie jest obsługiwany w tej przeglądarce.");
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });

      if (!this.adapter) {
        console.warn("[WebGPU] Nie znaleziono odpowiedniego adaptera GPU.");
        return false;
      }

      this.device = await this.adapter.requestDevice();
      this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;

      if (!this.context) {
        console.warn("[WebGPU] Nie można uzyskać kontekstu 'webgpu'.");
        return false;
      }

      this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format: this.presentationFormat,
        alphaMode: "premultiplied",
      });

      // 1. Alokacja bufora cząstek: 500,000 agentów * 32 bajty = 16 MB
      const particleByteSize = this.maxAgents * 32;
      this.particleBuffer = this.device.createBuffer({
        label: "Particle Storage Buffer",
        size: particleByteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // Inicjalizacja początkowych pozycji w VRAM
      this.initParticleBuffer();

      // 2. Alokacja bufora wag KAN (608 bajtów)
      this.kanParamsBuffer = this.device.createBuffer({
        label: "KAN Parameters Uniform Buffer",
        size: 608,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.updateKanWeights(initialModel);

      // 3. Alokacja bufora parametrów symulacji (112 bajtów)
      this.simUniformsBuffer = this.device.createBuffer({
        label: "Simulation Uniforms Buffer",
        size: 112,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // 4. Alokacja bufora statystyk atomowych (16 bajtów)
      this.statsBuffer = this.device.createBuffer({
        label: "Sim Stats Buffer",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      this.statsStagingBuffer = this.device.createBuffer({
        label: "Sim Stats Staging Buffer",
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      // 5. Kompilacja modułów shaderów
      const computeModule = this.device.createShaderModule({
        label: "KAN Compute Module",
        code: KanComputeWGSL,
      });

      const renderModule = this.device.createShaderModule({
        label: "KAN Render Module",
        code: KanRenderWGSL,
      });

      // 6. Utworzenie potoku Compute
      this.computePipeline = this.device.createComputePipeline({
        label: "KAN Swarm Compute Pipeline",
        layout: "auto",
        compute: {
          module: computeModule,
          entryPoint: "cs_main",
        },
      });

      // 7. Utworzenie potoku Render
      this.renderPipeline = this.device.createRenderPipeline({
        label: "KAN Swarm Render Pipeline",
        layout: "auto",
        vertex: {
          module: renderModule,
          entryPoint: "vs_main",
        },
        fragment: {
          module: renderModule,
          entryPoint: "fs_main",
          targets: [
            {
              format: this.presentationFormat,
              blend: {
                color: {
                  srcFactor: "src-alpha",
                  dstFactor: "one-minus-src-alpha",
                  operation: "add",
                },
                alpha: {
                  srcFactor: "one",
                  dstFactor: "one-minus-src-alpha",
                  operation: "add",
                },
              },
            },
          ],
        },
        primitive: {
          topology: "triangle-list",
          cullMode: "none",
        },
      });

      // 8. Utworzenie BindGroup
      this.computeBindGroup = this.device.createBindGroup({
        label: "Compute BindGroup",
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffer } },
          { binding: 1, resource: { buffer: this.kanParamsBuffer } },
          { binding: 2, resource: { buffer: this.simUniformsBuffer } },
          { binding: 3, resource: { buffer: this.statsBuffer } },
        ],
      });

      this.renderBindGroup = this.device.createBindGroup({
        label: "Render BindGroup",
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffer } },
          { binding: 1, resource: { buffer: this.simUniformsBuffer } },
        ],
      });

      this.isInitialized = true;
      console.log(`[WebGPU] Zero-Copy Swarm Pipeline zainicjalizowany pomyślnie. VRAM: ${(particleByteSize / 1024 / 1024).toFixed(1)} MB`);
      return true;
    } catch (err) {
      console.error("[WebGPU] Błąd inicjalizacji potoku WebGPU:", err);
      return false;
    }
  }

  private initParticleBuffer() {
    if (!this.device || !this.particleBuffer) return;
    const data = new Float32Array(this.maxAgents * 8);
    for (let i = 0; i < this.maxAgents; i++) {
      const idx = i * 8;
      // Pos (x, y, z, speed)
      data[idx + 0] = (Math.random() - 0.5) * 1.6;
      data[idx + 1] = (Math.random() - 0.5) * 1.6;
      data[idx + 2] = (Math.random() - 0.5) * 1.6;
      data[idx + 3] = 0.0;
      // Vel (vx, vy, vz, isViolated)
      data[idx + 4] = (Math.random() - 0.5) * 0.04;
      data[idx + 5] = (Math.random() - 0.5) * 0.04;
      data[idx + 6] = (Math.random() - 0.5) * 0.04;
      data[idx + 7] = 0.0;
    }
    this.device.queue.writeBuffer(this.particleBuffer, 0, data);
  }

  public updateKanWeights(model: KANModelData) {
    if (!this.device || !this.kanParamsBuffer) return;

    // Struktura KanParams w WGSL:
    // lambdas: 2 x vec4<f32> (8 floats) -> offset 0 (32 bajty)
    // factorsX: 12 x vec4<f32> (48 floats) -> offset 32 (192 bajty)
    // factorsY: 12 x vec4<f32> (48 floats) -> offset 224 (192 bajty)
    // factorsZ: 12 x vec4<f32> (48 floats) -> offset 416 (192 bajty)
    // Razem: 608 bajtów (152 floaty)

    const bufferData = new Float32Array(152);
    
    // Lambdas
    for (let r = 0; r < Math.min(8, model.lambdas.length); r++) {
      bufferData[r] = model.lambdas[r];
    }

    const K1 = model.degree + 1;
    // factorsX (offset 8)
    for (let r = 0; r < Math.min(8, model.rank); r++) {
      for (let k = 0; k < Math.min(6, K1); k++) {
        bufferData[8 + r * 6 + k] = model.factors[0][r][k] || 0.0;
      }
    }

    // factorsY (offset 8 + 48 = 56)
    for (let r = 0; r < Math.min(8, model.rank); r++) {
      for (let k = 0; k < Math.min(6, K1); k++) {
        bufferData[56 + r * 6 + k] = model.factors[1][r][k] || 0.0;
      }
    }

    // factorsZ (offset 56 + 48 = 104)
    for (let r = 0; r < Math.min(8, model.rank); r++) {
      for (let k = 0; k < Math.min(6, K1); k++) {
        bufferData[104 + r * 6 + k] = model.factors[2][r][k] || 0.0;
      }
    }

    this.device.queue.writeBuffer(this.kanParamsBuffer, 0, bufferData);
  }

  public render(input: SimStateInput): number {
    if (!this.isInitialized || !this.device || !this.context || !this.computePipeline || !this.renderPipeline) {
      return 0;
    }

    const numAgents = Math.min(input.numAgents, this.maxAgents);

    // 1. Zapis SimUniforms (112 bajtów = 28 floatów/uintów)
    const uniformData = new ArrayBuffer(112);
    const floatView = new Float32Array(uniformData);
    const uintView = new Uint32Array(uniformData);

    // viewProj matrix (16 floats) -> bytes 0..63
    floatView.set(input.viewProjMatrix, 0);

    // obstaclePos (vec4) -> bytes 64..79 (floats 16..19)
    floatView[16] = input.obstaclePos[0];
    floatView[17] = input.obstaclePos[1];
    floatView[18] = input.obstaclePos[2];
    floatView[19] = 0.18; // radius

    // params (vec4) -> bytes 80..95 (floats 20..23)
    floatView[20] = input.flowSpeed;
    floatView[21] = input.noiseAmount;
    floatView[22] = input.dt;
    floatView[23] = input.time;

    // config (vec4<u32>) -> bytes 96..111 (uints 24..27)
    uintView[24] = numAgents;
    uintView[25] = input.safetyGuardActive ? 1 : 0;
    uintView[26] = input.colorScheme;
    uintView[27] = input.resetRequested ? 1 : 0;

    this.device.queue.writeBuffer(this.simUniformsBuffer!, 0, uniformData);

    // 2. Przygotowanie CommandEncoder
    const commandEncoder = this.device.createCommandEncoder();

    // A. Compute Pass (Aktualizacja stanu cząstek w 100% na GPU)
    const computePass = commandEncoder.beginComputePass({
      label: "KAN Swarm Compute Pass",
    });
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.computeBindGroup!);
    const workgroupCount = Math.ceil(numAgents / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // Kopiowanie bufora statystyk do bufora staging tylko gdy bufor nie jest zmapowany (co 15 klatek)
    let shouldReadStats = false;
    this.frameCount++;
    if (
      this.frameCount % 15 === 0 &&
      !this.isStagingMapped &&
      this.statsBuffer &&
      this.statsStagingBuffer &&
      this.statsStagingBuffer.mapState === "unmapped"
    ) {
      commandEncoder.copyBufferToBuffer(this.statsBuffer, 0, this.statsStagingBuffer, 0, 16);
      shouldReadStats = true;
      this.isStagingMapped = true;
    }

    // B. Render Pass (Zero-Copy bezpośrednio z StorageBuffer)
    const textureView = this.context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      label: "KAN Swarm Zero-Copy Render Pass",
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroup!);
    // 6 wierzchołków na quad billboard, numAgents instancji
    renderPass.draw(6, numAgents, 0, 0);
    renderPass.end();

    // Zgłoszenie kolejki do wykonania
    this.device.queue.submit([commandEncoder.finish()]);

    // Asynchroniczny odczyt telemetrii naruszeń po zgłoszeniu do kolejki
    if (shouldReadStats) {
      this.pollTelemetryAsync();
    }

    return this.lastViolationCount;
  }

  private pollTelemetryAsync() {
    if (!this.statsStagingBuffer || this.statsStagingBuffer.mapState !== "unmapped") {
      this.isStagingMapped = false;
      return;
    }

    this.statsStagingBuffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        if (!this.statsStagingBuffer) {
          this.isStagingMapped = false;
          return;
        }
        const copy = this.statsStagingBuffer.getMappedRange(0, 16);
        const stats = new Uint32Array(copy.slice(0));
        this.lastViolationCount = stats[0];
        this.statsStagingBuffer.unmap();
        this.isStagingMapped = false;

        // Reset licznika w VRAM
        if (this.device && this.statsBuffer) {
          const zero = new Uint32Array([0, 0, 0, 0]);
          this.device.queue.writeBuffer(this.statsBuffer, 0, zero);
        }
      })
      .catch(() => {
        this.isStagingMapped = false;
      });
  }

  public resize(width: number, height: number) {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = Math.max(1, width);
      this.canvas.height = Math.max(1, height);
    }
  }

  public destroy() {
    this.isInitialized = false;
    this.particleBuffer?.destroy();
    this.kanParamsBuffer?.destroy();
    this.simUniformsBuffer?.destroy();
    this.statsBuffer?.destroy();
    this.statsStagingBuffer?.destroy();
    this.device?.destroy();
  }
}
