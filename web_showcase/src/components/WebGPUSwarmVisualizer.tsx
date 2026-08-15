import React, { useRef, useEffect } from "react";
import { WebGPUSwarmEngine } from "../engine/webgpuPipeline";
import type { KANModelData } from "../engine/kanEvaluator";

interface WebGPUSwarmVisualizerProps {
  modelData: KANModelData;
  numAgents: number;
  flowSpeed: number;
  noiseAmount: number;
  safetyGuardActive: boolean;
  colorScheme: number;
  obstaclePos: [number, number, number];
  viewProjMatrixRef: React.MutableRefObject<Float32Array>;
  onViolationCount: (violations: number) => void;
  onWebGPUStatus: (supported: boolean) => void;
}

export const WebGPUSwarmVisualizer: React.FC<WebGPUSwarmVisualizerProps> = ({
  modelData,
  numAgents,
  flowSpeed,
  noiseAmount,
  safetyGuardActive,
  colorScheme,
  obstaclePos,
  viewProjMatrixRef,
  onViolationCount,
  onWebGPUStatus,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<WebGPUSwarmEngine | null>(null);
  const isSupportedRef = useRef<boolean>(true);

  const initialModelRef = useRef<KANModelData>(modelData);

  // Inicjalizacja WebGPU Engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!WebGPUSwarmEngine.isSupported()) {
      isSupportedRef.current = false;
      onWebGPUStatus(false);
      return;
    }

    const engine = new WebGPUSwarmEngine(canvas, 500000);
    let isCancelled = false;

    engine.init(initialModelRef.current).then((success) => {
      if (isCancelled) {
        engine.destroy();
        return;
      }
      if (success) {
        engineRef.current = engine;
        onWebGPUStatus(true);
      } else {
        isSupportedRef.current = false;
        onWebGPUStatus(false);
      }
    });

    return () => {
      isCancelled = true;
      engine.destroy();
      engineRef.current = null;
    };
  }, [onWebGPUStatus]);

  // Aktualizacja wag przy zmianie modelu (Streaming ALS)
  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.updateKanWeights(modelData);
    }
  }, [modelData]);

  // Główna pętla renderowania WebGPU
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();

    const frame = (timeNow: number) => {
      const delta = Math.min((timeNow - lastTime) / 1000, 0.05);
      lastTime = timeNow;

      const engine = engineRef.current;
      const canvas = canvasRef.current;

      if (engine && canvas) {
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.floor(rect.width * dpr);
        const h = Math.floor(rect.height * dpr);

        if (canvas.width !== w || canvas.height !== h) {
          engine.resize(w, h);
        }

        const violations = engine.render({
          numAgents,
          flowSpeed,
          noiseAmount,
          safetyGuardActive,
          colorScheme,
          obstaclePos,
          dt: delta,
          time: timeNow * 0.001,
          viewProjMatrix: viewProjMatrixRef.current,
        });

        onViolationCount(violations);
      }

      animId = requestAnimationFrame(frame);
    };

    animId = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(animId);
  }, [
    numAgents,
    flowSpeed,
    noiseAmount,
    safetyGuardActive,
    colorScheme,
    obstaclePos,
    viewProjMatrixRef,
    onViolationCount,
  ]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 5,
      }}
    />
  );
};
