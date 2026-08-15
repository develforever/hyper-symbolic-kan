import React, { useState, useEffect } from "react";
import { ShieldCheck, Zap, Cpu, Gauge } from "lucide-react";

interface TelemetryOverlayProps {
  rank: number;
  degree: number;
  numAgents: number;
  violations: number;
  safetyGuardActive: boolean;
}

export const TelemetryOverlay: React.FC<TelemetryOverlayProps> = ({
  rank,
  degree,
  numAgents,
  violations,
  safetyGuardActive,
}) => {
  const [fps, setFps] = useState(60);
  const [frameTime, setFrameTime] = useState(16.6);

  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animId: number;

    const loop = () => {
      frameCount++;
      const now = performance.now();
      const delta = now - lastTime;
      if (delta >= 500) {
        const currentFps = Math.round((frameCount * 1000) / delta);
        setFps(currentFps);
        setFrameTime(parseFloat((1000 / Math.max(1, currentFps)).toFixed(1)));
        frameCount = 0;
        lastTime = now;
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, []);

  const totalPointsPerSec = fps * numAgents;
  const violationRate = safetyGuardActive ? 0.0 : (violations / Math.max(1, numAgents)) * 100;

  return (
    <div className="telemetry-panel">
      <div className="telemetry-header">
        <div className="status-dot-container">
          <span className="status-dot"></span>
          <span className="telemetry-title">HYPER-SYMBOLIC KAN ENGINE</span>
        </div>
        <span className="badge-zero-epochs">0 GRADIENT EPOCHS</span>
      </div>

      <div className="telemetry-grid">
        <div className="telemetry-card">
          <div className="telemetry-card-label">
            <Gauge size={14} className="icon-cyan" />
            <span>RENDER FPS</span>
          </div>
          <div className="telemetry-card-val text-cyan">
            {fps} <span className="telemetry-unit">FPS ({frameTime} ms)</span>
          </div>
        </div>

        <div className="telemetry-card">
          <div className="telemetry-card-label">
            <Zap size={14} className="icon-amber" />
            <span>THROUGHPUT</span>
          </div>
          <div className="telemetry-card-val text-amber">
            {(totalPointsPerSec / 1000).toFixed(0)}k{" "}
            <span className="telemetry-unit">evals / sec</span>
          </div>
        </div>

        <div className="telemetry-card">
          <div className="telemetry-card-label">
            <ShieldCheck size={14} className={safetyGuardActive ? "icon-emerald" : "icon-red"} />
            <span>SAFETY GUARD</span>
          </div>
          <div className={`telemetry-card-val ${safetyGuardActive ? "text-emerald" : "text-red"}`}>
            {safetyGuardActive ? "0.00%" : `${violationRate.toFixed(1)}%`}{" "}
            <span className="telemetry-unit">VIOLATIONS</span>
          </div>
        </div>

        <div className="telemetry-card">
          <div className="telemetry-card-label">
            <Cpu size={14} className="icon-blue" />
            <span>COMPLEXITY</span>
          </div>
          <div className="telemetry-card-val text-blue">
            R={rank} · K={degree}{" "}
            <span className="telemetry-unit">(3D Tensor CP)</span>
          </div>
        </div>
      </div>

      <div className="math-proof-box">
        <div className="math-label">ALGEBRAIC KAN FIELD FORMULATION:</div>
        <div className="math-formula">
          f(x, y, z) = &sum;<sub>r=1</sub><sup>{rank}</sup> &lambda;<sub>r</sub> &middot; &phi;<sub>r</sub><sup>(x)</sup> &middot; &phi;<sub>r</sub><sup>(y)</sup> &middot; &phi;<sub>r</sub><sup>(z)</sup>
          &nbsp;&nbsp;|&nbsp;&nbsp; &nabla;f &equiv; EXACT ANALYTIC GRADIENT
        </div>
      </div>
    </div>
  );
};
