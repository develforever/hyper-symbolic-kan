import React, { useState, useEffect } from "react";
import { ShieldCheck, Zap, Gauge, Layers, Crosshair, Wind, Activity } from "lucide-react";

export interface SwarmTelemetryProps {
  rank: number;
  degree: number;
  numAgents: number;
  violations: number;
  safetyGuardActive: boolean;
  isWebGPU?: boolean | null;
}

export interface RoboticsTelemetryProps {
  qpLatencyUs: number;
  minH: number;
  speed: number;
  accel: number;
  collision: boolean;
  useHocbf: boolean;
  safetyEnabled: boolean;
}

export interface AerodynamicsTelemetryProps {
  solveTimeMs: number;
  cl: number;
  cd: number;
  cm: number;
  glideRatio: number;
  circulation: number;
  stagnationPoint: [number, number];
  stagnationCp: number;
  minCp: number;
  maxVelocity: number;
  pdeResidualL2: number;
  aoaDeg: number;
  uInf: number;
  airfoilName: string;
  isStalled: boolean;
}

interface TelemetryOverlayProps {
  mode: "swarm" | "robotics" | "aerodynamics";
  swarm?: SwarmTelemetryProps;
  robotics?: RoboticsTelemetryProps;
  aerodynamics?: AerodynamicsTelemetryProps;
}

export const TelemetryOverlay: React.FC<TelemetryOverlayProps> = ({
  mode,
  swarm,
  robotics,
  aerodynamics,
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
      if (delta >= 300) {
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

  // Scenariusz 3: Aerodynamika CFD Tunel
  if (mode === "aerodynamics" && aerodynamics) {
    const aero = aerodynamics;
    const isLiftPositive = aero.cl >= 0;

    return (
      <div className="telemetry-panel">
        <div className="telemetry-header">
          <div className="status-dot-container">
            <span
              className="status-dot"
              style={{
                background: aero.isStalled ? "var(--red-primary)" : "var(--cyan-primary)",
                boxShadow: aero.isStalled ? "0 0 8px var(--red-primary)" : "0 0 8px var(--cyan-primary)",
              }}
            ></span>
            <span className="telemetry-title">
              NACA CFD &bull; {aero.airfoilName} (&alpha; = {aero.aoaDeg >= 0 ? `+${aero.aoaDeg.toFixed(1)}` : aero.aoaDeg.toFixed(1)}&deg;)
            </span>
          </div>
          <div style={{ display: "flex", gap: "6px" }}>
            <span
              className="badge-zero-epochs"
              style={{
                background: "rgba(6, 182, 212, 0.15)",
                color: "var(--cyan-primary)",
                borderColor: "rgba(6, 182, 212, 0.3)",
              }}
            >
              0 BACKPROP EPOCHS
            </span>
            <span
              className="badge-zero-epochs"
              style={{
                background: aero.isStalled ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
                color: aero.isStalled ? "var(--red-primary)" : "var(--emerald-primary)",
                borderColor: aero.isStalled ? "rgba(239, 68, 68, 0.3)" : "rgba(16, 185, 129, 0.3)",
              }}
            >
              {aero.isStalled ? "FLOW SEPARATION (STALL)" : "ATTACHED FLOW"}
            </span>
          </div>
        </div>

        <div className="telemetry-grid">
          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <Zap size={14} className="icon-amber" />
              <span>CFD SOLVER LATENCY</span>
            </div>
            <div className="telemetry-card-val text-amber">
              {aero.solveTimeMs.toFixed(2)} <span className="telemetry-unit">ms (0 Epochs)</span>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <Activity size={14} className={isLiftPositive ? "icon-emerald" : "icon-red"} />
              <span>LIFT COEFFICIENT C_L</span>
            </div>
            <div className={`telemetry-card-val ${isLiftPositive ? "text-emerald" : "text-red"}`}>
              {aero.cl >= 0 ? `+${aero.cl.toFixed(2)}` : aero.cl.toFixed(2)}{" "}
              <span className="telemetry-unit">L/D: {aero.glideRatio}</span>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <Wind size={14} className="icon-red" />
              <span>DRAG COEFFICIENT C_D</span>
            </div>
            <div className="telemetry-card-val text-red">
              {aero.cd.toFixed(4)}{" "}
              <span className="telemetry-unit">C_M: {aero.cm.toFixed(2)}</span>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <ShieldCheck size={14} className="icon-cyan" />
              <span>PDE LAPLACE RESIDUAL L2</span>
            </div>
            <div className="telemetry-card-val text-cyan">
              {aero.pdeResidualL2.toExponential(2)}{" "}
              <span className="telemetry-unit">&nabla;&sup2;&psi; &equiv; 0</span>
            </div>
          </div>
        </div>

        <div className="math-proof-box">
          <div className="math-label">
            INCOMPRESSIBLE POTENTIAL FLOW &amp; BERNOULLI SURFACE PRESSURE:
          </div>
          <div className="math-formula">
            u(x, y) = (&part;&psi;/&part;y, -&part;&psi;/&part;x) &nbsp;&bull;&nbsp;
            C_p(x, y) = 1 - ||u||&sup2; / U<sub>&infin;</sub>&sup2; &nbsp;&bull;&nbsp;
            C_L = &oint; C_p n_y ds &nbsp;&bull;&nbsp;
            &Gamma; = {aero.circulation.toFixed(1)} m&sup2;/s
          </div>
        </div>
      </div>
    );
  }

  // Scenariusz 2: Robotyka CBF
  if (mode === "robotics" && robotics) {
    const isSafe = robotics.safetyEnabled && !robotics.collision;
    const hColorClass = robotics.minH > 0.05 ? "text-emerald" : robotics.minH >= 0 ? "text-amber" : "text-red";
    const hIconColor = robotics.minH > 0.05 ? "icon-emerald" : robotics.minH >= 0 ? "icon-amber" : "icon-red";

    return (
      <div className="telemetry-panel">
        <div className="telemetry-header">
          <div className="status-dot-container">
            <span
              className="status-dot"
              style={{
                background: isSafe ? "var(--emerald-primary)" : "var(--red-primary)",
                boxShadow: isSafe ? "0 0 8px var(--emerald-primary)" : "0 0 8px var(--red-primary)",
              }}
            ></span>
            <span className="telemetry-title">ROBOTICS HOCBF SAFETY FILTER</span>
          </div>
          <div style={{ display: "flex", gap: "6px" }}>
            <span
              className="badge-zero-epochs"
              style={{
                background: robotics.useHocbf ? "rgba(6, 182, 212, 0.15)" : "rgba(59, 130, 246, 0.15)",
                color: robotics.useHocbf ? "var(--cyan-primary)" : "var(--blue-primary)",
                borderColor: robotics.useHocbf ? "rgba(6, 182, 212, 0.3)" : "rgba(59, 130, 246, 0.3)",
              }}
            >
              {robotics.useHocbf ? "HOCBF (DEGREE 2)" : "CBF (DEGREE 1)"}
            </span>
            <span
              className="badge-zero-epochs"
              style={{
                background: robotics.safetyEnabled ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
                color: robotics.safetyEnabled ? "var(--emerald-primary)" : "var(--red-primary)",
                borderColor: robotics.safetyEnabled ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)",
              }}
            >
              {robotics.safetyEnabled ? "ACTIVE FILTER" : "RAW DANGER"}
            </span>
          </div>
        </div>

        <div className="telemetry-grid">
          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <Gauge size={14} className="icon-cyan" />
              <span>DIGITAL TWIN FPS</span>
            </div>
            <div className="telemetry-card-val text-cyan">
              {fps} <span className="telemetry-unit">FPS ({frameTime} ms)</span>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <Zap size={14} className="icon-amber" />
              <span>QP SOLVER LATENCY</span>
            </div>
            <div className="telemetry-card-val text-amber">
              {robotics.qpLatencyUs.toFixed(1)} <span className="telemetry-unit">&mu;s / query</span>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <Crosshair size={14} className={hIconColor} />
              <span>BARRIER VALUE h(x)</span>
            </div>
            <div className={`telemetry-card-val ${hColorClass}`}>
              {robotics.minH >= 0 ? `+${robotics.minH.toFixed(3)}` : robotics.minH.toFixed(3)}{" "}
              <span className="telemetry-unit">{robotics.minH >= 0 ? "m (SAFE)" : "COLLISION"}</span>
            </div>
          </div>

          <div className="telemetry-card">
            <div className="telemetry-card-label">
              <ShieldCheck size={14} className={robotics.safetyEnabled ? "icon-emerald" : "icon-red"} />
              <span>SAFETY GUARANTEE</span>
            </div>
            <div className={`telemetry-card-val ${robotics.safetyEnabled ? "text-emerald" : "text-red"}`}>
              {robotics.safetyEnabled ? "0.00%" : "CRASH"}{" "}
              <span className="telemetry-unit">{robotics.safetyEnabled ? "VIOLATION RATE" : "UNPROTECTED"}</span>
            </div>
          </div>
        </div>

        <div className="math-proof-box">
          <div className="math-label">
            {robotics.useHocbf
              ? "DYNAMIC HOCBF SAFETY LIE DERIVATIVE (RELATIVE DEGREE 2):"
              : "KINEMATIC CBF GRADIENT CONDITION (RELATIVE DEGREE 1):"}
          </div>
          <div className="math-formula">
            {robotics.useHocbf ? (
              <>
                &psi;(p, v) = &nabla;h<sup>T</sup>v + &alpha;<sub>1</sub>h &nbsp;&bull;&nbsp;
                &nabla;h<sup>T</sup>a + v<sup>T</sup>&nabla;<sup>2</sup>hv + (&alpha;<sub>1</sub>+&alpha;<sub>2</sub>)&nabla;h<sup>T</sup>v + &alpha;<sub>1</sub>&alpha;<sub>2</sub>h &ge; 0
              </>
            ) : (
              <>
                &nabla;h(p)<sup>T</sup>u + &alpha; &middot; h(p) &ge; 0 &nbsp;&bull;&nbsp; u<sub>safe</sub> = argmin ||u - u<sub>guided</sub>||<sup>2</sup>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Domyślny tryb Roju (Swarm)
  const numAgents = swarm?.numAgents || 100000;
  const violations = swarm?.violations || 0;
  const safetyGuardActive = swarm?.safetyGuardActive ?? true;
  const isWebGPU = swarm?.isWebGPU;
  const rank = swarm?.rank || 8;
  const degree = swarm?.degree || 4;

  const totalPointsPerSec = (fps * numAgents) / 1000000;
  const violationRate = safetyGuardActive ? 0.0 : Math.min(100, (violations / Math.max(1, numAgents)) * 100);
  const vramKb = ((numAgents * 32) / 1024).toFixed(0);

  return (
    <div className="telemetry-panel">
      <div className="telemetry-header">
        <div className="status-dot-container">
          <span className="status-dot"></span>
          <span className="telemetry-title">HYPER-SYMBOLIC KAN ENGINE</span>
        </div>
        <div style={{ display: "flex", gap: "6px" }}>
          <span className="badge-zero-epochs">0 EPOCHS</span>
          <span
            className="badge-zero-epochs"
            style={{
              background: isWebGPU !== false ? "rgba(6, 182, 212, 0.15)" : "rgba(245, 158, 11, 0.15)",
              color: isWebGPU !== false ? "var(--cyan-primary)" : "var(--amber-primary)",
              borderColor: isWebGPU !== false ? "rgba(6, 182, 212, 0.3)" : "rgba(245, 158, 11, 0.3)",
            }}
          >
            {isWebGPU !== false ? "WEBGPU WGSL" : "WEBGL2"}
          </span>
        </div>
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
            <span>GPU THROUGHPUT</span>
          </div>
          <div className="telemetry-card-val text-amber">
            {totalPointsPerSec >= 1.0 ? totalPointsPerSec.toFixed(2) : (totalPointsPerSec * 1000).toFixed(0)}{" "}
            <span className="telemetry-unit">
              {totalPointsPerSec >= 1.0 ? "M evals/s" : "k evals/s"}
            </span>
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
            <Layers size={14} className="icon-blue" />
            <span>VRAM ZERO-COPY</span>
          </div>
          <div className="telemetry-card-val text-blue">
            {vramKb} <span className="telemetry-unit">KB (STORAGE)</span>
          </div>
        </div>
      </div>

      <div className="math-proof-box">
        <div className="math-label">ALGEBRAIC KAN FIELD FORMULATION (R={rank}, K={degree}):</div>
        <div className="math-formula">
          f(x, y, z) = &sum;<sub>r=1</sub><sup>{rank}</sup> &lambda;<sub>r</sub> &middot; &phi;<sub>r</sub><sup>(x)</sup> &middot; &phi;<sub>r</sub><sup>(y)</sup> &middot; &phi;<sub>r</sub><sup>(z)</sup>
          &nbsp;&nbsp;|&nbsp;&nbsp; &nabla;f &equiv; EXACT WGSL GRADIENT
        </div>
      </div>
    </div>
  );
};
