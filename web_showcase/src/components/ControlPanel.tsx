import React from "react";
import { Layers, Shield, RotateCcw, Crosshair, Palette, Sliders, Cpu } from "lucide-react";

interface ControlPanelProps {
  viewMode: "volume" | "swarm" | "dual";
  setViewMode: (mode: "volume" | "swarm" | "dual") => void;
  safetyGuardActive: boolean;
  setSafetyGuardActive: (active: boolean) => void;
  numAgents: number;
  setNumAgents: (n: number) => void;
  isoLevel: number;
  setIsoLevel: (v: number) => void;
  density: number;
  setDensity: (v: number) => void;
  colorScheme: number;
  setColorScheme: (s: number) => void;
  flowSpeed: number;
  setFlowSpeed: (s: number) => void;
  isWebGPU: boolean | null;
  onMoveObstacle: (dx: number, dy: number, dz: number) => void;
  onReset: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  viewMode,
  setViewMode,
  safetyGuardActive,
  setSafetyGuardActive,
  numAgents,
  setNumAgents,
  isoLevel,
  setIsoLevel,
  density,
  setDensity,
  colorScheme,
  setColorScheme,
  flowSpeed,
  setFlowSpeed,
  isWebGPU,
  onMoveObstacle,
  onReset,
}) => {
  return (
    <div className="control-panel">
      {/* Backend Status Card */}
      <div className="panel-section">
        <div className="section-title">
          <Cpu size={16} />
          <span>COMPUTE BACKEND</span>
        </div>
        <div
          className={`guard-toggle-card ${
            isWebGPU !== false ? "guard-active" : "guard-warning"
          }`}
          style={{ cursor: "default" }}
        >
          <div className="guard-info">
            <span className="guard-title">
              {isWebGPU !== false
                ? "WEBGPU WGSL PIPELINE"
                : "WEBGL2 / CPU FALLBACK"}
            </span>
            <span className="guard-desc">
              {isWebGPU !== false
                ? "Zero-Copy VRAM StorageBuffer (Up to 500k Agents @ 60 FPS)"
                : "Single-thread JS Evaluator (Capped at 15k Agents)"}
            </span>
          </div>
          <div
            className={`switch-indicator ${
              isWebGPU !== false ? "switch-on" : "switch-warn"
            }`}
          ></div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-title">
          <Layers size={16} />
          <span>VISUALIZATION MODE</span>
        </div>
        <div className="button-group">
          <button
            className={`btn-toggle ${viewMode === "volume" ? "active" : ""}`}
            onClick={() => setViewMode("volume")}
          >
            SDF Field
          </button>
          <button
            className={`btn-toggle ${viewMode === "swarm" ? "active" : ""}`}
            onClick={() => setViewMode("swarm")}
          >
            Swarm ({numAgents >= 1000 ? `${(numAgents / 1000).toFixed(0)}k` : numAgents})
          </button>
          <button
            className={`btn-toggle ${viewMode === "dual" ? "active" : ""}`}
            onClick={() => setViewMode("dual")}
          >
            Dual View
          </button>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-title">
          <Shield size={16} />
          <span>SAFETY INVARIANTS (MCT-NSE)</span>
        </div>
        <div
          className={`guard-toggle-card ${
            safetyGuardActive ? "guard-active" : "guard-inactive"
          }`}
          onClick={() => setSafetyGuardActive(!safetyGuardActive)}
        >
          <div className="guard-info">
            <span className="guard-title">
              {safetyGuardActive
                ? "CATEGORY GUARD: ENGAGED"
                : "UNFILTERED NEURAL: DANGER"}
            </span>
            <span className="guard-desc">
              {safetyGuardActive
                ? "100% Deterministic Safety Invariant Projection"
                : "No boundary enforcement (Raw neural violations)"}
            </span>
          </div>
          <div
            className={`switch-indicator ${
              safetyGuardActive ? "switch-on" : "switch-off"
            }`}
          ></div>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-title">
          <Sliders size={16} />
          <span>SIMULATION PARAMETERS</span>
        </div>

        <div className="slider-item">
          <div className="slider-label-row">
            <span>Agent Swarm Count</span>
            <span className="slider-val">{numAgents.toLocaleString()}</span>
          </div>
          <input
            type="range"
            min={1000}
            max={isWebGPU === false ? 15000 : 500000}
            step={isWebGPU === false ? 1000 : 5000}
            value={numAgents}
            onChange={(e) => setNumAgents(parseInt(e.target.value))}
            className="custom-slider"
          />

          {/* Szybkie presety agentów */}
          {isWebGPU !== false && (
            <div className="preset-row" style={{ display: "flex", gap: "4px", marginTop: "6px" }}>
              {[10000, 50000, 100000, 250000, 500000].map((preset) => (
                <button
                  key={preset}
                  className={`palette-btn ${numAgents === preset ? "palette-active" : ""}`}
                  style={{ flex: 1, padding: "4px 2px", fontSize: "9.5px", textAlign: "center" }}
                  onClick={() => setNumAgents(preset)}
                >
                  {preset >= 1000 ? `${preset / 1000}k` : preset}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="slider-item">
          <div className="slider-label-row">
            <span>Gradient Flow Velocity</span>
            <span className="slider-val">{flowSpeed.toFixed(1)}x</span>
          </div>
          <input
            type="range"
            min={0.2}
            max={3.0}
            step={0.1}
            value={flowSpeed}
            onChange={(e) => setFlowSpeed(parseFloat(e.target.value))}
            className="custom-slider"
          />
        </div>

        {viewMode !== "swarm" && (
          <>
            <div className="slider-item">
              <div className="slider-label-row">
                <span>SDF Isosurface Threshold</span>
                <span className="slider-val">{isoLevel.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.02}
                max={0.45}
                step={0.01}
                value={isoLevel}
                onChange={(e) => setIsoLevel(parseFloat(e.target.value))}
                className="custom-slider"
              />
            </div>

            <div className="slider-item">
              <div className="slider-label-row">
                <span>Field Raymarch Density</span>
                <span className="slider-val">{density.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={0.5}
                max={5.0}
                step={0.2}
                value={density}
                onChange={(e) => setDensity(parseFloat(e.target.value))}
                className="custom-slider"
              />
            </div>
          </>
        )}
      </div>

      <div className="panel-section">
        <div className="section-title">
          <Crosshair size={16} />
          <span>CONCEPT DRIFT (STREAMING ALS)</span>
        </div>
        <div className="drift-buttons-grid">
          <button className="btn-drift" onClick={() => onMoveObstacle(-0.2, 0.15, 0.0)}>
            Move Threat &larr;
          </button>
          <button className="btn-drift" onClick={() => onMoveObstacle(0.2, -0.15, 0.0)}>
            Move Threat &rarr;
          </button>
          <button className="btn-drift" onClick={() => onMoveObstacle(0.0, 0.25, -0.2)}>
            Move Threat &uarr;
          </button>
          <button className="btn-drift" onClick={() => onReset()}>
            <RotateCcw size={13} /> Reset Scene
          </button>
        </div>
      </div>

      <div className="panel-section">
        <div className="section-title">
          <Palette size={16} />
          <span>COLOR PALETTE</span>
        </div>
        <div className="palette-selector">
          {[
            { id: 0, label: "Plasma" },
            { id: 1, label: "Cyan Steel" },
            { id: 2, label: "Amber Energy" },
            { id: 3, label: "Emerald Matrix" },
          ].map((pal) => (
            <button
              key={pal.id}
              className={`palette-btn ${colorScheme === pal.id ? "palette-active" : ""}`}
              onClick={() => setColorScheme(pal.id)}
            >
              {pal.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
