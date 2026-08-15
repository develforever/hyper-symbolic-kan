import React from "react";
import {
  Layers,
  Shield,
  RotateCcw,
  Crosshair,
  Palette,
  Sliders,
  Cpu,
  Navigation,
  Activity,
  Radio,
  Wind,
  Compass,
  Eye,
} from "lucide-react";
import { type ScenarioType } from "./scenarios/ScenarioSelector";
import { type NACAProfileConfig } from "../engine/aerodynamicsCfdEngine";

interface SwarmControlProps {
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

interface RoboticsControlProps {
  useHocbf: boolean;
  setUseHocbf: (v: boolean) => void;
  safetyEnabled: boolean;
  setSafetyEnabled: (v: boolean) => void;
  alpha: number;
  setAlpha: (v: number) => void;
  alpha1: number;
  setAlpha1: (v: number) => void;
  alpha2: number;
  setAlpha2: (v: number) => void;
  vMax: number;
  setVMax: (v: number) => void;
  aMax: number;
  setAMax: (v: number) => void;
  tangentialGain: number;
  setTangentialGain: (v: number) => void;
  patrolMode: boolean;
  setPatrolMode: (v: boolean) => void;
  onResetDrone: () => void;
}

export interface AerodynamicsControlProps {
  aoaDeg: number;
  setAoaDeg: (v: number) => void;
  uInf: number;
  setUInf: (v: number) => void;
  airfoilConfig: NACAProfileConfig;
  setAirfoilConfig: React.Dispatch<React.SetStateAction<NACAProfileConfig>>;
  showStreamlines: boolean;
  setShowStreamlines: (v: boolean) => void;
  showPressureMap: boolean;
  setShowPressureMap: (v: boolean) => void;
  showVectors: boolean;
  setShowVectors: (v: boolean) => void;
  showSmokeParticles: boolean;
  setShowSmokeParticles: (v: boolean) => void;
  streamlineDensity: number;
  setStreamlineDensity: (v: number) => void;
  onResetTunnel: () => void;
}

interface ControlPanelProps {
  activeScenario: ScenarioType;
  swarm: SwarmControlProps;
  robotics: RoboticsControlProps;
  aerodynamics: AerodynamicsControlProps;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  activeScenario,
  swarm,
  robotics,
  aerodynamics,
}) => {
  // Scenariusz 3: Tunel Aerodynamiczny CFD (0 Epok)
  if (activeScenario === "aerodynamics") {
    const aero = aerodynamics;
    const currentName = `NACA ${Math.round(aero.airfoilConfig.camber * 100)}${Math.round(
      aero.airfoilConfig.camberPos * 10
    )}${Math.round(aero.airfoilConfig.thickness * 100) < 10 ? "0" : ""}${Math.round(
      aero.airfoilConfig.thickness * 100
    )}`;

    return (
      <div className="control-panel">
        {/* Status Solver KAN CFD */}
        <div className="panel-section">
          <div className="section-title">
            <Wind size={16} />
            <span>MESH-FREE CFD SOLVER (0 EPOCHS)</span>
          </div>
          <div
            className="guard-toggle-card guard-active"
            style={{ cursor: "default" }}
          >
            <div className="guard-info">
              <span className="guard-title">EXACT CONFORMAL &amp; KUTTA KAN</span>
              <span className="guard-desc">
                Analytical streamfunction &nabla;&sup2;&psi; = 0 &bull; Solve latency &le; 1.5 ms
              </span>
            </div>
            <div className="switch-indicator switch-on"></div>
          </div>
        </div>

        {/* Presety Profili NACA */}
        <div className="panel-section">
          <div className="section-title">
            <Compass size={16} />
            <span>NACA AIRFOIL PRESETS</span>
          </div>
          <div className="button-group" style={{ gridTemplateColumns: "1fr 1fr" }}>
            {[
              { label: "NACA 0012", m: 0.0, p: 0.0, t: 0.12 },
              { label: "NACA 2412", m: 0.02, p: 0.4, t: 0.12 },
              { label: "NACA 4415", m: 0.04, p: 0.4, t: 0.15 },
              { label: "NACA 0024", m: 0.0, p: 0.0, t: 0.24 },
            ].map((preset) => (
              <button
                key={preset.label}
                className={`btn-toggle ${currentName === preset.label ? "active" : ""}`}
                onClick={() => {
                  aero.setAirfoilConfig((prev) => ({
                    ...prev,
                    camber: preset.m,
                    camberPos: preset.p,
                    thickness: preset.t,
                  }));
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Parametry Aerodynamiczne */}
        <div className="panel-section">
          <div className="section-title">
            <Sliders size={16} />
            <span>FLOW PARAMETERS</span>
          </div>

          {/* Kąt Natarcia AoA */}
          <div className="slider-item">
            <div className="slider-label-row">
              <span>Angle of Attack &alpha;</span>
              <span className="slider-val" style={{ color: aero.aoaDeg > 15 ? "#ef4444" : "var(--cyan-primary)" }}>
                {aero.aoaDeg >= 0 ? `+${aero.aoaDeg.toFixed(1)}` : aero.aoaDeg.toFixed(1)}&deg;
                {aero.aoaDeg > 15 ? " (STALL)" : ""}
              </span>
            </div>
            <input
              type="range"
              min={-15.0}
              max={25.0}
              step={0.5}
              value={aero.aoaDeg}
              onChange={(e) => aero.setAoaDeg(parseFloat(e.target.value))}
              className="custom-slider"
            />
            {/* Szybkie presety AoA */}
            <div className="preset-row" style={{ display: "flex", gap: "4px", marginTop: "6px" }}>
              {[-5, 0, 5, 10, 15, 20].map((deg) => (
                <button
                  key={deg}
                  className={`palette-btn ${aero.aoaDeg === deg ? "palette-active" : ""}`}
                  style={{ flex: 1, padding: "4px 2px", fontSize: "9.5px", textAlign: "center" }}
                  onClick={() => aero.setAoaDeg(deg)}
                >
                  {deg >= 0 ? `+${deg}°` : `${deg}°`}
                </button>
              ))}
            </div>
          </div>

          {/* Prędkość Napływu U_inf */}
          <div className="slider-item">
            <div className="slider-label-row">
              <span>Freestream Speed U<sub>&infin;</sub></span>
              <span className="slider-val">{aero.uInf.toFixed(0)} m/s</span>
            </div>
            <input
              type="range"
              min={5.0}
              max={60.0}
              step={1.0}
              value={aero.uInf}
              onChange={(e) => aero.setUInf(parseFloat(e.target.value))}
              className="custom-slider"
            />
          </div>

          {/* Grubość profilu NACA */}
          <div className="slider-item">
            <div className="slider-label-row">
              <span>Airfoil Thickness (t/c)</span>
              <span className="slider-val">{(aero.airfoilConfig.thickness * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0.06}
              max={0.24}
              step={0.01}
              value={aero.airfoilConfig.thickness}
              onChange={(e) =>
                aero.setAirfoilConfig((prev) => ({
                  ...prev,
                  thickness: parseFloat(e.target.value),
                }))
              }
              className="custom-slider"
            />
          </div>

          {/* Ugięcie profilu Camber */}
          <div className="slider-item">
            <div className="slider-label-row">
              <span>Max Camber (m/c)</span>
              <span className="slider-val">{(aero.airfoilConfig.camber * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={0.08}
              step={0.01}
              value={aero.airfoilConfig.camber}
              onChange={(e) =>
                aero.setAirfoilConfig((prev) => ({
                  ...prev,
                  camber: parseFloat(e.target.value),
                  camberPos: prev.camberPos || 0.4,
                }))
              }
              className="custom-slider"
            />
          </div>

          {/* Gęstość linii prądu */}
          <div className="slider-item">
            <div className="slider-label-row">
              <span>Streamlines Rake Count</span>
              <span className="slider-val">{aero.streamlineDensity} lines</span>
            </div>
            <input
              type="range"
              min={12}
              max={40}
              step={2}
              value={aero.streamlineDensity}
              onChange={(e) => aero.setStreamlineDensity(parseInt(e.target.value))}
              className="custom-slider"
            />
          </div>
        </div>

        {/* Przełączniki Wizualizacji */}
        <div className="panel-section">
          <div className="section-title">
            <Eye size={16} />
            <span>VISUALIZATION OVERLAYS</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <button
              className={`btn-toggle ${aero.showPressureMap ? "active" : ""}`}
              onClick={() => aero.setShowPressureMap(!aero.showPressureMap)}
              style={{ justifyContent: "flex-start", padding: "8px 12px" }}
            >
              &bull; Surface Pressure C<sub>p</sub> Heatmap (Suction/Pressure)
            </button>
            <button
              className={`btn-toggle ${aero.showStreamlines ? "active" : ""}`}
              onClick={() => aero.setShowStreamlines(!aero.showStreamlines)}
              style={{ justifyContent: "flex-start", padding: "8px 12px" }}
            >
              &bull; 3D Analytical Streamlines (RK4)
            </button>
            <button
              className={`btn-toggle ${aero.showSmokeParticles ? "active" : ""}`}
              onClick={() => aero.setShowSmokeParticles(!aero.showSmokeParticles)}
              style={{ justifyContent: "flex-start", padding: "8px 12px" }}
            >
              &bull; Dynamic Smoke Particle Advection (120 FPS)
            </button>
            <button
              className={`btn-toggle ${aero.showVectors ? "active" : ""}`}
              onClick={() => aero.setShowVectors(!aero.showVectors)}
              style={{ justifyContent: "flex-start", padding: "8px 12px" }}
            >
              &bull; Aerodynamic Forces (Lift C<sub>L</sub>, Drag C<sub>D</sub> &amp; Stagnation)
            </button>
          </div>
        </div>

        {/* Reset Sceny */}
        <div className="panel-section">
          <button
            className="btn-drift"
            style={{ width: "100%", padding: "10px", display: "flex", justifyContent: "center", gap: "8px" }}
            onClick={aero.onResetTunnel}
          >
            <RotateCcw size={14} /> Reset NACA Wind Tunnel
          </button>
        </div>
      </div>
    );
  }

  // Scenariusz 2: Robotyka CBF
  if (activeScenario === "robotics") {
    return (
      <div className="control-panel">
        {/* Robotics Engine Status Card */}
        <div className="panel-section">
          <div className="section-title">
            <Navigation size={16} />
            <span>ROBOTICS CBF DYNAMICS</span>
          </div>
          <div className="button-group" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <button
              className={`btn-toggle ${robotics.useHocbf ? "active" : ""}`}
              onClick={() => robotics.setUseHocbf(true)}
            >
              2nd Order HOCBF
            </button>
            <button
              className={`btn-toggle ${!robotics.useHocbf ? "active" : ""}`}
              onClick={() => robotics.setUseHocbf(false)}
            >
              1st Order CBF
            </button>
          </div>
        </div>

        {/* Safety Filter Toggle Card */}
        <div className="panel-section">
          <div className="section-title">
            <Shield size={16} />
            <span>SAFETY FILTER INVARIANT</span>
          </div>
          <div
            className={`guard-toggle-card ${
              robotics.safetyEnabled ? "guard-active" : "guard-inactive"
            }`}
            onClick={() => robotics.setSafetyEnabled(!robotics.safetyEnabled)}
          >
            <div className="guard-info">
              <span className="guard-title">
                {robotics.safetyEnabled
                  ? "CBF FILTER: 0% COLLISIONS"
                  : "RAW CONTROLLER: CRASH RISK"}
              </span>
              <span className="guard-desc">
                {robotics.safetyEnabled
                  ? "Active-Set QP Projection guarantees forward invariance h(x) >= 0"
                  : "No safety envelope. Drone cuts through obstacles."}
              </span>
            </div>
            <div
              className={`switch-indicator ${
                robotics.safetyEnabled ? "switch-on" : "switch-off"
              }`}
            ></div>
          </div>
        </div>

        {/* Navigation Mode */}
        <div className="panel-section">
          <div className="section-title">
            <Radio size={16} />
            <span>NAVIGATION MISSION</span>
          </div>
          <div className="button-group" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <button
              className={`btn-toggle ${robotics.patrolMode ? "active" : ""}`}
              onClick={() => robotics.setPatrolMode(true)}
            >
              Autonomous Patrol
            </button>
            <button
              className={`btn-toggle ${!robotics.patrolMode ? "active" : ""}`}
              onClick={() => robotics.setPatrolMode(false)}
            >
              Single Goal
            </button>
          </div>
        </div>

        {/* Vector Decomposition Legend */}
        <div className="panel-section">
          <div className="section-title">
            <Activity size={16} />
            <span>3D VECTOR DECOMPOSITION</span>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "6px",
              background: "var(--bg-card)",
              padding: "10px 12px",
              borderRadius: "6px",
              border: "1px solid var(--border-subtle)",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#22c55e" }}></span>
              <span style={{ color: "#22c55e", fontWeight: 700 }}>u_des</span>
              <span style={{ color: "var(--text-dim)", fontSize: "10px" }}>Nominal Goal Vector</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#ef4444" }}></span>
              <span style={{ color: "#ef4444", fontWeight: 700 }}>&nabla;h(x)</span>
              <span style={{ color: "var(--text-dim)", fontSize: "10px" }}>Barrier Gradient Normal</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#06b6d4" }}></span>
              <span style={{ color: "#06b6d4", fontWeight: 700 }}>u_safe</span>
              <span style={{ color: "var(--text-dim)", fontSize: "10px" }}>QP-Filtered Safe Control</span>
            </div>
          </div>
        </div>

        {/* Mathematical Parameters */}
        <div className="panel-section">
          <div className="section-title">
            <Sliders size={16} />
            <span>CBF PARAMETERS</span>
          </div>

          {robotics.useHocbf ? (
            <>
              <div className="slider-item">
                <div className="slider-label-row">
                  <span>HOCBF Gain &alpha;<sub>1</sub> (Velocity)</span>
                  <span className="slider-val">{robotics.alpha1.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={1.0}
                  max={12.0}
                  step={0.5}
                  value={robotics.alpha1}
                  onChange={(e) => robotics.setAlpha1(parseFloat(e.target.value))}
                  className="custom-slider"
                />
              </div>

              <div className="slider-item">
                <div className="slider-label-row">
                  <span>HOCBF Gain &alpha;<sub>2</sub> (Acceleration)</span>
                  <span className="slider-val">{robotics.alpha2.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={1.0}
                  max={10.0}
                  step={0.5}
                  value={robotics.alpha2}
                  onChange={(e) => robotics.setAlpha2(parseFloat(e.target.value))}
                  className="custom-slider"
                />
              </div>

              <div className="slider-item">
                <div className="slider-label-row">
                  <span>Max Acceleration a<sub>max</sub></span>
                  <span className="slider-val">{robotics.aMax.toFixed(1)} m/s&sup2;</span>
                </div>
                <input
                  type="range"
                  min={3.0}
                  max={16.0}
                  step={0.5}
                  value={robotics.aMax}
                  onChange={(e) => robotics.setAMax(parseFloat(e.target.value))}
                  className="custom-slider"
                />
              </div>
            </>
          ) : (
            <div className="slider-item">
              <div className="slider-label-row">
                <span>Class-K Barrier Gain &alpha;</span>
                <span className="slider-val">{robotics.alpha.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={1.0}
                max={10.0}
                step={0.5}
                value={robotics.alpha}
                onChange={(e) => robotics.setAlpha(parseFloat(e.target.value))}
                className="custom-slider"
              />
            </div>
          )}

          <div className="slider-item">
            <div className="slider-label-row">
              <span>Tangential Guidance (Anti-Saddle)</span>
              <span className="slider-val">{robotics.tangentialGain.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={3.5}
              step={0.2}
              value={robotics.tangentialGain}
              onChange={(e) => robotics.setTangentialGain(parseFloat(e.target.value))}
              className="custom-slider"
            />
          </div>

          <div className="slider-item">
            <div className="slider-label-row">
              <span>Max Flight Speed v<sub>max</sub></span>
              <span className="slider-val">{robotics.vMax.toFixed(1)} m/s</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={4.0}
              step={0.1}
              value={robotics.vMax}
              onChange={(e) => robotics.setVMax(parseFloat(e.target.value))}
              className="custom-slider"
            />
          </div>
        </div>

        {/* Reset Actions */}
        <div className="panel-section">
          <button
            className="btn-drift"
            style={{ width: "100%", padding: "10px", display: "flex", justifyContent: "center", gap: "8px" }}
            onClick={robotics.onResetDrone}
          >
            <RotateCcw size={14} /> Reset Drone Position &amp; Waypoints
          </button>
        </div>
      </div>
    );
  }

  // Scenariusz 1: WebGPU Swarm (500k)
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
            swarm.isWebGPU !== false ? "guard-active" : "guard-warning"
          }`}
          style={{ cursor: "default" }}
        >
          <div className="guard-info">
            <span className="guard-title">
              {swarm.isWebGPU !== false
                ? "WEBGPU WGSL PIPELINE"
                : "WEBGL2 / CPU FALLBACK"}
            </span>
            <span className="guard-desc">
              {swarm.isWebGPU !== false
                ? "Zero-Copy VRAM StorageBuffer (Up to 500k Agents @ 60 FPS)"
                : "Single-thread JS Evaluator (Capped at 15k Agents)"}
            </span>
          </div>
          <div
            className={`switch-indicator ${
              swarm.isWebGPU !== false ? "switch-on" : "switch-warn"
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
            className={`btn-toggle ${swarm.viewMode === "volume" ? "active" : ""}`}
            onClick={() => swarm.setViewMode("volume")}
          >
            SDF Field
          </button>
          <button
            className={`btn-toggle ${swarm.viewMode === "swarm" ? "active" : ""}`}
            onClick={() => swarm.setViewMode("swarm")}
          >
            Swarm ({swarm.numAgents >= 1000 ? `${(swarm.numAgents / 1000).toFixed(0)}k` : swarm.numAgents})
          </button>
          <button
            className={`btn-toggle ${swarm.viewMode === "dual" ? "active" : ""}`}
            onClick={() => swarm.setViewMode("dual")}
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
            swarm.safetyGuardActive ? "guard-active" : "guard-inactive"
          }`}
          onClick={() => swarm.setSafetyGuardActive(!swarm.safetyGuardActive)}
        >
          <div className="guard-info">
            <span className="guard-title">
              {swarm.safetyGuardActive
                ? "CATEGORY GUARD: ENGAGED"
                : "UNFILTERED NEURAL: DANGER"}
            </span>
            <span className="guard-desc">
              {swarm.safetyGuardActive
                ? "100% Deterministic Safety Invariant Projection"
                : "No boundary enforcement (Raw neural violations)"}
            </span>
          </div>
          <div
            className={`switch-indicator ${
              swarm.safetyGuardActive ? "switch-on" : "switch-off"
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
            <span className="slider-val">{swarm.numAgents.toLocaleString()}</span>
          </div>
          <input
            type="range"
            min={1000}
            max={swarm.isWebGPU === false ? 15000 : 500000}
            step={swarm.isWebGPU === false ? 1000 : 5000}
            value={swarm.numAgents}
            onChange={(e) => swarm.setNumAgents(parseInt(e.target.value))}
            className="custom-slider"
          />

          {/* Szybkie presety agentów */}
          {swarm.isWebGPU !== false && (
            <div className="preset-row" style={{ display: "flex", gap: "4px", marginTop: "6px" }}>
              {[10000, 50000, 100000, 250000, 500000].map((preset) => (
                <button
                  key={preset}
                  className={`palette-btn ${swarm.numAgents === preset ? "palette-active" : ""}`}
                  style={{ flex: 1, padding: "4px 2px", fontSize: "9.5px", textAlign: "center" }}
                  onClick={() => swarm.setNumAgents(preset)}
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
            <span className="slider-val">{swarm.flowSpeed.toFixed(1)}x</span>
          </div>
          <input
            type="range"
            min={0.2}
            max={3.0}
            step={0.1}
            value={swarm.flowSpeed}
            onChange={(e) => swarm.setFlowSpeed(parseFloat(e.target.value))}
            className="custom-slider"
          />
        </div>

        {swarm.viewMode !== "swarm" && (
          <>
            <div className="slider-item">
              <div className="slider-label-row">
                <span>SDF Isosurface Threshold</span>
                <span className="slider-val">{swarm.isoLevel.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.02}
                max={0.45}
                step={0.01}
                value={swarm.isoLevel}
                onChange={(e) => swarm.setIsoLevel(parseFloat(e.target.value))}
                className="custom-slider"
              />
            </div>

            <div className="slider-item">
              <div className="slider-label-row">
                <span>Field Raymarch Density</span>
                <span className="slider-val">{swarm.density.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={0.5}
                max={5.0}
                step={0.2}
                value={swarm.density}
                onChange={(e) => swarm.setDensity(parseFloat(e.target.value))}
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
          <button className="btn-drift" onClick={() => swarm.onMoveObstacle(-0.2, 0.15, 0.0)}>
            Move Threat &larr;
          </button>
          <button className="btn-drift" onClick={() => swarm.onMoveObstacle(0.2, -0.15, 0.0)}>
            Move Threat &rarr;
          </button>
          <button className="btn-drift" onClick={() => swarm.onMoveObstacle(0.0, 0.25, -0.2)}>
            Move Threat &uarr;
          </button>
          <button className="btn-drift" onClick={() => swarm.onReset()}>
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
              className={`palette-btn ${swarm.colorScheme === pal.id ? "palette-active" : ""}`}
              onClick={() => swarm.setColorScheme(pal.id)}
            >
              {pal.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
