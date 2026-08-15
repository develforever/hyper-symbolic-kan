import React from "react";
import { ShieldCheck, Cpu, Navigation, Wind } from "lucide-react";

export type ScenarioType = "swarm" | "robotics" | "aerodynamics";

interface ScenarioSelectorProps {
  activeScenario: ScenarioType;
  onSelectScenario: (scenario: ScenarioType) => void;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  activeScenario,
  onSelectScenario,
}) => {
  return (
    <div className="scenario-selector-container">
      <div className="scenario-tabs">
        <button
          className={`scenario-tab-btn ${activeScenario === "swarm" ? "tab-active" : ""}`}
          onClick={() => onSelectScenario("swarm")}
        >
          <div className="tab-icon-wrapper">
            <Cpu size={15} className="tab-icon" />
          </div>
          <div className="tab-text-group">
            <span className="tab-title">1. WebGPU Swarm</span>
            <span className="tab-subtitle">500k Tensor Field &bull; SDF</span>
          </div>
          {activeScenario === "swarm" && <span className="tab-pill-badge">WGSL</span>}
        </button>

        <button
          className={`scenario-tab-btn ${activeScenario === "robotics" ? "tab-active" : ""}`}
          onClick={() => onSelectScenario("robotics")}
        >
          <div className="tab-icon-wrapper">
            <Navigation size={15} className="tab-icon" />
          </div>
          <div className="tab-text-group">
            <span className="tab-title">2. Drone Safety CBF</span>
            <span className="tab-subtitle">2nd Order HOCBF &bull; 120 FPS</span>
          </div>
          {activeScenario === "robotics" && (
            <span className="tab-pill-badge" style={{ background: "rgba(16, 185, 129, 0.2)", color: "var(--emerald-primary)", borderColor: "rgba(16, 185, 129, 0.4)" }}>
              <ShieldCheck size={11} /> 0% Collision
            </span>
          )}
        </button>

        <button
          className={`scenario-tab-btn ${activeScenario === "aerodynamics" ? "tab-active" : ""}`}
          onClick={() => onSelectScenario("aerodynamics")}
        >
          <div className="tab-icon-wrapper">
            <Wind size={15} className="tab-icon" />
          </div>
          <div className="tab-text-group">
            <span className="tab-title">3. Aerodynamics CFD</span>
            <span className="tab-subtitle">Mesh-Free NACA &bull; 0 Epochs</span>
          </div>
          {activeScenario === "aerodynamics" && (
            <span className="tab-pill-badge" style={{ background: "rgba(6, 182, 212, 0.2)", color: "var(--cyan-primary)", borderColor: "rgba(6, 182, 212, 0.4)" }}>
              &lt; 2 ms
            </span>
          )}
        </button>
      </div>
    </div>
  );
};
