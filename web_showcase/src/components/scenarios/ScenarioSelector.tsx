import React from "react";
import { ShieldCheck, Cpu, Navigation, Wind, TrendingUp, Activity } from "lucide-react";

export type ScenarioType = "swarm" | "robotics" | "aerodynamics" | "financialRisk" | "cardio";

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

        <button
          className={`scenario-tab-btn ${activeScenario === "financialRisk" ? "tab-active" : ""}`}
          onClick={() => onSelectScenario("financialRisk")}
        >
          <div className="tab-icon-wrapper">
            <TrendingUp size={15} className="tab-icon" />
          </div>
          <div className="tab-text-group">
            <span className="tab-title">4. 20D Risk Engine</span>
            <span className="tab-subtitle">TT-Cross 20D &bull; Analytical Greeks</span>
          </div>
          {activeScenario === "financialRisk" && (
            <span className="tab-pill-badge" style={{ background: "rgba(245, 158, 11, 0.2)", color: "var(--amber-primary)", borderColor: "rgba(245, 158, 11, 0.4)" }}>
              5,122 vs 5²⁰
            </span>
          )}
        </button>

        <button
          className={`scenario-tab-btn ${activeScenario === "cardio" ? "tab-active" : ""}`}
          onClick={() => onSelectScenario("cardio")}
        >
          <div className="tab-icon-wrapper">
            <Activity size={15} className="tab-icon" style={{ color: activeScenario === "cardio" ? "#ef4444" : undefined }} />
          </div>
          <div className="tab-text-group">
            <span className="tab-title">5. Cardio Electrophysiology</span>
            <span className="tab-subtitle">Mesh-Free Organ &amp; EKG &bull; RF Ablation</span>
          </div>
          {activeScenario === "cardio" && (
            <span className="tab-pill-badge" style={{ background: "rgba(239, 68, 68, 0.2)", color: "#ef4444", borderColor: "rgba(239, 68, 68, 0.4)" }}>
              &lt; 18 KB
            </span>
          )}
        </button>
      </div>
    </div>
  );
};


