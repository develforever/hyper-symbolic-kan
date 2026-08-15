import { useState, useMemo, useCallback } from "react";
import { KanFieldVisualizer } from "./components/KanFieldVisualizer";
import { ControlPanel } from "./components/ControlPanel";
import { TelemetryOverlay, type RoboticsTelemetryProps } from "./components/TelemetryOverlay";
import { ScenarioSelector, type ScenarioType } from "./components/scenarios/ScenarioSelector";
import { RoboticsCBFScenario } from "./components/scenarios/RoboticsCBFScenario";
import { KanEvaluator, type KANModelData } from "./engine/kanEvaluator";
import { RoboticsCBFEngine } from "./engine/roboticsCbfEngine";
import initialWeights from "./data/initial_kan_weights.json";
import { Sparkles, Terminal, Box } from "lucide-react";

export function App() {
  // Nawigacja Scenariuszy
  const [activeScenario, setActiveScenario] = useState<ScenarioType>("robotics");

  // Stan Scenariusza 1: WebGPU Swarm & Continuous KAN
  const [modelData, setModelData] = useState<KANModelData>(initialWeights as unknown as KANModelData);
  const [viewMode, setViewMode] = useState<"volume" | "swarm" | "dual">("dual");
  const [safetyGuardActive, setSafetyGuardActive] = useState<boolean>(true);
  const [numAgents, setNumAgents] = useState<number>(100000);
  const [isoLevel, setIsoLevel] = useState<number>(0.10);
  const [density, setDensity] = useState<number>(3.5);
  const [colorScheme, setColorScheme] = useState<number>(1); // Cyan Steel
  const [flowSpeed, setFlowSpeed] = useState<number>(1.0);
  const noiseAmount = 0.05;
  const [violations, setViolations] = useState<number>(0);
  const [obstaclePos, setObstaclePos] = useState<[number, number, number]>([0.4, -0.25, 0.1]);
  const [isWebGPU, setIsWebGPU] = useState<boolean | null>(null);

  // Stan Scenariusza 2: Robotics 3D Drone & Dynamic HOCBF
  const [useHocbf, setUseHocbf] = useState<boolean>(true);
  const [cbfSafetyEnabled, setCbfSafetyEnabled] = useState<boolean>(true);
  const [alpha, setAlpha] = useState<number>(3.5);
  const [alpha1, setAlpha1] = useState<number>(6.0);
  const [alpha2, setAlpha2] = useState<number>(4.0);
  const [vMax, setVMax] = useState<number>(2.2);
  const [aMax, setAMax] = useState<number>(9.0);
  const [tangentialGain, setTangentialGain] = useState<number>(1.8);
  const [patrolMode, setPatrolMode] = useState<boolean>(true);

  const [roboticsTelemetry, setRoboticsTelemetry] = useState<RoboticsTelemetryProps>({
    qpLatencyUs: 4.2,
    minH: 0.25,
    speed: 0.0,
    accel: 0.0,
    collision: false,
    useHocbf: true,
    safetyEnabled: true,
  });

  const evaluator = useMemo(() => {
    return new KanEvaluator(modelData);
  }, [modelData]);

  const cbfEngine = useMemo(() => {
    return new RoboticsCBFEngine({
      alpha,
      alpha1,
      alpha2,
      vMax,
      aMax,
      tangentialGain,
      useHocbf,
      safetyEnabled: cbfSafetyEnabled,
    });
  }, []);

  // Adaptacja Online Streaming ALS przy przesunięciu przeszkody (Swarm)
  const handleMoveObstacle = useCallback((dx: number, dy: number, dz: number) => {
    setObstaclePos((prev) => {
      const nextPos: [number, number, number] = [
        Math.max(-0.8, Math.min(0.8, prev[0] + dx)),
        Math.max(-0.8, Math.min(0.8, prev[1] + dy)),
        Math.max(-0.8, Math.min(0.8, prev[2] + dz)),
      ];

      // Strumieniowa adaptacja wag KAN (Streaming ALS w locie)
      const N_samples = 60;
      for (let i = 0; i < N_samples; i++) {
        const sx = (Math.random() - 0.5) * 1.8;
        const sy = (Math.random() - 0.5) * 1.8;
        const sz = (Math.random() - 0.5) * 1.8;

        const dOld1 = Math.sqrt((sx - -0.35) ** 2 + (sy - 0.2) ** 2 + sz ** 2);
        const dNew2 = Math.sqrt((sx - nextPos[0]) ** 2 + (sy - nextPos[1]) ** 2 + (sz - nextPos[2]) ** 2);

        const target =
          Math.exp(-(dOld1 ** 2) / 0.12) +
          0.8 * Math.exp(-(dNew2 ** 2) / 0.15) +
          0.25 * Math.cos(Math.PI * sx) * Math.sin(Math.PI * sy);

        evaluator.updateOnlineStreaming(sx, sy, sz, target, 0.08);
      }

      // Aktualizacja stanu wag dla shadera GPU
      setModelData((curr) => ({
        ...curr,
        lambdas: Array.from(evaluator.lambdas),
      }));

      return nextPos;
    });
  }, [evaluator]);

  const handleResetSwarm = useCallback(() => {
    setModelData(initialWeights as unknown as KANModelData);
    setObstaclePos([0.4, -0.25, 0.1]);
    setIsoLevel(0.10);
    setDensity(3.5);
    setFlowSpeed(1.0);
    setSafetyGuardActive(true);
  }, []);

  const handleResetDrone = useCallback(() => {
    cbfEngine.resetDrone([-0.75, -0.6, -0.2]);
    cbfEngine.setGoal([0.7, -0.4, 0.2]);
  }, [cbfEngine]);

  return (
    <div className="app-container">
      {/* Pasek nawigacyjny z Selektorem Scenariuszy */}
      <header className="app-header">
        <div className="header-brand">
          <Box className="header-logo-icon" size={22} />
          <div>
            <div className="header-title">HYPER-SYMBOLIC KAN SHOWCASE</div>
            <div className="header-subtitle">
              Continuous Tensor Fields &bull; Exact Analytical Gradients &bull; Monadic Invariants
            </div>
          </div>
        </div>

        {/* Selektor Scenariuszy */}
        <ScenarioSelector
          activeScenario={activeScenario}
          onSelectScenario={setActiveScenario}
        />

        <div className="header-badges">
          <span className="badge badge-accent">
            <Sparkles size={12} />{" "}
            {activeScenario === "robotics"
              ? "120 FPS HOCBF Digital Twin"
              : isWebGPU !== false
              ? "WebGPU WGSL Compute (500k)"
              : "WebGL2 Fallback"}
          </span>
          <span className="badge badge-mono">
            <Terminal size={12} /> 0 Backprop Epochs
          </span>
        </div>
      </header>

      {/* Główny obszar roboczy */}
      <main className="app-main">
        {/* Widok 3D Canvas w zależności od aktywnego scenariusza */}
        <div className="viewport-container">
          <div
            style={{
              width: "100%",
              height: "100%",
              position: "absolute",
              top: 0,
              left: 0,
              display: activeScenario === "swarm" ? "block" : "none",
            }}
          >
            <KanFieldVisualizer
              modelData={modelData}
              evaluator={evaluator}
              viewMode={viewMode}
              isoLevel={isoLevel}
              density={density}
              colorScheme={colorScheme}
              safetyGuardActive={safetyGuardActive}
              numAgents={numAgents}
              flowSpeed={flowSpeed}
              noiseAmount={noiseAmount}
              obstaclePos={obstaclePos}
              onViolationCount={setViolations}
              onWebGPUStatus={setIsWebGPU}
            />
          </div>

          <div
            style={{
              width: "100%",
              height: "100%",
              position: "absolute",
              top: 0,
              left: 0,
              display: activeScenario === "robotics" ? "block" : "none",
            }}
          >
            <RoboticsCBFScenario
              cbfEngine={cbfEngine}
              useHocbf={useHocbf}
              safetyEnabled={cbfSafetyEnabled}
              alpha={alpha}
              alpha1={alpha1}
              alpha2={alpha2}
              vMax={vMax}
              aMax={aMax}
              tangentialGain={tangentialGain}
              patrolMode={patrolMode}
              onTelemetryUpdate={setRoboticsTelemetry}
            />
          </div>

          {/* Nakładka telemetryczna */}
          <TelemetryOverlay
            mode={activeScenario}
            swarm={{
              rank: modelData.rank,
              degree: modelData.degree,
              numAgents,
              violations,
              safetyGuardActive,
              isWebGPU,
            }}
            robotics={roboticsTelemetry}
          />
        </div>

        {/* Boczny panel kontrolny */}
        <aside className="sidebar-container">
          <ControlPanel
            activeScenario={activeScenario}
            swarm={{
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
              onMoveObstacle: handleMoveObstacle,
              onReset: handleResetSwarm,
            }}
            robotics={{
              useHocbf,
              setUseHocbf,
              safetyEnabled: cbfSafetyEnabled,
              setSafetyEnabled: setCbfSafetyEnabled,
              alpha,
              setAlpha,
              alpha1,
              setAlpha1,
              alpha2,
              setAlpha2,
              vMax,
              setVMax,
              aMax,
              setAMax,
              tangentialGain,
              setTangentialGain,
              patrolMode,
              setPatrolMode,
              onResetDrone: handleResetDrone,
            }}
          />
        </aside>
      </main>
    </div>
  );
}

export default App;
