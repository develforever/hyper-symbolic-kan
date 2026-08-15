import { useState, useMemo, useCallback } from "react";
import { KanFieldVisualizer } from "./components/KanFieldVisualizer";
import { ControlPanel } from "./components/ControlPanel";
import { TelemetryOverlay } from "./components/TelemetryOverlay";
import { KanEvaluator, type KANModelData } from "./engine/kanEvaluator";
import initialWeights from "./data/initial_kan_weights.json";
import { Sparkles, Terminal, Box } from "lucide-react";

export function App() {
  const [modelData, setModelData] = useState<KANModelData>(initialWeights as unknown as KANModelData);
  const [viewMode, setViewMode] = useState<"volume" | "swarm" | "dual">("dual");
  const [safetyGuardActive, setSafetyGuardActive] = useState<boolean>(true);
  const [numAgents, setNumAgents] = useState<number>(8000);
  const [isoLevel, setIsoLevel] = useState<number>(0.10);
  const [density, setDensity] = useState<number>(3.5);
  const [colorScheme, setColorScheme] = useState<number>(1); // Cyan Steel
  const [flowSpeed, setFlowSpeed] = useState<number>(1.0);
  const noiseAmount = 0.05;
  const [violations, setViolations] = useState<number>(0);
  const [obstaclePos, setObstaclePos] = useState<[number, number, number]>([0.4, -0.25, 0.1]);

  const evaluator = useMemo(() => {
    return new KanEvaluator(modelData);
  }, [modelData]);

  // Adaptacja Online Streaming ALS przy przesunięciu przeszkody
  const handleMoveObstacle = useCallback((dx: number, dy: number, dz: number) => {
    setObstaclePos((prev) => {
      const nextPos: [number, number, number] = [
        Math.max(-0.8, Math.min(0.8, prev[0] + dx)),
        Math.max(-0.8, Math.min(0.8, prev[1] + dy)),
        Math.max(-0.8, Math.min(0.8, prev[2] + dz)),
      ];

      // Strumieniowa adaptacja wag KAN (Streaming ALS w locie w JS!)
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

  const handleReset = useCallback(() => {
    setModelData(initialWeights as unknown as KANModelData);
    setObstaclePos([0.4, -0.25, 0.1]);
    setIsoLevel(0.35);
    setDensity(2.4);
    setFlowSpeed(1.0);
    setSafetyGuardActive(true);
  }, []);

  return (
    <div className="app-container">
      {/* Pasek nawigacyjny */}
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

        <div className="header-badges">
          <span className="badge badge-accent">
            <Sparkles size={12} /> WebGL / WebGPU Shader
          </span>
          <span className="badge badge-mono">
            <Terminal size={12} /> 0 Backprop Epochs
          </span>
        </div>
      </header>

      {/* Główny obszar roboczy */}
      <main className="app-main">
        {/* Widok 3D Canvas */}
        <div className="viewport-container">
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
          />

          {/* Nakładka telemetryczna */}
          <TelemetryOverlay
            rank={modelData.rank}
            degree={modelData.degree}
            numAgents={numAgents}
            violations={violations}
            safetyGuardActive={safetyGuardActive}
          />
        </div>

        {/* Boczny panel kontrolny */}
        <aside className="sidebar-container">
          <ControlPanel
            viewMode={viewMode}
            setViewMode={setViewMode}
            safetyGuardActive={safetyGuardActive}
            setSafetyGuardActive={setSafetyGuardActive}
            numAgents={numAgents}
            setNumAgents={setNumAgents}
            isoLevel={isoLevel}
            setIsoLevel={setIsoLevel}
            density={density}
            setDensity={setDensity}
            colorScheme={colorScheme}
            setColorScheme={setColorScheme}
            flowSpeed={flowSpeed}
            setFlowSpeed={setFlowSpeed}
            onMoveObstacle={handleMoveObstacle}
            onReset={handleReset}
          />
        </aside>
      </main>
    </div>
  );
}

export default App;
