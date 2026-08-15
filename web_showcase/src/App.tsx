import { useState, useMemo, useCallback } from "react";
import { KanFieldVisualizer } from "./components/KanFieldVisualizer";
import { ControlPanel } from "./components/ControlPanel";
import {
  TelemetryOverlay,
  type RoboticsTelemetryProps,
  type AerodynamicsTelemetryProps,
} from "./components/TelemetryOverlay";
import { ScenarioSelector, type ScenarioType } from "./components/scenarios/ScenarioSelector";
import { RoboticsCBFScenario } from "./components/scenarios/RoboticsCBFScenario";
import { AerodynamicsCFDScenario } from "./components/scenarios/AerodynamicsCFDScenario";
import { FinancialRisk20DScenario } from "./components/scenarios/FinancialRisk20DScenario";
import { KanEvaluator, type KANModelData } from "./engine/kanEvaluator";
import { RoboticsCBFEngine } from "./engine/roboticsCbfEngine";
import { type NACAProfileConfig, type CFDSolverResult } from "./engine/aerodynamicsCfdEngine";
import {
  FinancialRiskEngine,
  type MarketCrashPreset,
  type RiskEngineTelemetry,
} from "./engine/financialRiskEngine";
import initialWeights from "./data/initial_kan_weights.json";
import { Sparkles, Terminal, Box } from "lucide-react";

export function App() {
  // Nawigacja Scenariuszy (1: Swarm, 2: Robotics, 3: Aerodynamics CFD, 4: Financial Risk 20D)
  const [activeScenario, setActiveScenario] = useState<ScenarioType>("financialRisk");

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

  // Stan Scenariusza 3: Mesh-Free Aerodynamics CFD Wind Tunnel (0 Epochs)
  const [aoaDeg, setAoaDeg] = useState<number>(5.0);
  const [uInf, setUInf] = useState<number>(25.0);
  const [airfoilConfig, setAirfoilConfig] = useState<NACAProfileConfig>({
    camber: 0.02,
    camberPos: 0.4,
    thickness: 0.12,
    chord: 1.0,
    nStations: 70,
  });
  const [showStreamlines, setShowStreamlines] = useState<boolean>(true);
  const [showPressureMap, setShowPressureMap] = useState<boolean>(true);
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [showSmokeParticles, setShowSmokeParticles] = useState<boolean>(true);
  const [streamlineDensity, setStreamlineDensity] = useState<number>(24);

  const [aerodynamicsTelemetry, setAerodynamicsTelemetry] = useState<AerodynamicsTelemetryProps>({
    solveTimeMs: 0.35,
    cl: 0.72,
    cd: 0.021,
    cm: -0.05,
    glideRatio: 34.3,
    circulation: 18.5,
    stagnationPoint: [-0.24, -0.02],
    stagnationCp: 1.0,
    minCp: -2.35,
    maxVelocity: 42.1,
    pdeResidualL2: 1.2e-14,
    aoaDeg: 5.0,
    uInf: 25.0,
    airfoilName: "NACA 2412",
    isStalled: false,
  });

  // Stan Scenariusza 4: 20D Financial Risk & Analytical Greeks Engine
  const [riskAxisX, setRiskAxisX] = useState<number>(0); // NVDA
  const [riskAxisY, setRiskAxisY] = useState<number>(7); // BTC
  const [riskVolShock, setRiskVolShock] = useState<number>(1.0);
  const [riskStressPreset, setRiskStressPreset] = useState<MarketCrashPreset>("EQUILIBRIUM");
  const [riskState20D, setRiskState20D] = useState<Float64Array>(() => new Float64Array(20));
  const [riskShowWireframe, setRiskShowWireframe] = useState<boolean>(true);
  const [riskShowContourLines, setRiskShowContourLines] = useState<boolean>(false);
  const [riskShowGreeksVectors, setRiskShowGreeksVectors] = useState<boolean>(true);
  const [riskShowCorrelationWeb, setRiskShowCorrelationWeb] = useState<boolean>(true);
  const [financialRiskTelemetry, setFinancialRiskTelemetry] = useState<RiskEngineTelemetry | undefined>(undefined);

  const riskEngine = useMemo(() => new FinancialRiskEngine(), []);

  const handleAeroTelemetry = useCallback((res: CFDSolverResult) => {
    setAerodynamicsTelemetry({
      solveTimeMs: res.solveTimeMs,
      cl: res.cl,
      cd: res.cd,
      cm: res.cm,
      glideRatio: res.glideRatio,
      circulation: res.circulation,
      stagnationPoint: res.stagnationPoint,
      stagnationCp: res.stagnationCp,
      minCp: res.minCp,
      maxVelocity: res.maxVelocity,
      pdeResidualL2: res.pdeResidualL2,
      aoaDeg: res.aoaDeg,
      uInf: res.uInf,
      airfoilName: res.airfoilName,
      isStalled: res.isStalled,
    });
  }, []);

  const handleResetTunnel = useCallback(() => {
    setAoaDeg(5.0);
    setUInf(25.0);
    setAirfoilConfig({
      camber: 0.02,
      camberPos: 0.4,
      thickness: 0.12,
      chord: 1.0,
      nStations: 70,
    });
    setShowStreamlines(true);
    setShowPressureMap(true);
    setShowVectors(true);
    setShowSmokeParticles(true);
    setStreamlineDensity(24);
  }, []);

  const handleResetRisk = useCallback(() => {
    setRiskAxisX(0);
    setRiskAxisY(7);
    setRiskVolShock(1.0);
    setRiskStressPreset("EQUILIBRIUM");
    setRiskState20D(new Float64Array(20));
    setRiskShowWireframe(true);
    setRiskShowContourLines(false);
    setRiskShowGreeksVectors(true);
    setRiskShowCorrelationWeb(true);
  }, []);

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

        {/* Selektor Scenariuszy (4 Opcje) */}
        <ScenarioSelector
          activeScenario={activeScenario}
          onSelectScenario={setActiveScenario}
        />

        <div className="header-badges">
          <span className="badge badge-accent">
            <Sparkles size={12} />{" "}
            {activeScenario === "financialRisk"
              ? "20D TT-KAN Risk Engine"
              : activeScenario === "aerodynamics"
              ? "Mesh-Free NACA CFD"
              : activeScenario === "robotics"
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
          {/* Scenariusz 1: WebGPU Swarm */}
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

          {/* Scenariusz 2: Robotics CBF Drone */}
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

          {/* Scenariusz 3: Aerodynamics CFD Wind Tunnel */}
          <div
            style={{
              width: "100%",
              height: "100%",
              position: "absolute",
              top: 0,
              left: 0,
              display: activeScenario === "aerodynamics" ? "block" : "none",
            }}
          >
            <AerodynamicsCFDScenario
              aoaDeg={aoaDeg}
              uInf={uInf}
              airfoilConfig={airfoilConfig}
              showStreamlines={showStreamlines}
              showPressureMap={showPressureMap}
              showVectors={showVectors}
              showSmokeParticles={showSmokeParticles}
              streamlineDensity={streamlineDensity}
              onTelemetryUpdate={handleAeroTelemetry}
            />
          </div>

          {/* Scenariusz 4: 20D Financial Risk Engine */}
          <div
            style={{
              width: "100%",
              height: "100%",
              position: "absolute",
              top: 0,
              left: 0,
              display: activeScenario === "financialRisk" ? "block" : "none",
            }}
          >
            <FinancialRisk20DScenario
              engine={riskEngine}
              axisX={riskAxisX}
              axisY={riskAxisY}
              state20D={riskState20D}
              volShock={riskVolShock}
              stressPreset={riskStressPreset}
              showWireframe={riskShowWireframe}
              showContourLines={riskShowContourLines}
              showGreeksVectors={riskShowGreeksVectors}
              showCorrelationWeb={riskShowCorrelationWeb}
              onTelemetryUpdate={setFinancialRiskTelemetry}
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
            aerodynamics={aerodynamicsTelemetry}
            financialRisk={financialRiskTelemetry}
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
            aerodynamics={{
              aoaDeg,
              setAoaDeg,
              uInf,
              setUInf,
              airfoilConfig,
              setAirfoilConfig,
              showStreamlines,
              setShowStreamlines,
              showPressureMap,
              setShowPressureMap,
              showVectors,
              setShowVectors,
              showSmokeParticles,
              setShowSmokeParticles,
              streamlineDensity,
              setStreamlineDensity,
              onResetTunnel: handleResetTunnel,
            }}
            financialRisk={{
              axisX: riskAxisX,
              setAxisX: setRiskAxisX,
              axisY: riskAxisY,
              setAxisY: setRiskAxisY,
              volShock: riskVolShock,
              setVolShock: setRiskVolShock,
              stressPreset: riskStressPreset,
              setStressPreset: setRiskStressPreset,
              state20D: riskState20D,
              setState20D: setRiskState20D,
              showWireframe: riskShowWireframe,
              setShowWireframe: setRiskShowWireframe,
              showContourLines: riskShowContourLines,
              setShowContourLines: setRiskShowContourLines,
              showGreeksVectors: riskShowGreeksVectors,
              setShowGreeksVectors: setRiskShowGreeksVectors,
              showCorrelationWeb: riskShowCorrelationWeb,
              setShowCorrelationWeb: setRiskShowCorrelationWeb,
              onResetRisk: handleResetRisk,
            }}
          />
        </aside>
      </main>
    </div>
  );
}

export default App;

