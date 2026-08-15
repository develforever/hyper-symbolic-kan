import React, { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { createKanShaderMaterial } from "../shaders/kanVolumeShader";
import { SwarmSimulation } from "./SwarmSimulation";
import { KanEvaluator, type KANModelData } from "../engine/kanEvaluator";

interface KanFieldVisualizerProps {
  modelData: KANModelData;
  evaluator: KanEvaluator;
  viewMode: "volume" | "swarm" | "dual";
  isoLevel: number;
  density: number;
  colorScheme: number;
  safetyGuardActive: boolean;
  numAgents: number;
  flowSpeed: number;
  noiseAmount: number;
  obstaclePos: [number, number, number];
  onViolationCount: (violations: number) => void;
}

const VolumeBox: React.FC<{
  modelData: KANModelData;
  isoLevel: number;
  density: number;
  colorScheme: number;
}> = ({ modelData, isoLevel, density, colorScheme }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  const material = useMemo(() => {
    return createKanShaderMaterial({
      lambdas: modelData.lambdas,
      factors: modelData.factors,
      rank: modelData.rank,
      degree: modelData.degree,
    });
  }, [modelData]);

  useEffect(() => {
    if (material) {
      material.uniforms.u_isoLevel.value = isoLevel;
      material.uniforms.u_density.value = density;
      material.uniforms.u_colorScheme.value = colorScheme;
    }
  }, [isoLevel, density, colorScheme, material]);

  useFrame((state) => {
    if (material) {
      material.uniforms.u_time.value = state.clock.getElapsedTime();
    }
  });

  return (
    <mesh ref={meshRef} material={material}>
      <boxGeometry args={[2, 2, 2]} />
    </mesh>
  );
};

export const KanFieldVisualizer: React.FC<KanFieldVisualizerProps> = ({
  modelData,
  evaluator,
  viewMode,
  isoLevel,
  density,
  colorScheme,
  safetyGuardActive,
  numAgents,
  flowSpeed,
  noiseAmount,
  obstaclePos,
  onViolationCount,
}) => {
  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas
        gl={{
          antialias: true,
          powerPreference: "high-performance",
        }}
      >
        <PerspectiveCamera makeDefault position={[2.6, 2.2, 3.2]} fov={45} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          maxDistance={8.0}
          minDistance={1.2}
        />

        {/* Oświetlenie sceny */}
        <ambientLight intensity={0.4} />
        <directionalLight position={[3, 4, 3]} intensity={1.2} />
        <directionalLight position={[-3, -2, -3]} intensity={0.3} color="#4080ff" />

        {/* Wolumetryczne pole skalarne KAN (Raymarching bez siatki) */}
        {(viewMode === "volume" || viewMode === "dual") && (
          <VolumeBox
            modelData={modelData}
            isoLevel={isoLevel}
            density={density}
            colorScheme={colorScheme}
          />
        )}

        {/* Symulacja roju 10k cząstek */}
        {(viewMode === "swarm" || viewMode === "dual") && (
          <SwarmSimulation
            evaluator={evaluator}
            numAgents={numAgents}
            safetyGuardActive={safetyGuardActive}
            flowSpeed={flowSpeed}
            noiseAmount={noiseAmount}
            onViolationCount={onViolationCount}
          />
        )}

        {/* Ramka domenowa [-1, 1]^3 */}
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(2, 2, 2)]} />
          <lineBasicMaterial color="#334155" opacity={0.5} transparent />
        </lineSegments>

        {/* Strefa No-Fly Zone (Inwariant Kategorialny) */}
        <mesh position={[-0.35, 0.2, 0.0]}>
          <sphereGeometry args={[0.35, 24, 24]} />
          <meshBasicMaterial
            color={safetyGuardActive ? "#06b6d4" : "#ef4444"}
            wireframe
            transparent
            opacity={0.35}
          />
        </mesh>

        {/* Ruchoma przeszkoda (Interaktywne źródło pola) */}
        <mesh position={obstaclePos}>
          <sphereGeometry args={[0.18, 20, 20]} />
          <meshStandardMaterial
            color="#f59e0b"
            emissive="#d97706"
            emissiveIntensity={0.6}
            roughness={0.2}
            metalness={0.8}
          />
        </mesh>
      </Canvas>
    </div>
  );
};
