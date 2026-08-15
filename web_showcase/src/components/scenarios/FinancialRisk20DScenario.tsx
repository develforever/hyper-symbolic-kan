import React, { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  FinancialRiskEngine,
  ASSETS_20D,
  type MarketCrashPreset,
  type RiskEngineTelemetry,
  type Surface2DResult,
} from "../../engine/financialRiskEngine";

export interface FinancialRisk20DScenarioProps {
  engine: FinancialRiskEngine;
  axisX: number;
  axisY: number;
  state20D: Float64Array;
  volShock: number;
  stressPreset: MarketCrashPreset;
  showWireframe: boolean;
  showContourLines: boolean;
  showGreeksVectors: boolean;
  showCorrelationWeb: boolean;
  onTelemetryUpdate: (telemetry: RiskEngineTelemetry) => void;
}

/**
 * 3D Dynamic Risk Hypersurface Mesh
 */
const RiskHypersurfaceMesh: React.FC<{
  surfaceData: Surface2DResult;
  showWireframe: boolean;
  showContourLines: boolean;
}> = ({ surfaceData, showWireframe, showContourLines }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);

  const { geometry, wireGeometry } = useMemo(() => {
    const N = surfaceData.resolution;
    const geom = new THREE.BufferGeometry();

    const positions = new Float32Array(N * N * 3);
    const normals = new Float32Array(N * N * 3);
    const colors = new Float32Array(N * N * 3);
    const indices: number[] = [];
    const wireIndices: number[] = [];

    const scaleXY = 2.2;
    const scaleZ = 0.08;
    const midZ = (surfaceData.minZ + surfaceData.maxZ) / 2.0;

    for (let iy = 0; iy < N; iy++) {
      const y = surfaceData.yValues[iy] * scaleXY;
      for (let ix = 0; ix < N; ix++) {
        const x = surfaceData.xValues[ix] * scaleXY;
        const idx = iy * N + ix;
        const rawZ = surfaceData.zValues[idx];
        const z = (rawZ - midZ) * scaleZ;

        positions[idx * 3] = x;
        positions[idx * 3 + 1] = z; // Y is UP in Three.js standard
        positions[idx * 3 + 2] = y;

        normals[idx * 3] = surfaceData.normals[idx * 3];
        normals[idx * 3 + 1] = surfaceData.normals[idx * 3 + 2]; // map Z normal to Y
        normals[idx * 3 + 2] = surfaceData.normals[idx * 3 + 1];

        colors[idx * 3] = surfaceData.colors[idx * 3];
        colors[idx * 3 + 1] = surfaceData.colors[idx * 3 + 1];
        colors[idx * 3 + 2] = surfaceData.colors[idx * 3 + 2];

        // Grid triangulations
        if (ix < N - 1 && iy < N - 1) {
          const a = iy * N + ix;
          const b = iy * N + (ix + 1);
          const c = (iy + 1) * N + ix;
          const d = (iy + 1) * N + (ix + 1);

          indices.push(a, b, c);
          indices.push(b, d, c);
        }

        // Wireframe grid lines
        if (ix < N - 1) {
          wireIndices.push(idx, idx + 1);
        }
        if (iy < N - 1) {
          wireIndices.push(idx, idx + N);
        }
      }
    }

    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.setIndex(indices);

    const wireGeom = new THREE.BufferGeometry();
    wireGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    wireGeom.setIndex(wireIndices);

    return { geometry: geom, wireGeometry: wireGeom };
  }, [surfaceData]);

  return (
    <group>
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          roughness={0.25}
          metalness={0.65}
          transparent
          opacity={0.88}
          side={THREE.DoubleSide}
        />
      </mesh>

      {showWireframe && (
        <lineSegments ref={wireRef} geometry={wireGeometry}>
          <lineBasicMaterial
            color="#06b6d4"
            transparent
            opacity={0.25}
            linewidth={1}
          />
        </lineSegments>
      )}

      {showContourLines && (
        <mesh position={[0, -0.01, 0]} geometry={geometry}>
          <meshBasicMaterial
            color="#ffffff"
            wireframe
            transparent
            opacity={0.08}
          />
        </mesh>
      )}
    </group>
  );
};

/**
 * 3D Current Portfolio Operating Point with Tangent Gradient & Curvature Vectors
 */
const PortfolioOperatingPoint: React.FC<{
  currentX: number;
  currentY: number;
  currentValue: number;
  deltaX: number;
  deltaY: number;
  gammaX: number;
  gammaY: number;
  surfaceData: Surface2DResult;
  showVectors: boolean;
}> = ({
  currentX,
  currentY,
  currentValue,
  deltaX,
  deltaY,
  gammaX,
  gammaY,
  surfaceData,
  showVectors,
}) => {
  const sphereRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const vectorGroupRef = useRef<THREE.Group>(null);

  const scaleXY = 2.2;
  const scaleZ = 0.08;
  const midZ = (surfaceData.minZ + surfaceData.maxZ) / 2.0;

  const posX = currentX * scaleXY;
  const posY = (currentValue - midZ) * scaleZ;
  const posZ = currentY * scaleXY;

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (sphereRef.current) {
      const s = 1.0 + 0.12 * Math.sin(t * 4.0);
      sphereRef.current.scale.set(s, s, s);
    }
    if (glowRef.current) {
      const s = 1.6 + 0.3 * Math.sin(t * 3.0);
      glowRef.current.scale.set(s, s, s);
    }
  });

  // Vector gradient arrow length
  const arrowLen = Math.min(1.2, Math.sqrt(deltaX * deltaX + deltaY * deltaY) * 0.15);
  const gradAngle = Math.atan2(deltaY, deltaX);

  return (
    <group position={[posX, posY, posZ]}>
      {/* Central Operating Sphere */}
      <mesh ref={sphereRef}>
        <sphereGeometry args={[0.07, 24, 24]} />
        <meshStandardMaterial
          color="#f59e0b"
          emissive="#f59e0b"
          emissiveIntensity={1.2}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Pulsing Outer Glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshBasicMaterial
          color="#f59e0b"
          transparent
          opacity={0.3}
          wireframe
        />
      </mesh>

      {/* Projection Stalk to Ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([0, 0, 0, 0, -posY - 1.2, 0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="#f59e0b"
          transparent
          opacity={0.5}
        />
      </line>

      {/* Tangent Gradient Vectors & Curvature Ring */}
      {showVectors && (
        <group ref={vectorGroupRef}>
          {/* Delta Gradient Arrow in X-Z plane */}
          <group rotation={[0, -gradAngle, 0]}>
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  args={[new Float32Array([0, 0, 0, arrowLen, 0, 0]), 3]}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#06b6d4" linewidth={2} />
            </line>
            <mesh position={[arrowLen, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
              <coneGeometry args={[0.03, 0.08, 12]} />
              <meshBasicMaterial color="#06b6d4" />
            </mesh>
          </group>

          {/* Gamma Curvature Ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.15, 0.17, 32]} />
            <meshBasicMaterial
              color="#ec4899"
              side={THREE.DoubleSide}
              transparent
              opacity={Math.min(0.8, (Math.abs(gammaX) + Math.abs(gammaY)) * 0.2 + 0.2)}
            />
          </mesh>
        </group>
      )}

      <Text
        position={[0, 0.22, 0]}
        fontSize={0.11}
        color="#f8fafc"
        anchorX="center"
        anchorY="bottom"
      >
        {`V = $${currentValue.toFixed(2)}M`}
      </Text>
    </group>
  );
};

/**
 * 20D Spatial Asset Perimeter Radar Arena with Exposure Pillars
 */
const SpatialAssetPerimeterRing: React.FC<{
  state20D: Float64Array;
  deltas: Float64Array;
  gammas: Float64Array;
  axisX: number;
  axisY: number;
  radius?: number;
}> = ({ state20D, deltas, gammas, axisX, axisY, radius = 3.4 }) => {
  const nodes = useMemo(() => {
    return ASSETS_20D.map((asset, i) => {
      const angle = (i / ASSETS_20D.length) * Math.PI * 2 - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const isX = i === axisX;
      const isY = i === axisY;
      const delta = deltas[i] || 0.0;
      const gamma = gammas[i] || 0.0;
      const stateVal = state20D[i] || 0.0;

      // Pillar height proportional to delta exposure
      const pillarHeight = Math.max(0.1, Math.min(1.2, Math.abs(delta) * 0.4 + 0.15));

      return {
        ...asset,
        x,
        z,
        angle,
        isX,
        isY,
        delta,
        gamma,
        stateVal,
        pillarHeight,
      };
    });
  }, [state20D, deltas, gammas, axisX, axisY, radius]);

  return (
    <group position={[0, -0.6, 0]}>
      {/* Circular Perimeter Base Rail */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
        <ringGeometry args={[radius - 0.04, radius + 0.04, 64]} />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.6} side={THREE.DoubleSide} />
      </mesh>

      {nodes.map((node) => {
        const isSelected = node.isX || node.isY;
        const color = isSelected ? (node.isX ? "#06b6d4" : "#10b981") : node.color;

        return (
          <group key={node.id} position={[node.x, 0, node.z]}>
            {/* Exposure Pillar */}
            <mesh position={[0, node.pillarHeight / 2, 0]}>
              <cylinderGeometry args={[0.04, 0.04, node.pillarHeight, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={isSelected ? 0.8 : 0.2}
                roughness={0.3}
                metalness={0.7}
                transparent
                opacity={isSelected ? 0.95 : 0.65}
              />
            </mesh>

            {/* Asset Node Head Sphere */}
            <mesh position={[0, node.pillarHeight + 0.06, 0]}>
              <sphereGeometry args={[isSelected ? 0.09 : 0.06, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={isSelected ? 1.2 : 0.4}
              />
            </mesh>

            {/* Selection Pulsing Ring */}
            {isSelected && (
              <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
                <ringGeometry args={[0.12, 0.16, 24]} />
                <meshBasicMaterial color={color} side={THREE.DoubleSide} />
              </mesh>
            )}

            {/* Ticker & Greek Label */}
            <Text
              position={[0, node.pillarHeight + 0.24, 0]}
              fontSize={isSelected ? 0.13 : 0.09}
              color={isSelected ? "#ffffff" : "#94a3b8"}
              anchorX="center"
              anchorY="middle"
            >
              {`${node.symbol}${isSelected ? ` [${node.isX ? "X" : "Y"}]` : ""}`}
            </Text>

            <Text
              position={[0, node.pillarHeight + 0.14, 0]}
              fontSize={0.07}
              color={node.delta >= 0 ? "#10b981" : "#ef4444"}
              anchorX="center"
              anchorY="middle"
            >
              {`Δ:${node.delta >= 0 ? "+" : ""}${node.delta.toFixed(1)}`}
            </Text>
          </group>
        );
      })}
    </group>
  );
};

/**
 * Cross-Asset Correlation Contagion Network Arcs
 */
const CorrelationContagionWeb: React.FC<{
  stressPreset: MarketCrashPreset;
  radius?: number;
}> = ({ stressPreset, radius = 3.4 }) => {
  const lineSegments = useMemo(() => {
    const isStress = stressPreset !== "EQUILIBRIUM";
    const positions: number[] = [];
    const colors: number[] = [];

    const numAssets = ASSETS_20D.length;
    for (let i = 0; i < numAssets; i++) {
      const angleI = (i / numAssets) * Math.PI * 2 - Math.PI / 2;
      const xi = Math.cos(angleI) * radius;
      const zi = Math.sin(angleI) * radius;

      for (let j = i + 1; j < numAssets; j++) {
        const catI = ASSETS_20D[i].category;
        const catJ = ASSETS_20D[j].category;

        // Same category or high beta coupling
        const isCorrelated = catI === catJ || (catI === "MegaCap" && catJ === "Indices");

        if (isCorrelated) {
          const angleJ = (j / numAssets) * Math.PI * 2 - Math.PI / 2;
          const xj = Math.cos(angleJ) * radius;
          const zj = Math.sin(angleJ) * radius;

          // Arc through center with slight dip
          const midX = (xi + xj) * 0.45;
          const midZ = (zi + zj) * 0.45;
          const midY = isStress ? -0.3 : -0.5;

          // Line segment 1: I to Mid
          positions.push(xi, -0.6, zi, midX, midY, midZ);
          // Line segment 2: Mid to J
          positions.push(midX, midY, midZ, xj, -0.6, zj);

          const r = isStress ? 0.95 : 0.15;
          const g = isStress ? 0.35 : 0.65;
          const b = isStress ? 0.2 : 0.85;

          colors.push(r, g, b, r, g, b);
          colors.push(r, g, b, r, g, b);
        }
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return geom;
  }, [stressPreset, radius]);

  return (
    <lineSegments geometry={lineSegments}>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={stressPreset !== "EQUILIBRIUM" ? 0.45 : 0.18}
        linewidth={1}
      />
    </lineSegments>
  );
};

/**
 * Main 3D Scenario Component for 20D Financial Risk Engine
 */
export const FinancialRisk20DScenario: React.FC<FinancialRisk20DScenarioProps> = ({
  engine,
  axisX,
  axisY,
  state20D,
  volShock,
  stressPreset,
  showWireframe,
  showContourLines,
  showGreeksVectors,
  showCorrelationWeb,
  onTelemetryUpdate,
}) => {
  // Precompute 2D Surface slice along Axis X and Axis Y
  const surfaceData = useMemo(() => {
    const res = engine.evaluate2DSurfaceGrid(axisX, axisY, state20D, 36);
    res.minZ = Math.min(...res.zValues);
    res.maxZ = Math.max(...res.zValues);
    return res;
  }, [engine, axisX, axisY, state20D, volShock, stressPreset]);

  // Compute instantaneous full risk telemetry
  const telemetry = useMemo(() => {
    return engine.computeRiskTelemetry(state20D, volShock, stressPreset, axisX, axisY);
  }, [engine, state20D, volShock, stressPreset, axisX, axisY]);

  // Push telemetry update to parent
  useEffect(() => {
    onTelemetryUpdate(telemetry);
  }, [telemetry, onTelemetryUpdate]);

  const currentVal = telemetry.portfolioValueM;
  const currentDeltaX = telemetry.greeks[axisX]?.delta || 0.0;
  const currentDeltaY = telemetry.greeks[axisY]?.delta || 0.0;
  const currentGammaX = telemetry.greeks[axisX]?.gamma || 0.0;
  const currentGammaY = telemetry.greeks[axisY]?.gamma || 0.0;

  const deltasArray = useMemo(() => {
    const arr = new Float64Array(engine.D);
    telemetry.greeks.forEach((g, i) => {
      arr[i] = g.delta;
    });
    return arr;
  }, [engine.D, telemetry.greeks]);

  const gammasArray = useMemo(() => {
    const arr = new Float64Array(engine.D);
    telemetry.greeks.forEach((g, i) => {
      arr[i] = g.gamma;
    });
    return arr;
  }, [engine.D, telemetry.greeks]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas shadows={false} gl={{ antialias: true, alpha: true }}>
        <PerspectiveCamera makeDefault position={[4.5, 3.8, 5.2]} fov={45} />
        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={2.5}
          maxDistance={12.0}
          maxPolarAngle={Math.PI / 2 - 0.02}
        />

        <ambientLight intensity={0.65} />
        <directionalLight position={[6, 10, 8]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-4, 6, -4]} intensity={0.6} color="#06b6d4" />
        <pointLight position={[0, -2, 0]} intensity={0.4} color="#f59e0b" />

        {/* 3D Risk Hypersurface Slice */}
        <RiskHypersurfaceMesh
          surfaceData={surfaceData}
          showWireframe={showWireframe}
          showContourLines={showContourLines}
        />

        {/* Current Portfolio Operating Point with Tangent Gradient & Curvature */}
        <PortfolioOperatingPoint
          currentX={state20D[axisX]}
          currentY={state20D[axisY]}
          currentValue={currentVal}
          deltaX={currentDeltaX}
          deltaY={currentDeltaY}
          gammaX={currentGammaX}
          gammaY={currentGammaY}
          surfaceData={surfaceData}
          showVectors={showGreeksVectors}
        />

        {/* 20D Spatial Asset Perimeter Radar */}
        <SpatialAssetPerimeterRing
          state20D={state20D}
          deltas={deltasArray}
          gammas={gammasArray}
          axisX={axisX}
          axisY={axisY}
          radius={3.4}
        />

        {/* Cross-Asset Correlation Contagion Web */}
        {showCorrelationWeb && (
          <CorrelationContagionWeb
            stressPreset={stressPreset}
            radius={3.4}
          />
        )}

        {/* Axis Labels in 3D */}
        <Text
          position={[2.5, -0.7, 0]}
          fontSize={0.14}
          color="#06b6d4"
          anchorX="center"
          anchorY="middle"
        >
          {`X: ${ASSETS_20D[axisX].symbol} Return (Δ = ${currentDeltaX >= 0 ? "+" : ""}${currentDeltaX.toFixed(2)})`}
        </Text>

        <Text
          position={[0, -0.7, 2.5]}
          fontSize={0.14}
          color="#10b981"
          anchorX="center"
          anchorY="middle"
          rotation={[0, Math.PI / 2, 0]}
        >
          {`Y: ${ASSETS_20D[axisY].symbol} Return (Δ = ${currentDeltaY >= 0 ? "+" : ""}${currentDeltaY.toFixed(2)})`}
        </Text>

        {/* Floor Grid Plane */}
        <gridHelper args={[8, 24, "#1e293b", "#0f172a"]} position={[0, -0.65, 0]} />
      </Canvas>
    </div>
  );
};
