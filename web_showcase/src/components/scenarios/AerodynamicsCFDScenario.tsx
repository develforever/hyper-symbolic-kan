import React, { useRef, useMemo, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  AerodynamicsCFDEngine,
  type CFDSolverResult,
  type NACAProfileConfig,
  type AirfoilSurfacePoint,
} from "../../engine/aerodynamicsCfdEngine";

export interface AerodynamicsCFDScenarioProps {
  aoaDeg: number;
  uInf: number;
  airfoilConfig: NACAProfileConfig;
  showStreamlines: boolean;
  showPressureMap: boolean;
  showVectors: boolean;
  showSmokeParticles: boolean;
  streamlineDensity: number;
  onTelemetryUpdate: (result: CFDSolverResult) => void;
}

/**
 * 3D Wytłaczane Skrzydło NACA z dynamicznym cieniowaniem współczynnikiem ciśnienia C_p na wierzchołkach
 */
const NACA3DWing: React.FC<{
  surfacePoints: AirfoilSurfacePoint[];
  span?: number;
  showPressureMap: boolean;
}> = ({ surfacePoints, span = 1.6, showPressureMap }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  const wingGeometry = useMemo(() => {
    if (!surfacePoints || surfacePoints.length < 3) return new THREE.BufferGeometry();

    const N = surfacePoints.length;
    const halfSpan = span / 2;
    const spanSegments = 16;
    const totalVertices = (spanSegments + 1) * N + 2 * N;
    const positions = new Float32Array(totalVertices * 3);
    const colors = new Float32Array(totalVertices * 3);
    const normals = new Float32Array(totalVertices * 3);
    const indices: number[] = [];

    // Helper: Map Cp to RGB color
    const cpToColor = (cp: number): [number, number, number] => {
      if (!showPressureMap) {
        return [0.15, 0.23, 0.35]; // Carbon dark slate
      }
      if (cp <= 0) {
        // Suction (Negative Pressure): Deep cyan / sky blue / brilliant azure
        const t = Math.min(1.0, Math.abs(cp) / 2.5);
        return [
          0.05 + 0.15 * (1 - t),
          0.45 + 0.50 * t,
          0.85 + 0.15 * t,
        ];
      } else {
        // Positive Pressure / Stagnation: Amber to vibrant crimson red
        const t = Math.min(1.0, cp / 1.0);
        return [
          0.92 + 0.08 * t,
          0.42 * (1 - t),
          0.12 * (1 - t),
        ];
      }
    };

    let vertOffset = 0;

    // 1. Generate body vertices along span
    for (let s = 0; s <= spanSegments; s++) {
      const zFrac = s / spanSegments;
      const z = -halfSpan + zFrac * span;

      for (let i = 0; i < N; i++) {
        const pt = surfacePoints[i];
        const idx3 = vertOffset * 3;

        positions[idx3] = pt.x;
        positions[idx3 + 1] = pt.y;
        positions[idx3 + 2] = z;

        const [r, g, b] = cpToColor(pt.cp);
        colors[idx3] = r;
        colors[idx3 + 1] = g;
        colors[idx3 + 2] = b;

        normals[idx3] = pt.nx;
        normals[idx3 + 1] = pt.ny;
        normals[idx3 + 2] = 0;

        vertOffset++;
      }
    }

    // Body indices
    for (let s = 0; s < spanSegments; s++) {
      const rowA = s * N;
      const rowB = (s + 1) * N;

      for (let i = 0; i < N; i++) {
        const nextI = (i + 1) % N;
        const p1 = rowA + i;
        const p2 = rowA + nextI;
        const p3 = rowB + i;
        const p4 = rowB + nextI;

        indices.push(p1, p3, p2);
        indices.push(p2, p3, p4);
      }
    }

    // 2. Endcap Left (z = -halfSpan)
    const capLeftStart = vertOffset;
    for (let i = 0; i < N; i++) {
      const pt = surfacePoints[i];
      const idx3 = vertOffset * 3;
      positions[idx3] = pt.x;
      positions[idx3 + 1] = pt.y;
      positions[idx3 + 2] = -halfSpan;

      const [r, g, b] = cpToColor(pt.cp);
      colors[idx3] = r * 0.75;
      colors[idx3 + 1] = g * 0.75;
      colors[idx3 + 2] = b * 0.75;

      normals[idx3] = 0;
      normals[idx3 + 1] = 0;
      normals[idx3 + 2] = -1;
      vertOffset++;
    }

    // Triangulate Left Endcap
    for (let i = 1; i < N - 1; i++) {
      indices.push(capLeftStart, capLeftStart + i + 1, capLeftStart + i);
    }

    // 3. Endcap Right (z = +halfSpan)
    const capRightStart = vertOffset;
    for (let i = 0; i < N; i++) {
      const pt = surfacePoints[i];
      const idx3 = vertOffset * 3;
      positions[idx3] = pt.x;
      positions[idx3 + 1] = pt.y;
      positions[idx3 + 2] = halfSpan;

      const [r, g, b] = cpToColor(pt.cp);
      colors[idx3] = r * 0.75;
      colors[idx3 + 1] = g * 0.75;
      colors[idx3 + 2] = b * 0.75;

      normals[idx3] = 0;
      normals[idx3 + 1] = 0;
      normals[idx3 + 2] = 1;
      vertOffset++;
    }

    // Triangulate Right Endcap
    for (let i = 1; i < N - 1; i++) {
      indices.push(capRightStart, capRightStart + i, capRightStart + i + 1);
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();

    return geom;
  }, [surfacePoints, span, showPressureMap]);

  return (
    <mesh ref={meshRef} geometry={wingGeometry}>
      <meshStandardMaterial
        vertexColors
        roughness={0.2}
        metalness={0.7}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

/**
 * 3D Analityczne Linie Prądu (Streamlines)
 */
const StreamlinesMesh: React.FC<{
  streamlines: Array<Array<[number, number]>>;
  span?: number;
}> = ({ streamlines, span = 1.6 }) => {
  const lineGroup = useMemo(() => {
    const group = new THREE.Group();
    const spanSlices = [-0.65, -0.32, 0.0, 0.32, 0.65];

    spanSlices.forEach((zVal) => {
      const isCenter = zVal === 0.0;

      streamlines.forEach((linePts) => {
        if (linePts.length < 2) return;

        const pts3d = linePts.map((p) => new THREE.Vector3(p[0], p[1], zVal));
        const curve = new THREE.CatmullRomCurve3(pts3d);
        const geom = new THREE.TubeGeometry(
          curve,
          Math.min(90, linePts.length),
          isCenter ? 0.004 : 0.0028,
          6,
          false
        );
        const mat = new THREE.MeshBasicMaterial({
          color: isCenter ? "#38bdf8" : "#0284c7",
          transparent: true,
          opacity: isCenter ? 0.9 : 0.4,
        });
        const mesh = new THREE.Mesh(geom, mat);
        group.add(mesh);
      });
    });

    return group;
  }, [streamlines, span]);

  return <primitive object={lineGroup} />;
};

/**
 * Dynamiczne Cząstki Dymu (Smoke Streaklines) w Tunelu Aerodynamicznym
 */
const SmokeParticlesSystem: React.FC<{
  engine: AerodynamicsCFDEngine;
}> = ({ engine }) => {
  const pointsRef = useRef<THREE.Points>(null);
  const particleCount = engine.particles.length;

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const col = new Float32Array(particleCount * 3);
    return [pos, col];
  }, [particleCount]);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;

    engine.stepParticles(Math.min(delta, 0.033), 1.25);

    const posAttr = pointsRef.current.geometry.attributes.position as THREE.BufferAttribute;
    const colAttr = pointsRef.current.geometry.attributes.color as THREE.BufferAttribute;
    const pts = engine.particles;

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const i3 = i * 3;

      positions[i3] = p.x;
      positions[i3 + 1] = p.y;
      positions[i3 + 2] = p.z;

      const speedRatio = p.speed / Math.max(1.0, engine.uInf);
      if (speedRatio > 1.1) {
        colors[i3] = 0.4;
        colors[i3 + 1] = 0.95;
        colors[i3 + 2] = 1.0;
      } else if (speedRatio < 0.75) {
        colors[i3] = 0.95;
        colors[i3 + 1] = 0.55;
        colors[i3 + 2] = 0.15;
      } else {
        colors[i3] = 0.2;
        colors[i3 + 1] = 0.75;
        colors[i3 + 2] = 0.95;
      }
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        vertexColors
        transparent
        opacity={0.85}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
};

/**
 * Wizualizacja Wektorów Sił Aerodynamicznych (Lift CL & Drag CD)
 */
const AerodynamicForceVectors: React.FC<{
  cl: number;
  cd: number;
  stagnationPoint: [number, number];
  stagnationCp: number;
  origin?: [number, number, number];
}> = ({ cl, cd, stagnationPoint, stagnationCp, origin = [0, 0, 0] }) => {
  const liftScale = 0.35;
  const dragScale = 0.95;

  const liftLen = Math.max(0.06, Math.abs(cl) * liftScale);
  const dragLen = Math.max(0.06, Math.abs(cd) * dragScale);

  const liftDir = cl >= 0 ? 1 : -1;

  return (
    <group position={origin}>
      {/* 1. Wektor Siły Nośnej Lift CL (Pionowy w punkcie c/4) */}
      <group position={[0, 0.02, 0]}>
        <mesh position={[0, (liftDir * liftLen) / 2, 0]}>
          <cylinderGeometry args={[0.01, 0.01, liftLen, 12]} />
          <meshStandardMaterial color="#10b981" emissive="#059669" emissiveIntensity={0.6} />
        </mesh>
        <mesh position={[0, liftDir * liftLen, 0]} rotation={[liftDir < 0 ? Math.PI : 0, 0, 0]}>
          <coneGeometry args={[0.03, 0.09, 12]} />
          <meshStandardMaterial color="#10b981" emissive="#059669" emissiveIntensity={0.8} />
        </mesh>
        <Text
          position={[0, liftDir * liftLen + 0.08 * liftDir, 0]}
          fontSize={0.075}
          color="#10b981"
          anchorX="center"
          anchorY="middle"
        >
          {`LIFT C_L = ${cl >= 0 ? `+${cl.toFixed(2)}` : cl.toFixed(2)}`}
        </Text>
      </group>

      {/* 2. Wektor Oporu Drag CD (Poziomy w punkcie spływu c/4) */}
      <group position={[0.45, 0, 0]}>
        <mesh position={[dragLen / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <cylinderGeometry args={[0.008, 0.008, dragLen, 12]} />
          <meshStandardMaterial color="#ef4444" emissive="#dc2626" emissiveIntensity={0.6} />
        </mesh>
        <mesh position={[dragLen, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <coneGeometry args={[0.025, 0.07, 12]} />
          <meshStandardMaterial color="#ef4444" emissive="#dc2626" emissiveIntensity={0.8} />
        </mesh>
        <Text
          position={[dragLen + 0.12, 0.02, 0]}
          fontSize={0.06}
          color="#ef4444"
          anchorX="left"
          anchorY="middle"
        >
          {`DRAG C_D = ${cd.toFixed(3)}`}
        </Text>
      </group>

      {/* 3. Znacznik Punktu Spiętrzenia (Stagnation Point) */}
      <group position={[stagnationPoint[0], stagnationPoint[1], 0]}>
        <mesh>
          <sphereGeometry args={[0.024, 16, 16]} />
          <meshStandardMaterial color="#f59e0b" emissive="#d97706" emissiveIntensity={0.9} />
        </mesh>
        <mesh>
          <sphereGeometry args={[0.048, 12, 12]} />
          <meshBasicMaterial color="#f59e0b" wireframe transparent opacity={0.35} />
        </mesh>
        <Text
          position={[0, -0.07, 0]}
          fontSize={0.048}
          color="#f59e0b"
          anchorX="center"
        >
          {`STAGNATION (Cp = +${stagnationCp.toFixed(2)})`}
        </Text>
      </group>
    </group>
  );
};

/**
 * Szklana Obudowa Tunelu Aerodynamicznego i Dysza Napływowa
 */
const WindTunnelEnclosure: React.FC = () => {
  return (
    <group>
      {/* Szklana komora pomiarowa test section */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(3.2, 1.9, 1.8)]} />
        <lineBasicMaterial color="#0284c7" transparent opacity={0.3} />
      </lineSegments>

      {/* Podłoga z siatką pomocniczą */}
      <gridHelper args={[3.2, 16, "#06b6d4", "#1e293b"]} position={[0, -0.95, 0]} />

      {/* Górna siatka referencyjna */}
      <gridHelper args={[3.2, 16, "#0f172a", "#1e293b"]} position={[0, 0.95, 0]} />

      {/* Wlot powietrza po lewej */}
      <group position={[-1.6, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <mesh>
          <ringGeometry args={[0.85, 0.92, 32]} />
          <meshBasicMaterial color="#38bdf8" transparent opacity={0.4} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* Wylot tunelu po prawej */}
      <group position={[1.6, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <mesh>
          <ringGeometry args={[0.85, 0.92, 32]} />
          <meshBasicMaterial color="#0369a1" transparent opacity={0.25} side={THREE.DoubleSide} />
        </mesh>
      </group>
    </group>
  );
};

export const AerodynamicsCFDScenario: React.FC<AerodynamicsCFDScenarioProps> = ({
  aoaDeg,
  uInf,
  airfoilConfig,
  showStreamlines,
  showPressureMap,
  showVectors,
  showSmokeParticles,
  streamlineDensity,
  onTelemetryUpdate,
}) => {
  const engine = useMemo(() => {
    return new AerodynamicsCFDEngine(airfoilConfig, aoaDeg, uInf);
  }, []);

  const [solverResult, setSolverResult] = useState<CFDSolverResult>(() => engine.solve());
  const [streamlines, setStreamlines] = useState<Array<Array<[number, number]>>>(() =>
    engine.generateStreamlines(streamlineDensity)
  );

  // Synchronizacja parametrów wejściowych i natychmiastowe przeliczenie CFD w 0 epokach (< 2 ms)
  useEffect(() => {
    engine.aoaDeg = aoaDeg;
    engine.uInf = uInf;
    engine.airfoilConfig = { ...airfoilConfig };

    const res = engine.solve();
    setSolverResult(res);
    setStreamlines(engine.generateStreamlines(streamlineDensity));
    onTelemetryUpdate(res);
  }, [engine, aoaDeg, uInf, airfoilConfig, streamlineDensity, onTelemetryUpdate]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas
        gl={{
          antialias: true,
          powerPreference: "high-performance",
        }}
      >
        <PerspectiveCamera makeDefault position={[0.3, 0.85, 2.6]} fov={45} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          maxDistance={6.0}
          minDistance={0.8}
        />

        {/* Oświetlenie Tunelu */}
        <ambientLight intensity={0.55} />
        <directionalLight position={[3, 5, 4]} intensity={1.5} />
        <directionalLight position={[-3, -3, -3]} intensity={0.4} color="#38bdf8" />
        <pointLight position={[0, 0, 1.2]} intensity={0.4} color="#06b6d4" />

        {/* 1. Obudowa Tunelu Aerodynamicznego */}
        <WindTunnelEnclosure />

        {/* 2. Profil Skrzydła NACA 3D */}
        <NACA3DWing
          surfacePoints={solverResult.surfacePoints}
          showPressureMap={showPressureMap}
          span={1.5}
        />

        {/* 3. Linie Prądu (Streamlines) */}
        {showStreamlines && (
          <StreamlinesMesh streamlines={streamlines} span={1.5} />
        )}

        {/* 4. Cząstki Dymu (Smoke) */}
        {showSmokeParticles && (
          <SmokeParticlesSystem engine={engine} />
        )}

        {/* 5. Wektory Sił Lift & Drag */}
        {showVectors && (
          <AerodynamicForceVectors
            cl={solverResult.cl}
            cd={solverResult.cd}
            stagnationPoint={solverResult.stagnationPoint}
            stagnationCp={solverResult.stagnationCp}
          />
        )}
      </Canvas>

      {/* Pasek informacyjny na dole sceny */}
      <div
        style={{
          position: "absolute",
          bottom: "16px",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(10, 15, 26, 0.85)",
          backdropFilter: "blur(8px)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "8px",
          padding: "6px 14px",
          fontSize: "11px",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        <span>
          <b style={{ color: "var(--cyan-primary)" }}>{solverResult.airfoilName}</b> &bull; &alpha; ={" "}
          <b style={{ color: "#38bdf8" }}>{aoaDeg >= 0 ? `+${aoaDeg.toFixed(1)}` : aoaDeg.toFixed(1)}&deg;</b>
        </span>
        <span>&bull;</span>
        <span>
          <b style={{ color: "var(--emerald-primary)" }}>C_L: {solverResult.cl}</b> &bull;{" "}
          <b style={{ color: "var(--red-primary)" }}>C_D: {solverResult.cd}</b>
        </span>
        <span>&bull;</span>
        <span>
          <b style={{ color: "var(--amber-primary)" }}>L/D: {solverResult.glideRatio}</b>
        </span>
      </div>
    </div>
  );
};
