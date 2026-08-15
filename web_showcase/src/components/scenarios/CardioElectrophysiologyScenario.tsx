import React, { useRef, useMemo, useEffect, useState, useCallback } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  CardioElectrophysiologyEngine,
  type CardioTelemetry,
  type CardioRhythmPreset,
  type ECGLead,
} from "../../engine/cardioElectrophysiologyEngine";
import { Flame } from "lucide-react";

export interface CardioElectrophysiologyScenarioProps {
  rhythmPreset: CardioRhythmPreset;
  conductionVelocity: number;
  excitability: number;
  actionPotentialDuration: number;
  anisotropyRatio: number;
  ablationRadiusMm: number;
  activeLead: ECGLead;
  showFiberVectors: boolean;
  showVcGDipole: boolean;
  showAblationScars: boolean;
  onTelemetryUpdate: (telemetry: CardioTelemetry) => void;
  engineRef?: React.MutableRefObject<CardioElectrophysiologyEngine | null>;
}

/**
 * 3D Biventricular Heart Surface with Real-Time Action Potential Wave Propagation
 */
const BiventricularHeartMesh: React.FC<{
  engine: CardioElectrophysiologyEngine;
  onAblatePoint: (point: THREE.Vector3) => void;
}> = ({ engine, onAblatePoint }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const nodes = engine.nodes;
  const numNodes = nodes.length;

  const { geometry } = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(numNodes * 3);
    const col = new Float32Array(numNodes * 3);
    const norm = new Float32Array(numNodes * 3);

    for (let i = 0; i < numNodes; i++) {
      const node = nodes[i];
      const i3 = i * 3;
      pos[i3] = node.x;
      pos[i3 + 1] = node.y;
      pos[i3 + 2] = node.z;

      // Base normal pointing outwards from long axis
      const distXZ = Math.sqrt(node.x * node.x + node.z * node.z) || 1.0;
      norm[i3] = node.x / distXZ;
      norm[i3 + 1] = 0.1;
      norm[i3 + 2] = node.z / distXZ;

      // Initial color (resting myocardium)
      col[i3] = 0.12;
      col[i3 + 1] = 0.10;
      col[i3 + 2] = 0.25;
    }

    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(norm, 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint16Array(engine.triangles), 1));
    geom.computeVertexNormals();
    geom.computeBoundingBox();
    geom.computeBoundingSphere();

    return { geometry: geom, colors: col };
  }, [engine, nodes, numNodes]);

  // Color Mapping Helper for Transmembrane Potential v(x, t)
  const mapPotentialToColor = (
    v: number,
    w: number,
    isAblated: boolean,
    outRgb: [number, number, number]
  ): void => {
    if (isAblated) {
      // Necrotic Ablation Scar: Charcoal core with incandescent border
      outRgb[0] = 0.16;
      outRgb[1] = 0.13;
      outRgb[2] = 0.15;
      return;
    }

    if (v < 0.15) {
      // Resting Myocardium: Rich anatomical ruby-burgundy with subtle depth glow
      const t = v / 0.15;
      outRgb[0] = 0.42 + 0.25 * t;
      outRgb[1] = 0.15 + 0.20 * t;
      outRgb[2] = 0.28 + 0.22 * t + 0.1 * w;
    } else if (v < 0.60) {
      // Depolarization & Plateau: Intense Radiant Amber-Gold to Electric Orange
      const t = (v - 0.15) / 0.45;
      outRgb[0] = 0.67 + 0.33 * t;
      outRgb[1] = 0.35 + 0.45 * t;
      outRgb[2] = 0.50 * (1.0 - t) + 0.05 * t;
    } else {
      // Wavefront Peak: Blazing Neon Cyan / Electric Blue Fire
      const t = Math.min(1.0, (v - 0.60) / 0.40);
      outRgb[0] = 0.90 * (1.0 - t) + 0.10 * t;
      outRgb[1] = 0.80 + 0.20 * t;
      outRgb[2] = 0.20 + 0.80 * t;
    }
  };

  useFrame((_, delta) => {
    if (!meshRef.current) return;

    // Step electrophysiology PDE solver at 120 FPS
    engine.step(Math.min(0.033, delta));

    const colorAttr = meshRef.current.geometry.attributes.color as THREE.BufferAttribute;
    if (!colorAttr) return;

    const colArray = colorAttr.array as Float32Array;
    const tempRgb: [number, number, number] = [0, 0, 0];

    for (let i = 0; i < numNodes; i++) {
      const node = nodes[i];
      mapPotentialToColor(node.v, node.w, node.isAblated, tempRgb);
      const i3 = i * 3;
      colArray[i3] = tempRgb[0];
      colArray[i3 + 1] = tempRgb[1];
      colArray[i3 + 2] = tempRgb[2];
    }

    colorAttr.needsUpdate = true;
  });

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    if (e.point) {
      onAblatePoint(e.point);
    }
  };

  return (
    <group>
      <mesh
        ref={meshRef}
        geometry={geometry}
        onPointerDown={handlePointerDown}
        frustumCulled={false}
        castShadow
        receiveShadow
      >
        <meshStandardMaterial
          vertexColors
          roughness={0.35}
          metalness={0.20}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh geometry={geometry} frustumCulled={false}>
        <meshBasicMaterial
          color="#38bdf8"
          wireframe
          transparent
          opacity={0.08}
        />
      </mesh>
    </group>
  );
};

/**
 * Visual Indicators for RF Catheter Ablation Scars & Thermal Halo
 */
const AblationScarsVisualizer: React.FC<{
  engine: CardioElectrophysiologyEngine;
}> = ({ engine }) => {
  return (
    <group>
      {engine.scars.map((scar) => (
        <group key={scar.id} position={[scar.x, scar.y, scar.z]}>
          {/* Cauterized central necrotic lesion */}
          <mesh>
            <sphereGeometry args={[scar.radius * 0.75, 16, 16]} />
            <meshStandardMaterial
              color="#2a1b24"
              emissive="#ff4422"
              emissiveIntensity={0.35}
              roughness={0.9}
            />
          </mesh>
          {/* Thermal glow perimeter ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[scar.radius * 0.8, scar.radius * 1.15, 24]} />
            <meshBasicMaterial
              color="#f97316"
              side={THREE.DoubleSide}
              transparent
              opacity={0.8}
            />
          </mesh>
        </group>
      ))}
    </group>
  );
};

/**
 * 3D Vectorcardiogram (VCG) Dipole Vector Arrow Rotating at Cardiac Center
 */
const VCGDipoleVectorIndicator: React.FC<{
  engine: CardioElectrophysiologyEngine;
}> = ({ engine }) => {
  const arrowGroupRef = useRef<THREE.Group>(null);
  const origin: [number, number, number] = [0, -0.25, 0];

  useFrame(() => {
    if (!arrowGroupRef.current) return;
    const [px, py, pz] = engine.dipoleVector;
    const mag = engine.dipoleMagnitude;

    const dir = new THREE.Vector3(px, py, pz);
    if (mag > 1e-4) {
      dir.normalize();
      const targetQuat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir
      );
      arrowGroupRef.current.quaternion.slerp(targetQuat, 0.25);
      const scale = Math.max(0.1, Math.min(2.5, mag * 2.8));
      arrowGroupRef.current.scale.set(1.0, scale, 1.0);
    }
  });

  return (
    <group position={origin}>
      {/* Centroid Sphere */}
      <mesh>
        <sphereGeometry args={[0.035, 16, 16]} />
        <meshStandardMaterial
          color="#06b6d4"
          emissive="#06b6d4"
          emissiveIntensity={0.8}
        />
      </mesh>
      {/* Dynamic 3D Dipole Vector Arrow */}
      <group ref={arrowGroupRef}>
        <mesh position={[0, 0.22, 0]}>
          <cylinderGeometry args={[0.012, 0.012, 0.44, 12]} />
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#0284c7"
            emissiveIntensity={0.9}
          />
        </mesh>
        <mesh position={[0, 0.46, 0]}>
          <coneGeometry args={[0.035, 0.10, 12]} />
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#0284c7"
            emissiveIntensity={1.0}
          />
        </mesh>
      </group>
      <Text
        position={[0, -0.09, 0]}
        fontSize={0.05}
        color="#38bdf8"
        anchorX="center"
        anchorY="top"
      >
        3D DIPOLE P(t)
      </Text>
    </group>
  );
};

/**
 * Chebyshev Continuous Myocardial Fiber Streamlines
 */
const FiberStreamlinesMesh: React.FC<{
  engine: CardioElectrophysiologyEngine;
}> = ({ engine }) => {
  const lineSegments = useMemo(() => {
    const lines: THREE.Vector3[][] = [];
    const sampleStep = 3;

    for (let i = 0; i < engine.nodes.length; i += sampleStep) {
      const node = engine.nodes[i];
      if (node.isAblated) continue;

      const p0 = new THREE.Vector3(node.x, node.y, node.z);
      const len = 0.06;
      const p1 = new THREE.Vector3(
        node.x + node.fx * len,
        node.y + node.fy * len,
        node.z + node.fz * len
      );
      lines.push([p0, p1]);
    }

    const group = new THREE.Group();
    lines.forEach(([p0, p1]) => {
      const geom = new THREE.BufferGeometry().setFromPoints([p0, p1]);
      const mat = new THREE.LineBasicMaterial({
        color: "#06b6d4",
        transparent: true,
        opacity: 0.35,
      });
      const line = new THREE.Line(geom, mat);
      group.add(line);
    });

    return group;
  }, [engine]);

  return <primitive object={lineSegments} />;
};

/**
 * Real-Time 120 FPS Medical EKG Oscilloscope Screen (Millimeter Paper Grid)
 */
const RealTimeEkgOscilloscope: React.FC<{
  engine: CardioElectrophysiologyEngine;
  activeLead: ECGLead;
}> = ({ engine, activeLead }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    const renderOscilloscope = () => {
      const width = canvas.width;
      const height = canvas.height;

      // 1. Background Grid (Medical Millimeter Paper standard: 25 mm/s, 10 mm/mV)
      ctx.fillStyle = "#040b0f";
      ctx.fillRect(0, 0, width, height);

      // Minor grid (1 mm)
      ctx.strokeStyle = "rgba(6, 182, 212, 0.08)";
      ctx.lineWidth = 1;
      const gridMinor = 12;
      for (let x = 0; x < width; x += gridMinor) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridMinor) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Major grid (5 mm)
      ctx.strokeStyle = "rgba(6, 182, 212, 0.22)";
      ctx.lineWidth = 1.2;
      const gridMajor = gridMinor * 5;
      for (let x = 0; x < width; x += gridMajor) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridMajor) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // 2. Draw Rolling EKG Waveform
      const bufferLen = engine.ecgBufferLength;
      const voltages = engine.ecgVoltageBuffer;
      const head = engine.ecgBufferHead;
      const midY = height * 0.55;
      const scaleY = (gridMajor / 1.0) * 1.35; // 1 mV = 1 major box

      ctx.beginPath();
      ctx.strokeStyle = engine.rhythm === "VF" || engine.rhythm === "VT" ? "#ef4444" : "#10b981";
      ctx.lineWidth = 2.0;
      ctx.shadowColor = engine.rhythm === "VF" || engine.rhythm === "VT" ? "rgba(239, 68, 68, 0.8)" : "rgba(16, 185, 129, 0.8)";
      ctx.shadowBlur = 6;

      for (let i = 0; i < bufferLen; i++) {
        const bufIdx = (head + i) % bufferLen;
        const x = (i / (bufferLen - 1)) * width;
        const v = voltages[bufIdx];
        const y = midY - v * scaleY;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // 3. Sweep Bar indicator
      const sweepX = width - 4;
      ctx.fillStyle = "rgba(56, 189, 248, 0.7)";
      ctx.fillRect(sweepX, 0, 3, height);

      animId = requestAnimationFrame(renderOscilloscope);
    };

    animId = requestAnimationFrame(renderOscilloscope);
    return () => cancelAnimationFrame(animId);
  }, [engine]);

  const tel = engine.getTelemetry();

  return (
    <div
      style={{
        position: "absolute",
        bottom: "18px",
        right: "18px",
        width: "480px",
        background: "rgba(6, 10, 18, 0.92)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "10px",
        padding: "12px 14px",
        backdropFilter: "blur(14px)",
        boxShadow: "0 12px 36px rgba(0, 0, 0, 0.65)",
        zIndex: 25,
      }}
    >
      {/* Header with BPM & Rhythm Alert */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "8px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            style={{
              width: "10px",
              height: "10px",
              borderRadius: "50%",
              background: tel.isArrhythmia ? "#ef4444" : "#10b981",
              boxShadow: tel.isArrhythmia ? "0 0 10px #ef4444" : "0 0 10px #10b981",
              animation: tel.isArrhythmia ? "pulse 0.8s infinite ease-in-out" : "pulse 2s infinite ease-in-out",
            }}
          ></span>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "11px",
              fontWeight: 700,
              color: tel.isArrhythmia ? "#ef4444" : "var(--text-main)",
              letterSpacing: "0.06em",
            }}
          >
            REAL-TIME 12-LEAD EKG &bull; LEAD {activeLead}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "14px",
              fontWeight: 700,
              color: tel.isArrhythmia ? "#ef4444" : "#10b981",
            }}
          >
            {tel.heartRateBpm}{" "}
            <span style={{ fontSize: "10px", fontWeight: 400, color: "var(--text-dim)" }}>
              BPM
            </span>
          </span>
          <span
            className="badge-zero-epochs"
            style={{
              background: tel.isArrhythmia ? "rgba(239, 68, 68, 0.18)" : "rgba(16, 185, 129, 0.18)",
              color: tel.isArrhythmia ? "#ef4444" : "#10b981",
              borderColor: tel.isArrhythmia ? "rgba(239, 68, 68, 0.4)" : "rgba(16, 185, 129, 0.4)",
            }}
          >
            {tel.rhythmName}
          </span>
        </div>
      </div>

      {/* Canvas Oscilloscope */}
      <div style={{ position: "relative", width: "100%", height: "135px", borderRadius: "6px", overflow: "hidden", border: "1px solid rgba(6, 182, 212, 0.2)" }}>
        <canvas
          ref={canvasRef}
          width={452}
          height={135}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
        {/* Scale labels */}
        <div
          style={{
            position: "absolute",
            bottom: "4px",
            left: "8px",
            fontSize: "9px",
            fontFamily: "var(--font-mono)",
            color: "rgba(6, 182, 212, 0.6)",
            pointerEvents: "none",
          }}
        >
          25 mm/s &bull; 10 mm/mV &bull; 120 FPS
        </div>
      </div>
    </div>
  );
};

export const CardioElectrophysiologyScenario: React.FC<CardioElectrophysiologyScenarioProps> = ({
  rhythmPreset,
  conductionVelocity,
  excitability,
  actionPotentialDuration,
  anisotropyRatio,
  ablationRadiusMm,
  activeLead,
  showFiberVectors,
  showVcGDipole,
  showAblationScars,
  onTelemetryUpdate,
  engineRef,
}) => {
  const engine = useMemo(() => {
    const eng = new CardioElectrophysiologyEngine();
    if (engineRef) {
      engineRef.current = eng;
    }
    return eng;
  }, [engineRef]);

  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Sync Parameters to Engine
  useEffect(() => {
    engine.setRhythmPreset(rhythmPreset);
  }, [engine, rhythmPreset]);

  useEffect(() => {
    engine.sigmaFiber = conductionVelocity;
    engine.sigmaCross = conductionVelocity / Math.max(1.0, anisotropyRatio);
    engine.k = excitability;
    engine.eps0 = 0.008 * (240.0 / Math.max(100, actionPotentialDuration));
    engine.activeLead = activeLead;
  }, [engine, conductionVelocity, anisotropyRatio, excitability, actionPotentialDuration, activeLead]);

  // Handle User Click on Heart Surface for RF Ablation
  const handleAblatePoint = useCallback(
    (point: THREE.Vector3) => {
      const radius = (ablationRadiusMm / 10.0) * 0.15; // mm to world coordinate
      const success = engine.ablateAt(point.x, point.y, point.z, radius);
      if (success) {
        setToastMessage(`RF ABLATION APPLIED (${engine.scars.length} LESIONS) • CONDUCTIVITY BLOCKED`);
        setTimeout(() => setToastMessage(null), 2500);
      }
    },
    [engine, ablationRadiusMm]
  );

  // Telemetry updates loop
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();

    const loop = () => {
      const now = performance.now();
      if (now - lastTime >= 120) {
        onTelemetryUpdate(engine.getTelemetry());
        lastTime = now;
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [engine, onTelemetryUpdate]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas
        gl={{
          antialias: true,
          powerPreference: "high-performance",
        }}
      >
        <PerspectiveCamera makeDefault position={[0.4, -0.05, 2.5]} fov={45} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          target={[0, -0.2, 0]}
          maxDistance={6.0}
          minDistance={1.0}
        />

        {/* Ambient & Directional Lighting */}
        <ambientLight intensity={0.65} />
        <directionalLight position={[3, 4, 3]} intensity={1.4} />
        <directionalLight position={[-3, -2, -2]} intensity={0.4} color="#38bdf8" />
        <pointLight position={[0, 0, 1.5]} intensity={0.5} color="#06b6d4" />

        {/* 1. 3D Biventricular Heart Surface */}
        <BiventricularHeartMesh
          engine={engine}
          onAblatePoint={handleAblatePoint}
        />

        {/* 2. RF Ablation Scars */}
        {showAblationScars && <AblationScarsVisualizer engine={engine} />}

        {/* 3. 3D VCG Dipole Vector */}
        {showVcGDipole && <VCGDipoleVectorIndicator engine={engine} />}

        {/* 4. Chebyshev Fiber Orientation Glyphs */}
        {showFiberVectors && <FiberStreamlinesMesh engine={engine} />}
      </Canvas>

      {/* Real-Time EKG Oscilloscope Screen */}
      <RealTimeEkgOscilloscope engine={engine} activeLead={activeLead} />

      {/* Interactive Helper Banner on Top-Right */}
      <div
        style={{
          position: "absolute",
          top: "18px",
          right: "18px",
          background: "rgba(10, 15, 26, 0.85)",
          backdropFilter: "blur(8px)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "8px",
          padding: "8px 14px",
          fontSize: "11px",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        <Flame size={14} color="#f97316" />
        <span>
          <b style={{ color: "var(--amber-primary)" }}>CLICK HEART WALL</b> to apply RF Catheter Ablation &amp; Terminate Reentry
        </span>
      </div>

      {/* Toast notification on ablation */}
      {toastMessage && (
        <div
          style={{
            position: "absolute",
            top: "60px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(239, 68, 68, 0.9)",
            color: "#fff",
            border: "1px solid rgba(255, 255, 255, 0.2)",
            borderRadius: "6px",
            padding: "8px 18px",
            fontSize: "11.5px",
            fontWeight: 700,
            fontFamily: "var(--font-mono)",
            boxShadow: "0 8px 24px rgba(239, 68, 68, 0.4)",
            zIndex: 30,
            pointerEvents: "none",
          }}
        >
          {toastMessage}
        </div>
      )}
    </div>
  );
};
