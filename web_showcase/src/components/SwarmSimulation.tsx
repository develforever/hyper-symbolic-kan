import React, { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { KanEvaluator } from "../engine/kanEvaluator";

interface SwarmSimulationProps {
  evaluator: KanEvaluator;
  numAgents: number;
  safetyGuardActive: boolean;
  flowSpeed: number;
  noiseAmount: number;
  onViolationCount?: (violations: number) => void;
}

export const SwarmSimulation: React.FC<SwarmSimulationProps> = ({
  evaluator,
  numAgents,
  safetyGuardActive,
  flowSpeed,
  noiseAmount,
  onViolationCount,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tempGrad = useMemo(() => new Float32Array(3), []);
  const tempColor = useMemo(() => new THREE.Color(), []);

  // Stan cząstek: pozycja (x, y, z), prędkość (vx, vy, vz)
  const agentStates = useMemo(() => {
    const states = new Float32Array(numAgents * 6);
    for (let i = 0; i < numAgents; i++) {
      const idx = i * 6;
      // Pozycja wewnątrz sześcianu [-0.85, 0.85]^3
      states[idx + 0] = (Math.random() - 0.5) * 1.7;
      states[idx + 1] = (Math.random() - 0.5) * 1.7;
      states[idx + 2] = (Math.random() - 0.5) * 1.7;
      // Prędkości początkowe
      states[idx + 3] = (Math.random() - 0.5) * 0.05;
      states[idx + 4] = (Math.random() - 0.5) * 0.05;
      states[idx + 5] = (Math.random() - 0.5) * 0.05;
    }
    return states;
  }, [numAgents]);

  const BOUND_LIMIT = 0.92;
  const NO_FLY_ZONE_RADIUS = 0.35;
  const NO_FLY_ZONE_CENTER = useMemo(() => new THREE.Vector3(-0.35, 0.2, 0.0), []);

  useFrame((_, delta) => {
    if (!meshRef.current) return;

    const dt = Math.min(delta, 0.05);
    let violations = 0;

    for (let i = 0; i < numAgents; i++) {
      const idx = i * 6;
      let px = agentStates[idx + 0];
      let py = agentStates[idx + 1];
      let pz = agentStates[idx + 2];
      let vx = agentStates[idx + 3];
      let vy = agentStates[idx + 4];
      let vz = agentStates[idx + 5];

      // 1. Obliczenie analitycznego gradientu pola KAN \nabla f(p)
      evaluator.gradient3D(px, py, pz, tempGrad);

      // Siła odpychania od wysokiego potencjału (Gradient Descent)
      const gx = tempGrad[0];
      const gy = tempGrad[1];
      const gz = tempGrad[2];

      const ax = -gx * flowSpeed * 1.5 + (Math.random() - 0.5) * noiseAmount;
      const ay = -gy * flowSpeed * 1.5 + (Math.random() - 0.5) * noiseAmount;
      const az = -gz * flowSpeed * 1.5 + (Math.random() - 0.5) * noiseAmount;

      // Całkowanie Eulera-Chromera
      vx = vx * 0.94 + ax * dt;
      vy = vy * 0.94 + ay * dt;
      vz = vz * 0.94 + az * dt;

      let nextPx = px + vx * dt * 5.0;
      let nextPy = py + vy * dt * 5.0;
      let nextPz = pz + vz * dt * 5.0;

      // Sprawdzenie naruszenia strefy No-Fly Zone
      const dx = nextPx - NO_FLY_ZONE_CENTER.x;
      const dy = nextPy - NO_FLY_ZONE_CENTER.y;
      const dz = nextPz - NO_FLY_ZONE_CENTER.z;
      const distToNoFly = Math.sqrt(dx * dx + dy * dy + dz * dz);

      const isBoundViolated =
        Math.abs(nextPx) > BOUND_LIMIT ||
        Math.abs(nextPy) > BOUND_LIMIT ||
        Math.abs(nextPz) > BOUND_LIMIT;

      const isNoFlyViolated = distToNoFly < NO_FLY_ZONE_RADIUS;

      if (isBoundViolated || isNoFlyViolated) {
        violations++;
      }

      // 2. Kategorialny Guard Bezpieczeństwa (MCT-NSE)
      if (safetyGuardActive) {
        // Projekcja barierowa na granice sześcianu
        if (nextPx > BOUND_LIMIT) { nextPx = BOUND_LIMIT; vx = -Math.abs(vx) * 0.5; }
        if (nextPx < -BOUND_LIMIT) { nextPx = -BOUND_LIMIT; vx = Math.abs(vx) * 0.5; }
        if (nextPy > BOUND_LIMIT) { nextPy = BOUND_LIMIT; vy = -Math.abs(vy) * 0.5; }
        if (nextPy < -BOUND_LIMIT) { nextPy = -BOUND_LIMIT; vy = Math.abs(vy) * 0.5; }
        if (nextPz > BOUND_LIMIT) { nextPz = BOUND_LIMIT; vz = -Math.abs(vz) * 0.5; }
        if (nextPz < -BOUND_LIMIT) { nextPz = -BOUND_LIMIT; vz = Math.abs(vz) * 0.5; }

        // Projekcja barierowa poza strefę No-Fly Zone
        if (distToNoFly < NO_FLY_ZONE_RADIUS && distToNoFly > 1e-4) {
          const pushScale = NO_FLY_ZONE_RADIUS / distToNoFly;
          nextPx = NO_FLY_ZONE_CENTER.x + dx * pushScale;
          nextPy = NO_FLY_ZONE_CENTER.y + dy * pushScale;
          nextPz = NO_FLY_ZONE_CENTER.z + dz * pushScale;
          
          // Odbicie wektora prędkości
          const nx = dx / distToNoFly;
          const ny = dy / distToNoFly;
          const nz = dz / distToNoFly;
          const vDotN = vx * nx + vy * ny + vz * nz;
          vx = (vx - 1.8 * vDotN * nx) * 0.6;
          vy = (vy - 1.8 * vDotN * ny) * 0.6;
          vz = (vz - 1.8 * vDotN * nz) * 0.6;
        }
      } else {
        // Bez filtra: cząstki wylatujące za ekran są resetowane losowo
        if (Math.abs(nextPx) > 1.3 || Math.abs(nextPy) > 1.3 || Math.abs(nextPz) > 1.3) {
          nextPx = (Math.random() - 0.5) * 1.5;
          nextPy = (Math.random() - 0.5) * 1.5;
          nextPz = (Math.random() - 0.5) * 1.5;
        }
      }

      let speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

      // Płynny respawn cząstek o zanikającej prędkości dla ciągłej dynamiki roju
      if (speed < 0.002 || Math.random() < 0.003) {
        nextPx = (Math.random() - 0.5) * 1.6;
        nextPy = (Math.random() - 0.5) * 1.6;
        nextPz = (Math.random() - 0.5) * 1.6;
        vx = (Math.random() - 0.5) * 0.04;
        vy = (Math.random() - 0.5) * 0.04;
        vz = (Math.random() - 0.5) * 0.04;
        speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      }

      agentStates[idx + 0] = nextPx;
      agentStates[idx + 1] = nextPy;
      agentStates[idx + 2] = nextPz;
      agentStates[idx + 3] = vx;
      agentStates[idx + 4] = vy;
      agentStates[idx + 5] = vz;

      // Aktualizacja transformacji instancji
      dummy.position.set(nextPx, nextPy, nextPz);
      const scale = THREE.MathUtils.clamp(0.012 + speed * 0.05, 0.008, 0.024);
      dummy.scale.set(scale, scale, scale * 1.8);
      
      // Orientacja wzdłuż wektora prędkości
      if (speed > 1e-4) {
        dummy.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          new THREE.Vector3(vx / speed, vy / speed, vz / speed)
        );
      }
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);

      // Kolorowanie cząstek na podstawie prędkości i bezpieczeństwa
      if (isNoFlyViolated && !safetyGuardActive) {
        tempColor.setRGB(1.0, 0.1, 0.2); // Czerwony - naruszenie strefy
      } else {
        const t = THREE.MathUtils.clamp(speed * 12.0, 0.0, 1.0);
        // Przejście z chłodnego błękitu do jaskrawego turkusu
        tempColor.setRGB(0.1 + 0.3 * t, 0.6 + 0.4 * t, 0.9 + 0.1 * t);
      }
      meshRef.current.setColorAt(i, tempColor);
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }

    if (onViolationCount) {
      onViolationCount(safetyGuardActive ? 0 : violations);
    }
  });

  return (
    <group>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, numAgents]}
        frustumCulled={false}
      >
        <coneGeometry args={[0.5, 1.5, 5]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
    </group>
  );
};
