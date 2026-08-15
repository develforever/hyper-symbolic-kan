import React, { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  RoboticsCBFEngine,
  type Obstacle3D,
  type CBFStepResult,
} from "../../engine/roboticsCbfEngine";

export interface RoboticsCBFScenarioProps {
  cbfEngine: RoboticsCBFEngine;
  useHocbf: boolean;
  safetyEnabled: boolean;
  alpha: number;
  alpha1: number;
  alpha2: number;
  vMax: number;
  aMax: number;
  tangentialGain: number;
  patrolMode: boolean;
  onTelemetryUpdate: (data: {
    qpLatencyUs: number;
    minH: number;
    speed: number;
    accel: number;
    collision: boolean;
    reachedGoal: boolean;
    useHocbf: boolean;
    safetyEnabled: boolean;
  }) => void;
}

// 3D Model Drona z animowanymi wirnikami i pochyleniem (pitch/roll/yaw)
const Drone3D: React.FC<{
  position: [number, number, number];
  velocity: [number, number, number];
  acceleration: [number, number, number];
  safetyEnabled: boolean;
}> = ({ position, velocity, acceleration, safetyEnabled }) => {
  const groupRef = useRef<THREE.Group>(null);
  const rotor1Ref = useRef<THREE.Mesh>(null);
  const rotor2Ref = useRef<THREE.Mesh>(null);
  const rotor3Ref = useRef<THREE.Mesh>(null);
  const rotor4Ref = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (!groupRef.current) return;

    // Pozycja drona
    groupRef.current.position.set(position[0], position[1], position[2]);

    // Dynamiczny kąt pochylenia (Pitch & Roll) bazujący na prędkości i przyspieszeniu
    const speed = Math.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
    const targetPitch = Math.max(-0.45, Math.min(0.45, -velocity[1] * 0.25 - acceleration[1] * 0.05));
    const targetRoll = Math.max(-0.45, Math.min(0.45, velocity[0] * 0.25 + acceleration[0] * 0.05));
    let targetYaw = 0;
    if (speed > 0.05) {
      targetYaw = Math.atan2(-velocity[0], velocity[1]);
    }

    // Płynna interpolacja obrotu
    groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, targetPitch, 0.15);
    groupRef.current.rotation.z = THREE.MathUtils.lerp(groupRef.current.rotation.z, targetRoll, 0.15);
    groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, targetYaw, 0.1);

    // Animacja obrotu wirników (propellers)
    const spinRate = 35 + speed * 20;
    if (rotor1Ref.current) rotor1Ref.current.rotation.y += spinRate * delta;
    if (rotor2Ref.current) rotor2Ref.current.rotation.y -= spinRate * delta;
    if (rotor3Ref.current) rotor3Ref.current.rotation.y += spinRate * delta;
    if (rotor4Ref.current) rotor4Ref.current.rotation.y -= spinRate * delta;
  });

  const armDist = 0.075;
  const rotorRadius = 0.045;

  return (
    <group ref={groupRef}>
      {/* Kadłub centralny drona */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.065, 0.02, 0.065]} />
        <meshStandardMaterial
          color="#0f172a"
          roughness={0.2}
          metalness={0.9}
        />
      </mesh>

      {/* Górna kopułka komputera pokładowego KAN CBF */}
      <mesh position={[0, 0.015, 0]}>
        <cylinderGeometry args={[0.025, 0.03, 0.015, 16]} />
        <meshStandardMaterial
          color={safetyEnabled ? "#06b6d4" : "#ef4444"}
          emissive={safetyEnabled ? "#0891b2" : "#b91c1c"}
          emissiveIntensity={0.6}
          roughness={0.1}
          metalness={0.8}
        />
      </mesh>

      {/* Przedni reflektor LED */}
      <mesh position={[0, 0, 0.035]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>

      {/* 4 ramiona z włókna węglowego (X-frame) */}
      <mesh position={[armDist / 2, 0, armDist / 2]} rotation={[0, -Math.PI / 4, 0]}>
        <boxGeometry args={[0.01, 0.006, armDist * 1.4]} />
        <meshStandardMaterial color="#334155" metalness={0.7} roughness={0.3} />
      </mesh>
      <mesh position={[-armDist / 2, 0, armDist / 2]} rotation={[0, Math.PI / 4, 0]}>
        <boxGeometry args={[0.01, 0.006, armDist * 1.4]} />
        <meshStandardMaterial color="#334155" metalness={0.7} roughness={0.3} />
      </mesh>

      {/* 4 silniki bezszczotkowe */}
      {[
        [armDist, 0.008, armDist],
        [-armDist, 0.008, armDist],
        [armDist, 0.008, -armDist],
        [-armDist, 0.008, -armDist],
      ].map((pos, idx) => (
        <mesh key={idx} position={pos as [number, number, number]}>
          <cylinderGeometry args={[0.012, 0.012, 0.016, 12]} />
          <meshStandardMaterial color="#1e293b" metalness={0.9} roughness={0.2} />
        </mesh>
      ))}

      {/* 4 wirniki ze śmigłami */}
      <mesh ref={rotor1Ref} position={[armDist, 0.018, armDist]}>
        <boxGeometry args={[rotorRadius * 2, 0.002, 0.008]} />
        <meshStandardMaterial color="#38bdf8" transparent opacity={0.7} />
      </mesh>
      <mesh ref={rotor2Ref} position={[-armDist, 0.018, armDist]}>
        <boxGeometry args={[rotorRadius * 2, 0.002, 0.008]} />
        <meshStandardMaterial color="#38bdf8" transparent opacity={0.7} />
      </mesh>
      <mesh ref={rotor3Ref} position={[armDist, 0.018, -armDist]}>
        <boxGeometry args={[rotorRadius * 2, 0.002, 0.008]} />
        <meshStandardMaterial color="#38bdf8" transparent opacity={0.7} />
      </mesh>
      <mesh ref={rotor4Ref} position={[-armDist, 0.018, -armDist]}>
        <boxGeometry args={[rotorRadius * 2, 0.002, 0.008]} />
        <meshStandardMaterial color="#38bdf8" transparent opacity={0.7} />
      </mesh>

      {/* Status Aura / Bariera drona */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshBasicMaterial
          color={safetyEnabled ? "#10b981" : "#ef4444"}
          wireframe
          transparent
          opacity={0.18}
        />
      </mesh>
    </group>
  );
};

// Wizualizacja Wektorów Sterowania i Gradientu Bariery
const VectorArrow: React.FC<{
  origin: [number, number, number];
  dir: [number, number, number];
  color: string;
  lengthScale?: number;
  label?: string;
}> = ({ origin, dir, color, lengthScale = 0.25, label }) => {
  const arrowRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!arrowRef.current) return;
    const len = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (len < 1e-4) {
      arrowRef.current.visible = false;
      return;
    }
    arrowRef.current.visible = true;
    arrowRef.current.position.set(origin[0], origin[1], origin[2]);

    const target = new THREE.Vector3(origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]);
    arrowRef.current.lookAt(target);
    const scaledLen = Math.min(0.6, len * lengthScale);
    arrowRef.current.scale.set(1, 1, Math.max(0.05, scaledLen));
  });

  return (
    <group ref={arrowRef}>
      {/* Trzon strzałki */}
      <mesh position={[0, 0, 0.5]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.006, 0.006, 1.0, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {/* Grot strzałki */}
      <mesh position={[0, 0, 1.0]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.02, 0.08, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {label && (
        <Text
          position={[0, 0.04, 0.5]}
          fontSize={0.04}
          color={color}
          anchorX="center"
          anchorY="middle"
        >
          {label}
        </Text>
      )}
    </group>
  );
};

// Interaktywna Przeszkoda z możliwością przesuwania kursorem myszy (Raycaster Drag)
const DraggableObstacle: React.FC<{
  obstacle: Obstacle3D;
  dSafe: number;
  onDrag: (id: string, newPos: [number, number, number]) => void;
  setOrbitDisabled: (disabled: boolean) => void;
}> = ({ obstacle, dSafe, onDrag, setOrbitDisabled }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera, raycaster, gl } = useThree();

  const dragPlane = useMemo(() => new THREE.Plane(), []);
  const planeIntersect = useMemo(() => new THREE.Vector3(), []);

  const handlePointerDown = (e: any) => {
    e.stopPropagation();
    setIsDragging(true);
    setOrbitDisabled(true);

    // Utworzenie płaszczyzny przecinającej prostopadłej do kamery na głębokości przeszkody
    const camDir = new THREE.Vector3();
    camera.getWorldDirection(camDir);
    dragPlane.setFromNormalAndCoplanarPoint(camDir.negate(), new THREE.Vector3(...obstacle.position));
  };

  useEffect(() => {
    const handleWindowPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;

      const rect = gl.domElement.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
      if (raycaster.ray.intersectPlane(dragPlane, planeIntersect)) {
        onDrag(obstacle.id, [planeIntersect.x, planeIntersect.y, planeIntersect.z]);
      }
    };

    const handleWindowPointerUp = () => {
      if (isDragging) {
        setIsDragging(false);
        setOrbitDisabled(false);
      }
    };

    window.addEventListener("pointermove", handleWindowPointerMove);
    window.addEventListener("pointerup", handleWindowPointerUp);

    return () => {
      window.removeEventListener("pointermove", handleWindowPointerMove);
      window.removeEventListener("pointerup", handleWindowPointerUp);
    };
  }, [isDragging, camera, dragPlane, gl, onDrag, obstacle.id, planeIntersect, raycaster, setOrbitDisabled]);

  return (
    <group position={obstacle.position}>
      {/* Główna bryła przeszkody */}
      <mesh
        ref={meshRef}
        onPointerDown={handlePointerDown}
        onPointerOver={() => setIsHovered(true)}
        onPointerOut={() => setIsHovered(false)}
      >
        <sphereGeometry args={[obstacle.radius, 28, 28]} />
        <meshStandardMaterial
          color={obstacle.color || "#f59e0b"}
          emissive={obstacle.color || "#f59e0b"}
          emissiveIntensity={isHovered || isDragging ? 0.8 : 0.4}
          roughness={0.25}
          metalness={0.7}
        />
      </mesh>

      {/* Sfera bufora bezpieczeństwa CBF h(x) = 0 */}
      <mesh>
        <sphereGeometry args={[obstacle.radius + dSafe, 20, 20]} />
        <meshBasicMaterial
          color={isDragging ? "#ffffff" : obstacle.color || "#f59e0b"}
          wireframe
          transparent
          opacity={isDragging ? 0.4 : 0.18}
        />
      </mesh>

      {/* Etykieta przeszkody */}
      <Text
        position={[0, obstacle.radius + 0.08, 0]}
        fontSize={0.06}
        color="#f1f5f9"
        anchorX="center"
        anchorY="bottom"
      >
        {obstacle.name}
      </Text>
    </group>
  );
};

// Ślad trajektorii drona
const TrajectoryTrail: React.FC<{ points: [number, number, number][] }> = ({ points }) => {
  const lineObj = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(200 * 3);
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: "#06b6d4", transparent: true, opacity: 0.75 });
    return new THREE.Line(geom, mat);
  }, []);

  useFrame(() => {
    const posAttr = lineObj.geometry.attributes.position as THREE.BufferAttribute;
    const count = points.length;

    for (let i = 0; i < count; i++) {
      posAttr.setXYZ(i, points[i][0], points[i][1], points[i][2]);
    }
    posAttr.needsUpdate = true;
    lineObj.geometry.setDrawRange(0, count);
  });

  return <primitive object={lineObj} />;
};

// Holograficzny Punkt Docelowy (Goal Beacon)
const GoalBeacon: React.FC<{ position: [number, number, number]; isReached: boolean }> = ({
  position,
  isReached,
}) => {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z += 0.03;
      ringRef.current.rotation.x = Math.PI / 2 + Math.sin(state.clock.elapsedTime * 3) * 0.15;
    }
  });

  return (
    <group position={position}>
      {/* Pierścień docelowy */}
      <mesh ref={ringRef}>
        <torusGeometry args={[0.12, 0.015, 16, 32]} />
        <meshStandardMaterial
          color={isReached ? "#10b981" : "#38bdf8"}
          emissive={isReached ? "#10b981" : "#0284c7"}
          emissiveIntensity={0.8}
        />
      </mesh>

      {/* Pionowy słup światła */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.008, 0.008, 1.2, 8]} />
        <meshBasicMaterial
          color={isReached ? "#10b981" : "#38bdf8"}
          transparent
          opacity={0.35}
        />
      </mesh>

      {/* Etykieta */}
      <Text position={[0, 0.22, 0]} fontSize={0.065} color="#38bdf8" anchorX="center">
        GOAL
      </Text>
    </group>
  );
};

export const RoboticsCBFScenario: React.FC<RoboticsCBFScenarioProps> = ({
  cbfEngine,
  useHocbf,
  safetyEnabled,
  alpha,
  alpha1,
  alpha2,
  vMax,
  aMax,
  tangentialGain,
  patrolMode,
  onTelemetryUpdate,
}) => {
  const [orbitDisabled, setOrbitDisabled] = useState(false);
  const [stepData, setStepData] = useState<CBFStepResult | null>(null);
  const [waypoints] = useState<[number, number, number][]>([
    [-0.75, -0.6, -0.2],
    [0.7, -0.4, 0.2],
    [0.65, 0.65, -0.15],
    [-0.7, 0.55, 0.25],
  ]);
  const [currentWpIndex, setCurrentWpIndex] = useState(1);

  // Synchronizacja parametrów z silnikiem CBF
  useEffect(() => {
    cbfEngine.config.useHocbf = useHocbf;
    cbfEngine.config.safetyEnabled = safetyEnabled;
    cbfEngine.config.alpha = alpha;
    cbfEngine.config.alpha1 = alpha1;
    cbfEngine.config.alpha2 = alpha2;
    cbfEngine.config.vMax = vMax;
    cbfEngine.config.aMax = aMax;
    cbfEngine.config.tangentialGain = tangentialGain;
  }, [cbfEngine, useHocbf, safetyEnabled, alpha, alpha1, alpha2, vMax, aMax, tangentialGain]);

  const handleObstacleDrag = useCallback(
    (id: string, newPos: [number, number, number]) => {
      cbfEngine.setObstaclePosition(id, newPos);
    },
    [cbfEngine]
  );

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas
        gl={{
          antialias: true,
          powerPreference: "high-performance",
        }}
      >
        <PerspectiveCamera makeDefault position={[2.2, 2.0, 2.8]} fov={45} />
        <OrbitControls
          enabled={!orbitDisabled}
          enableDamping
          dampingFactor={0.05}
          maxDistance={7.0}
          minDistance={1.0}
        />

        {/* Oświetlenie */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[4, 5, 4]} intensity={1.4} />
        <directionalLight position={[-4, -3, -3]} intensity={0.4} color="#38bdf8" />
        <pointLight position={[0, 0, 0]} intensity={0.3} color="#06b6d4" />

        {/* Pętla symulacji R3F (do 120 FPS) */}
        <PhysicsLoop
          cbfEngine={cbfEngine}
          patrolMode={patrolMode}
          waypoints={waypoints}
          currentWpIndex={currentWpIndex}
          setCurrentWpIndex={setCurrentWpIndex}
          onStep={(res) => {
            setStepData(res);
            const speed = Math.sqrt(res.velocity[0] ** 2 + res.velocity[1] ** 2 + res.velocity[2] ** 2);
            const accel = Math.sqrt(res.acceleration[0] ** 2 + res.acceleration[1] ** 2 + res.acceleration[2] ** 2);
            onTelemetryUpdate({
              qpLatencyUs: res.qpLatencyUs,
              minH: res.minH,
              speed,
              accel,
              collision: res.collision,
              reachedGoal: res.reachedGoal,
              useHocbf,
              safetyEnabled,
            });
          }}
        />

        {/* Granica domenowa Chebyshev [-L, L]^3 */}
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(1.9, 1.9, 1.9)]} />
          <lineBasicMaterial color="#334155" transparent opacity={0.4} />
        </lineSegments>

        {/* Płaszczyzna siatki pomocniczej na dnie */}
        <gridHelper args={[1.9, 10, "#06b6d4", "#1e293b"]} position={[0, -0.95, 0]} />

        {/* Interaktywne Przeszkody 3D */}
        {cbfEngine.obstacles.map((obs) => (
          <DraggableObstacle
            key={obs.id}
            obstacle={obs}
            dSafe={cbfEngine.config.dSafe}
            onDrag={handleObstacleDrag}
            setOrbitDisabled={setOrbitDisabled}
          />
        ))}

        {/* Cel podróży */}
        <GoalBeacon
          position={cbfEngine.goal}
          isReached={stepData ? stepData.reachedGoal : false}
        />

        {/* Ślad lotu drona */}
        <TrajectoryTrail points={cbfEngine.trajectoryHistory} />

        {/* Model Drona 3D */}
        {stepData && (
          <>
            <Drone3D
              position={stepData.position}
              velocity={stepData.velocity}
              acceleration={stepData.acceleration}
              safetyEnabled={safetyEnabled}
            />

            {/* Wektory Sterowania */}
            {/* 1. Zielony: Nominalny / Pożądany */}
            <VectorArrow
              origin={stepData.position}
              dir={stepData.nominalControl}
              color="#22c55e"
              lengthScale={useHocbf ? 0.08 : 0.25}
              label="u_des"
            />

            {/* 2. Czerwony: Gradient Bariery \nabla h(x) */}
            <VectorArrow
              origin={stepData.position}
              dir={stepData.barrierGradient}
              color="#ef4444"
              lengthScale={0.35}
              label="∇h"
            />

            {/* 3. Niebieski: Bezpieczny Skorygowany CBF */}
            <VectorArrow
              origin={stepData.position}
              dir={stepData.filteredControl}
              color="#06b6d4"
              lengthScale={useHocbf ? 0.08 : 0.25}
              label="u_safe"
            />
          </>
        )}
      </Canvas>

      {/* Instrukcja interakcji 3D na dole viewportu */}
      <div
        style={{
          position: "absolute",
          bottom: "16px",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(10, 15, 26, 0.8)",
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
          <b style={{ color: "var(--amber-primary)" }}>Myszka Lewy Przycisk:</b> Chwyć i przesuń przeszkodę w locie drona
        </span>
        <span>&bull;</span>
        <span>
          <b style={{ color: "var(--cyan-primary)" }}>Prawy Przycisk:</b> Obrót kamery 3D
        </span>
      </div>
    </div>
  );
};

// Komponent wewnętrzny pętli fizyki
const PhysicsLoop: React.FC<{
  cbfEngine: RoboticsCBFEngine;
  patrolMode: boolean;
  waypoints: [number, number, number][];
  currentWpIndex: number;
  setCurrentWpIndex: React.Dispatch<React.SetStateAction<number>>;
  onStep: (res: CBFStepResult) => void;
}> = ({
  cbfEngine,
  patrolMode,
  waypoints,
  currentWpIndex,
  setCurrentWpIndex,
  onStep,
}) => {
  useFrame((_, delta) => {
    // Bezpieczny clamping delta time (np. min 120 FPS lub podział sub-step)
    const dt = Math.min(delta, 0.033);

    const res = cbfEngine.step(dt);
    onStep(res);

    // W trybie patrolowym: przełączanie na kolejny waypoint po dotarciu
    if (patrolMode && res.reachedGoal) {
      const nextIdx = (currentWpIndex + 1) % waypoints.length;
      setCurrentWpIndex(nextIdx);
      cbfEngine.setGoal(waypoints[nextIdx]);
    }
  });

  return null;
};
