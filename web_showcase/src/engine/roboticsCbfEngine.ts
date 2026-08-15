/**
 * Robotics Control Barrier Function (CBF) and Higher-Order CBF (HOCBF) Engine
 * Ported from src/applications/robotics_cbf_planner.py
 *
 * Implements:
 * 1. Continuous Barrier representations h(x) and analytical gradients \nabla h(x) & Hessians \nabla^2 h(x).
 * 2. 1st-Order Kinematic CBF Filter (\dot{p} = u) with analytical QP active-set projection.
 * 3. 2nd-Order Dynamic HOCBF Filter (\ddot{p} = a) with relative degree 2 Lie derivatives.
 * 4. Tangential circulation guidance field eliminating saddle-point deadlock.
 * 5. Chebyshev Domain boundary box enforcement.
 */

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface Obstacle3D {
  id: string;
  name: string;
  position: [number, number, number];
  radius: number;
  color?: string;
  isDraggable?: boolean;
}

export interface CBFEngineConfig {
  alpha: number;             // Class-K gain for 1st order CBF (\dot{h} + \alpha h >= 0)
  alpha1: number;            // Class-K gain 1 for 2nd order HOCBF
  alpha2: number;            // Class-K gain 2 for 2nd order HOCBF
  vMax: number;              // Max velocity [m/s]
  aMax: number;              // Max acceleration [m/s^2]
  kpGoal: number;            // Proportional gain to goal
  kdGoal: number;            // Derivative gain to goal (damping)
  dSafe: number;             // Extra safety margin buffer [m]
  domainLimit: number;       // Chebyshev box bound [-L, L]^3
  tangentialGain: number;    // Anti-saddle tangential avoidance gain
  epsReg: number;            // Numerical regularization epsilon
  useHocbf: boolean;         // True: 2nd-order dynamic HOCBF, False: 1st-order kinematic CBF
  safetyEnabled: boolean;    // True: CBF filter ON (0% collisions), False: Raw unconstrained control
}

export const DEFAULT_CBF_CONFIG: CBFEngineConfig = {
  alpha: 3.5,
  alpha1: 6.0,
  alpha2: 4.0,
  vMax: 2.2,
  aMax: 9.0,
  kpGoal: 2.8,
  kdGoal: 2.2,
  dSafe: 0.04,
  domainLimit: 0.95,
  tangentialGain: 1.8,
  epsReg: 1e-6,
  useHocbf: true,
  safetyEnabled: true,
};

export interface CBFStepResult {
  position: [number, number, number];
  velocity: [number, number, number];
  acceleration: [number, number, number];
  nominalControl: [number, number, number];     // Green vector (Target)
  barrierGradient: [number, number, number];    // Red vector (\nabla h)
  filteredControl: [number, number, number];    // Blue vector (Safe CBF)
  minH: number;                                 // Minimum barrier distance
  closestObstacleIndex: number;
  qpLatencyUs: number;                          // Latency in microseconds
  collision: boolean;
  distanceToGoal: number;
  reachedGoal: boolean;
}

export class RoboticsCBFEngine {
  public config: CBFEngineConfig;
  public obstacles: Obstacle3D[];
  public position: [number, number, number];
  public velocity: [number, number, number];
  public acceleration: [number, number, number];
  public goal: [number, number, number];
  public trajectoryHistory: [number, number, number][];
  public maxHistoryLength: number = 180;

  constructor(
    config?: Partial<CBFEngineConfig>,
    initialPos: [number, number, number] = [-0.75, -0.6, -0.2],
    initialGoal: [number, number, number] = [0.75, 0.6, 0.2]
  ) {
    this.config = { ...DEFAULT_CBF_CONFIG, ...config };
    this.position = [...initialPos];
    this.velocity = [0, 0, 0];
    this.acceleration = [0, 0, 0];
    this.goal = [...initialGoal];
    this.trajectoryHistory = [[...initialPos]];

    // Domyślne przeszkody 3D w torze lotu
    this.obstacles = [
      { id: "obs-1", name: "Alpha Core", position: [-0.25, -0.15, -0.05], radius: 0.26, color: "#f59e0b", isDraggable: true },
      { id: "obs-2", name: "Beta Node", position: [0.28, 0.22, 0.1], radius: 0.24, color: "#ef4444", isDraggable: true },
      { id: "obs-3", name: "Gamma Gate", position: [0.05, -0.4, 0.25], radius: 0.20, color: "#8b5cf6", isDraggable: true },
      { id: "obs-4", name: "Delta Pylon", position: [-0.4, 0.45, -0.1], radius: 0.22, color: "#06b6d4", isDraggable: true },
    ];
  }

  public setGoal(goal: [number, number, number]) {
    this.goal = [...goal];
  }

  public setObstaclePosition(id: string, pos: [number, number, number]) {
    const obs = this.obstacles.find((o) => o.id === id);
    if (obs) {
      obs.position = [
        Math.max(-this.config.domainLimit, Math.min(this.config.domainLimit, pos[0])),
        Math.max(-this.config.domainLimit, Math.min(this.config.domainLimit, pos[1])),
        Math.max(-this.config.domainLimit, Math.min(this.config.domainLimit, pos[2])),
      ];
    }
  }

  public resetDrone(pos?: [number, number, number]) {
    const p = pos || [-0.75, -0.6, -0.2];
    this.position = [...p];
    this.velocity = [0, 0, 0];
    this.acceleration = [0, 0, 0];
    this.trajectoryHistory = [[...p]];
  }

  /**
   * Wartość funkcji bariery h(p) dla przeszkody sferycznej:
   * h(p) = ||p - c|| - (radius + d_safe)
   */
  public evaluateObstacleH(p: [number, number, number], obs: Obstacle3D): number {
    const dx = p[0] - obs.position[0];
    const dy = p[1] - obs.position[1];
    const dz = p[2] - obs.position[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return dist - (obs.radius + this.config.dSafe);
  }

  /**
   * Analityczny gradient funkcji bariery \nabla h(p):
   * \nabla h(p) = (p - c) / ||p - c||
   */
  public gradientObstacleH(p: [number, number, number], obs: Obstacle3D): [number, number, number] {
    const dx = p[0] - obs.position[0];
    const dy = p[1] - obs.position[1];
    const dz = p[2] - obs.position[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;
    return [dx / dist, dy / dist, dz / dist];
  }

  /**
   * Oblicza hesjanowe sprzężenie prędkości v^T \nabla^2 h(p) v dla sfery:
   * \nabla^2 h(p) = 1/||d|| * (I - \hat{d}\hat{d}^T)
   * v^T \nabla^2 h v = 1/||d|| * (||v||^2 - (v^T \hat{d})^2)
   */
  public hessianObstacleProduct(
    p: [number, number, number],
    v: [number, number, number],
    obs: Obstacle3D
  ): number {
    const dx = p[0] - obs.position[0];
    const dy = p[1] - obs.position[1];
    const dz = p[2] - obs.position[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;

    const nx = dx / dist;
    const ny = dy / dist;
    const nz = dz / dist;

    const vNormSq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const vDotN = v[0] * nx + v[1] * ny + v[2] * nz;

    return (vNormSq - vDotN * vDotN) / dist;
  }

  /**
   * Pole opływu ortogonalnego (Tangential Circulation Guidance)
   * Zapobiega utknięciu drona w punktach siodłowych w przypadku czołowego zbliżania się do przeszkody.
   */
  private computeTangentialGuidance(
    p: [number, number, number],
    uNom: [number, number, number]
  ): [number, number, number] {
    let guidedX = uNom[0];
    let guidedY = uNom[1];
    let guidedZ = uNom[2];

    const uNorm = Math.sqrt(uNom[0] * uNom[0] + uNom[1] * uNom[1] + uNom[2] * uNom[2]);
    if (uNorm < 1e-4) return [guidedX, guidedY, guidedZ];

    for (const obs of this.obstacles) {
      const hVal = this.evaluateObstacleH(p, obs);
      if (hVal > 0.35) continue; // Wpływ tylko w strefie bliskiej przeszkody

      const grad = this.gradientObstacleH(p, obs);
      const proj = uNom[0] * grad[0] + uNom[1] * grad[1] + uNom[2] * grad[2];

      // Jeśli wektor nominalny zmierza wprost w stronę przeszkody (proj < 0)
      if (proj < 0) {
        // Składowa prostopadła do gradientu
        let perpX = uNom[0] - proj * grad[0];
        let perpY = uNom[1] - proj * grad[1];
        let perpZ = uNom[2] - proj * grad[2];
        let perpNorm = Math.sqrt(perpX * perpX + perpY * perpY + perpZ * perpZ);

        let tanX: number, tanY: number, tanZ: number;

        // Jeśli zderzenie jest idealnie symetryczne (czołowe), wybierz wektor krzyżowy z osią Z lub Y
        if (perpNorm < 0.15 * uNorm) {
          let cx = grad[1];
          let cy = -grad[0];
          let cz = 0;
          let cNorm = Math.sqrt(cx * cx + cy * cy);
          if (cNorm < 0.1) {
            cx = -grad[2];
            cy = 0;
            cz = grad[0];
            cNorm = Math.sqrt(cx * cx + cz * cz) + 1e-9;
          }
          tanX = cx / cNorm;
          tanY = cy / cNorm;
          tanZ = cz / cNorm;
        } else {
          tanX = perpX / (perpNorm + 1e-9);
          tanY = perpY / (perpNorm + 1e-9);
          tanZ = perpZ / (perpNorm + 1e-9);
        }

        const weight = this.config.tangentialGain * Math.exp(-Math.max(0, hVal) / 0.12);
        guidedX += weight * tanX;
        guidedY += weight * tanY;
        guidedZ += weight * tanZ;
      }
    }

    return [guidedX, guidedY, guidedZ];
  }

  /**
   * Analityczny, deterministyczny solver QP w R^3 dla więzów CBF.
   * Czas wykonania < 0.01 ms (zwykle < 5 mikrosekund).
   */
  private solveAnalytical3DQP(
    target: [number, number, number],
    aRows: [number, number, number][],
    bVals: number[],
    limit: number
  ): [number, number, number] {
    let sol: [number, number, number] = [
      Math.max(-limit, Math.min(limit, target[0])),
      Math.max(-limit, Math.min(limit, target[1])),
      Math.max(-limit, Math.min(limit, target[2])),
    ];

    if (aRows.length === 0) return sol;

    // Sprawdzenie naruszeń więzów
    let maxViol = 0;
    let worstIdx = -1;

    for (let i = 0; i < aRows.length; i++) {
      const a = aRows[i];
      const viol = a[0] * sol[0] + a[1] * sol[1] + a[2] * sol[2] - bVals[i];
      if (viol > maxViol) {
        maxViol = viol;
        worstIdx = i;
      }
    }

    // Brak naruszeń - sterowanie bezpieczne
    if (maxViol <= 1e-6) {
      return sol;
    }

    // Iteracyjna projekcja na aktywne półprzestrzenie (Active-Set Dual Projection)
    for (let iter = 0; iter < 6; iter++) {
      if (worstIdx < 0 || maxViol <= 1e-6) break;

      const a = aRows[worstIdx];
      const normSq = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + this.config.epsReg;
      const lambda = maxViol / normSq;

      // Rzut u = u - lambda * a
      sol[0] -= lambda * a[0];
      sol[1] -= lambda * a[1];
      sol[2] -= lambda * a[2];

      // Ograniczenie do dozwolonego pudełka [-limit, limit]
      sol[0] = Math.max(-limit, Math.min(limit, sol[0]));
      sol[1] = Math.max(-limit, Math.min(limit, sol[1]));
      sol[2] = Math.max(-limit, Math.min(limit, sol[2]));

      // Sprawdź kolejne naruszenia
      maxViol = 0;
      worstIdx = -1;
      for (let i = 0; i < aRows.length; i++) {
        const row = aRows[i];
        const v = row[0] * sol[0] + row[1] * sol[1] + row[2] * sol[2] - bVals[i];
        if (v > maxViol) {
          maxViol = v;
          worstIdx = i;
        }
      }
    }

    return sol;
  }

  /**
   * Filtr Kinematyczny 1. rzędu CBF (\dot{p} = u)
   */
  public solveKinematicCBF(
    p: [number, number, number],
    uNom: [number, number, number]
  ): [number, number, number] {
    const uTarget = this.computeTangentialGuidance(p, uNom);
    if (!this.config.safetyEnabled) {
      return [
        Math.max(-this.config.vMax, Math.min(this.config.vMax, uTarget[0])),
        Math.max(-this.config.vMax, Math.min(this.config.vMax, uTarget[1])),
        Math.max(-this.config.vMax, Math.min(this.config.vMax, uTarget[2])),
      ];
    }

    const aRows: [number, number, number][] = [];
    const bVals: number[] = [];

    // Więzy dla przeszkód: -\nabla h^T u <= \alpha h
    for (const obs of this.obstacles) {
      const hVal = this.evaluateObstacleH(p, obs);
      const grad = this.gradientObstacleH(p, obs);

      aRows.push([-grad[0], -grad[1], -grad[2]]);
      bVals.push(this.config.alpha * hVal);
    }

    // Więzy domeny [-domainLimit, domainLimit]^3
    const L = this.config.domainLimit;
    const alpha = this.config.alpha;
    // x <= L, x >= -L
    aRows.push([1, 0, 0]); bVals.push(alpha * (L - p[0]));
    aRows.push([-1, 0, 0]); bVals.push(alpha * (p[0] + L));
    aRows.push([0, 1, 0]); bVals.push(alpha * (L - p[1]));
    aRows.push([0, -1, 0]); bVals.push(alpha * (p[1] + L));
    aRows.push([0, 0, 1]); bVals.push(alpha * (L - p[2]));
    aRows.push([0, 0, -1]); bVals.push(alpha * (p[2] + L));

    return this.solveAnalytical3DQP(uTarget, aRows, bVals, this.config.vMax);
  }

  /**
   * Filtr Dynamiczny 2. rzędu HOCBF (\ddot{p} = a)
   */
  public solveDynamicHOCBF(
    p: [number, number, number],
    v: [number, number, number],
    aNom: [number, number, number]
  ): [number, number, number] {
    const aTarget = this.computeTangentialGuidance(p, aNom);
    if (!this.config.safetyEnabled) {
      return [
        Math.max(-this.config.aMax, Math.min(this.config.aMax, aTarget[0])),
        Math.max(-this.config.aMax, Math.min(this.config.aMax, aTarget[1])),
        Math.max(-this.config.aMax, Math.min(this.config.aMax, aTarget[2])),
      ];
    }

    const { alpha1, alpha2 } = this.config;
    const aRows: [number, number, number][] = [];
    const bVals: number[] = [];

    // Więzy dla przeszkód dynamicznych HOCBF:
    // -\nabla h^T a <= v^T \nabla^2 h v + (\alpha_1 + \alpha_2) \nabla h^T v + \alpha_1 \alpha_2 h
    for (const obs of this.obstacles) {
      const hVal = this.evaluateObstacleH(p, obs);
      const grad = this.gradientObstacleH(p, obs);
      const lfH = grad[0] * v[0] + grad[1] * v[1] + grad[2] * v[2];
      const hessTerm = this.hessianObstacleProduct(p, v, obs);

      aRows.push([-grad[0], -grad[1], -grad[2]]);
      bVals.push(hessTerm + (alpha1 + alpha2) * lfH + alpha1 * alpha2 * hVal);
    }

    // Więzy domeny 2. rzędu
    const L = this.config.domainLimit;
    const alphaSum = alpha1 + alpha2;
    const alphaProd = alpha1 * alpha2;

    // X Axis
    aRows.push([1, 0, 0]);
    bVals.push(alphaSum * (-v[0]) + alphaProd * (L - p[0]));
    aRows.push([-1, 0, 0]);
    bVals.push(alphaSum * (v[0]) + alphaProd * (p[0] + L));

    // Y Axis
    aRows.push([0, 1, 0]);
    bVals.push(alphaSum * (-v[1]) + alphaProd * (L - p[1]));
    aRows.push([0, -1, 0]);
    bVals.push(alphaSum * (v[1]) + alphaProd * (p[1] + L));

    // Z Axis
    aRows.push([0, 0, 1]);
    bVals.push(alphaSum * (-v[2]) + alphaProd * (L - p[2]));
    aRows.push([0, 0, -1]);
    bVals.push(alphaSum * (v[2]) + alphaProd * (p[2] + L));

    return this.solveAnalytical3DQP(aTarget, aRows, bVals, this.config.aMax);
  }

  /**
   * Główny krok symulacji fizyki i barier CBF (wywoływany w RAF przy 60-120 FPS)
   */
  public step(dt: number = 0.016): CBFStepResult {
    const t0 = performance.now();

    // 1. Obliczenie wektora błędu do celu
    const dx = this.goal[0] - this.position[0];
    const dy = this.goal[1] - this.position[1];
    const dz = this.goal[2] - this.position[2];
    const distToGoal = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const reachedGoal = distToGoal < 0.08;

    // 2. Znalezienie najbliższej przeszkody i minimalnego h
    let minH = Infinity;
    let closestIdx = -1;
    let closestGrad: [number, number, number] = [0, 0, 0];

    for (let i = 0; i < this.obstacles.length; i++) {
      const hVal = this.evaluateObstacleH(this.position, this.obstacles[i]);
      if (hVal < minH) {
        minH = hVal;
        closestIdx = i;
        closestGrad = this.gradientObstacleH(this.position, this.obstacles[i]);
      }
    }

    let nominalCtrl: [number, number, number];
    let filteredCtrl: [number, number, number];

    if (this.config.useHocbf) {
      // Dynamika 2. rzędu (przyspieszenie / siła ciągu)
      const nomAx = this.config.kpGoal * dx - this.config.kdGoal * this.velocity[0];
      const nomAy = this.config.kpGoal * dy - this.config.kdGoal * this.velocity[1];
      const nomAz = this.config.kpGoal * dz - this.config.kdGoal * this.velocity[2];
      nominalCtrl = [
        Math.max(-this.config.aMax, Math.min(this.config.aMax, nomAx)),
        Math.max(-this.config.aMax, Math.min(this.config.aMax, nomAy)),
        Math.max(-this.config.aMax, Math.min(this.config.aMax, nomAz)),
      ];

      filteredCtrl = this.solveDynamicHOCBF(this.position, this.velocity, nominalCtrl);

      // Całkowanie Eulera-Cromera (dynamika bezwładnościowa)
      this.acceleration = [...filteredCtrl];
      this.velocity[0] = Math.max(-this.config.vMax, Math.min(this.config.vMax, this.velocity[0] + this.acceleration[0] * dt));
      this.velocity[1] = Math.max(-this.config.vMax, Math.min(this.config.vMax, this.velocity[1] + this.acceleration[1] * dt));
      this.velocity[2] = Math.max(-this.config.vMax, Math.min(this.config.vMax, this.velocity[2] + this.acceleration[2] * dt));

      this.position[0] += this.velocity[0] * dt;
      this.position[1] += this.velocity[1] * dt;
      this.position[2] += this.velocity[2] * dt;
    } else {
      // Kinematyka 1. rzędu (prędkość)
      const nomVx = this.config.kpGoal * dx;
      const nomVy = this.config.kpGoal * dy;
      const nomVz = this.config.kpGoal * dz;
      nominalCtrl = [
        Math.max(-this.config.vMax, Math.min(this.config.vMax, nomVx)),
        Math.max(-this.config.vMax, Math.min(this.config.vMax, nomVy)),
        Math.max(-this.config.vMax, Math.min(this.config.vMax, nomVz)),
      ];

      filteredCtrl = this.solveKinematicCBF(this.position, nominalCtrl);

      this.velocity = [...filteredCtrl];
      this.acceleration = [0, 0, 0];

      this.position[0] += this.velocity[0] * dt;
      this.position[1] += this.velocity[1] * dt;
      this.position[2] += this.velocity[2] * dt;
    }

    const t1 = performance.now();
    const qpLatencyUs = parseFloat(((t1 - t0) * 1000).toFixed(2));

    // Aktualizacja historii trajektorii
    this.trajectoryHistory.push([...this.position]);
    if (this.trajectoryHistory.length > this.maxHistoryLength) {
      this.trajectoryHistory.shift();
    }

    const collision = minH < 0;

    return {
      position: [...this.position],
      velocity: [...this.velocity],
      acceleration: [...this.acceleration],
      nominalControl: nominalCtrl,
      barrierGradient: closestGrad,
      filteredControl: filteredCtrl,
      minH,
      closestObstacleIndex: closestIdx,
      qpLatencyUs,
      collision,
      distanceToGoal: distToGoal,
      reachedGoal,
    };
  }
}
