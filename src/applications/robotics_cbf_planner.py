r"""
Robotics Control Barrier Function (CBF) Planner with Continuous KAN Fields.

Implements:
1. Continuous KAN-based barrier representations h(x) and analytical gradients \nabla h(x).
2. Kinematic CBF Safety Filter (Relative Degree = 1, velocity control u).
3. Dynamic Higher-Order CBF (HOCBF / Exponential CBF, Relative Degree = 2, acceleration control a).
4. Multi-agent / Drone Swarm Collision Avoidance with zero inter-agent and obstacle collisions.
5. Tangential circulation field preventing saddle-point deadlocks on symmetric head-on approaches.
6. Domain Boundary Box CBF for keeping agents inside [-1, 1]^D Chebyshev orthogonal bounds.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.optimize import minimize


@dataclass
class CBFConfig:
    """Konfiguracja parametrów numerycznych i fizycznych dla planera CBF."""
    alpha: float = 3.0                    # Wzmocnienie klasy-K dla CBF 1. rzędu (\dot{h} + \alpha h >= 0)
    alpha_hocbf: Tuple[float, float] = (6.0, 4.0)  # Wzmocnienia (\alpha_1, \alpha_2) dla HOCBF 2. rzędu
    eps_reg: float = 1e-6                 # Regularyzacja gradientu (||grad||^2 + eps)
    v_max: float = 2.0                    # Maksymalna dopuszczalna prędkość agenta [m/s]
    a_max: float = 10.0                   # Maksymalne dopuszczalne przyspieszenie [m/s^2]
    d_safe: float = 0.05                  # Margines bufora bezpieczeństwa [m]
    domain_limit: float = 0.95            # Granica domeny [-domain_limit, domain_limit]^D
    kp_goal: float = 3.0                  # Współczynnik proporcjonalny dążenia do celu
    tangential_gain: float = 1.5          # Wzmocnienie omijania bocznego (eliminacja punktów siodłowych)


class ContinuousKANObstacleField:
    """
    Ciągłe pole przeszkód oparte na modelach KAN (TDFFNet, TensorTrainKAN, CP-KAN).
    
    Definiuje barierę bezpieczeństwa h(x):
      h(x) = f_KAN(x) - threshold >= 0 (strefa bezpieczna)
      h(x) < 0 (wnętrze przeszkody / kolizja)
    """
    def __init__(self, kan_model: Any, threshold: float = 0.0, invert: bool = False, name: str = "kan_obstacle"):
        self.kan_model = kan_model
        self.threshold = threshold
        self.invert = invert
        self.name = name

    def evaluate_h(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Zwraca wartość bariery h(x) dla punktów X o kształcie (N, D) lub (D,)."""
        X_arr = np.atleast_2d(X)
        f_val = self.kan_model.evaluate(X_arr)
        if self.invert:
            h_val = self.threshold - f_val
        else:
            h_val = f_val - self.threshold
        if X.ndim == 1:
            return float(np.asarray(h_val).ravel()[0])
        return np.asarray(h_val).ravel()

    def gradient_h(self, X: np.ndarray) -> np.ndarray:
        """Zwraca analityczny gradient \nabla h(x) dla punktów X."""
        X_arr = np.atleast_2d(X)
        grad_f = self.kan_model.gradient(X_arr)
        if self.invert:
            grad_h = -grad_f
        else:
            grad_h = grad_f
        if X.ndim == 1:
            return np.asarray(grad_h[0], dtype=np.float64)
        return np.asarray(grad_h, dtype=np.float64)


class SyntheticSphereObstacle:
    """Analityczna przeszkoda sferyczna w przestrzeni D-wymiarowej."""
    def __init__(self, center: np.ndarray, radius: float, name: str = "sphere_obstacle"):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.name = name

    def evaluate_h(self, X: np.ndarray) -> Union[float, np.ndarray]:
        X_arr = np.atleast_2d(X)
        dist = np.linalg.norm(X_arr - self.center, axis=1)
        h = dist - self.radius
        if X.ndim == 1:
            return float(h[0])
        return h

    def gradient_h(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.atleast_2d(X)
        diff = X_arr - self.center
        dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9
        grad = diff / dist
        if X.ndim == 1:
            return np.asarray(grad[0], dtype=np.float64)
        return np.asarray(grad, dtype=np.float64)


class DomainBoxCBF:
    """Bariera ograniczająca pozycję agenta wewnątrz hipersześcianu [-L, L]^D."""
    def __init__(self, limit: float = 0.95, dim: int = 3):
        self.limit = limit
        self.dim = dim

    def get_constraints(self, p: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        p_arr = np.asarray(p, dtype=np.float64).ravel()
        D = len(p_arr)
        
        A = np.zeros((2 * D, D))
        b = np.zeros(2 * D)
        
        for d in range(D):
            A[2 * d, d] = 1.0
            b[2 * d] = alpha * (self.limit - p_arr[d])
            
            A[2 * d + 1, d] = -1.0
            b[2 * d + 1] = alpha * (p_arr[d] + self.limit)
            
        return A, b


class InterAgentCBF:
    """Bariera bezpieczeństwa między agentami dla roju / floty dronów."""
    def __init__(self, r_safe: float = 0.06):
        self.r_safe = r_safe

    def get_pairwise_constraints(
        self,
        agent_idx: int,
        positions: np.ndarray,
        alpha: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        N, D = positions.shape
        p_i = positions[agent_idx]
        A_list = []
        b_list = []
        
        for j in range(N):
            if j == agent_idx:
                continue
            p_j = positions[j]
            diff = p_i - p_j
            dist = np.linalg.norm(diff)
            
            h_ij = dist - 2.0 * self.r_safe
            if dist > 1e-7:
                grad_ij = diff / dist
            else:
                grad_ij = np.random.randn(D)
                grad_ij /= (np.linalg.norm(grad_ij) + 1e-9)
                
            A_list.append(-grad_ij)
            b_list.append(alpha * h_ij)
            
        return A_list, b_list


class CBFPlanner:
    """
    Główny silnik planowania trajektorii bezkolizyjnych z Funkcjami Barierowymi CBF.
    """
    def __init__(self, config: Optional[CBFConfig] = None):
        self.config = config or CBFConfig()
        self.domain_box = DomainBoxCBF(limit=self.config.domain_limit)
        self.inter_agent = InterAgentCBF(r_safe=self.config.d_safe)

    def _compute_tangential_guidance(self, p: np.ndarray, u_nom: np.ndarray, obstacles: List[Any]) -> np.ndarray:
        """
        Oblicza ortogonalne pole omijania przeszkód w przypadku zbliżania się czołowego (head-on collision).
        Zapobiega utknięciu w punktach siodłowych CBF.
        """
        D = len(p)
        u_guided = u_nom.copy()
        
        for obs in obstacles:
            h_val = float(np.asarray(obs.evaluate_h(p)).ravel()[0])
            if h_val > 0.25:
                continue
                
            grad_h = np.asarray(obs.gradient_h(p), dtype=np.float64).ravel()
            grad_norm = np.linalg.norm(grad_h)
            if grad_norm < 1e-6:
                continue
            e_grad = grad_h / grad_norm
            
            proj = np.dot(u_nom, e_grad)
            if proj < 0:
                v_perp = u_nom - proj * e_grad
                v_perp_norm = np.linalg.norm(v_perp)
                
                if v_perp_norm < 0.1 * (np.linalg.norm(u_nom) + 1e-6):
                    if D == 3:
                        cand = np.cross(e_grad, np.array([0.0, 0.0, 1.0]))
                        if np.linalg.norm(cand) < 1e-3:
                            cand = np.cross(e_grad, np.array([0.0, 1.0, 0.0]))
                        e_tangent = cand / (np.linalg.norm(cand) + 1e-9)
                    else:
                        e_tangent = np.zeros(D)
                        e_tangent[1 if abs(e_grad[0]) > 0.5 else 0] = 1.0
                        e_tangent -= np.dot(e_tangent, e_grad) * e_grad
                        e_tangent /= (np.linalg.norm(e_tangent) + 1e-9)
                else:
                    e_tangent = v_perp / (v_perp_norm + 1e-9)
                    
                weight = self.config.tangential_gain * np.exp(-max(0.0, h_val) / 0.15)
                u_guided += weight * e_tangent
                
        return u_guided

    def solve_kinematic_qp(
        self,
        p: np.ndarray,
        u_nom: np.ndarray,
        obstacles: List[Any],
        include_domain: bool = True
    ) -> np.ndarray:
        r"""
        Rozwiązuje filtr bezpieczeństwa CBF dla kinematyki 1. rzędu (\dot{p} = u):
        
        \min_u \frac{1}{2} \|u - u_{guided}\|^2
        s.t.  -\nabla h_k(p)^T u \le \alpha h_k(p)
              \|u\|_\infty \le v_{max}
        """
        p = np.asarray(p, dtype=np.float64)
        u_nom = np.asarray(u_nom, dtype=np.float64)
        D = len(p)
        
        u_target = self._compute_tangential_guidance(p, u_nom, obstacles)
        
        A_rows = []
        b_vals = []
        
        for obs in obstacles:
            h_val = float(np.asarray(obs.evaluate_h(p)).ravel()[0])
            grad_val = np.asarray(obs.gradient_h(p), dtype=np.float64).ravel()
            
            grad_norm_sq = np.sum(grad_val ** 2)
            if grad_norm_sq < self.config.eps_reg:
                grad_val = grad_val + np.random.normal(0.0, 1e-4, size=D)
                
            A_rows.append(-grad_val)
            b_vals.append(self.config.alpha * h_val)
            
        if include_domain:
            A_dom, b_dom = self.domain_box.get_constraints(p, self.config.alpha)
            A_rows.append(A_dom)
            b_vals.append(b_dom)
            
        if not A_rows:
            return np.clip(u_target, -self.config.v_max, self.config.v_max)
            
        A = np.vstack(A_rows)
        b = np.concatenate([np.atleast_1d(x) for x in b_vals])
        
        violations = A @ u_target - b
        if np.all(violations <= 1e-7):
            return np.clip(u_target, -self.config.v_max, self.config.v_max)
            
        def objective(u):
            diff = u - u_target
            return 0.5 * np.dot(diff, diff)
            
        def objective_grad(u):
            return u - u_target
            
        bounds = [(-self.config.v_max, self.config.v_max)] * D
        constraints = [{'type': 'ineq', 'fun': lambda u, A_mat=A, b_vec=b: b_vec - A_mat @ u,
                        'jac': lambda u, A_mat=A: -A_mat}]
                        
        res = minimize(
            objective,
            x0=np.clip(u_target, -self.config.v_max, self.config.v_max),
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-8, 'maxiter': 50, 'disp': False}
        )
        
        if res.success:
            return res.x
            
        worst_idx = int(np.argmax(violations))
        a_k = A[worst_idx]
        norm_sq = np.sum(a_k ** 2) + self.config.eps_reg
        u_fallback = u_target - (violations[worst_idx] / norm_sq) * a_k
        return np.clip(u_fallback, -self.config.v_max, self.config.v_max)

    def solve_dynamic_hocbf_qp(
        self,
        p: np.ndarray,
        v: np.ndarray,
        a_nom: np.ndarray,
        obstacles: List[Any],
        include_domain: bool = True
    ) -> np.ndarray:
        r"""
        Rozwiązuje filtr bezpieczeństwa HOCBF dla dynamiki 2. rzędu (\ddot{p} = a):
        
        h_e(p, v) = \nabla h(p)^T v + \alpha_1 h(p)
        \dot{h}_e + \alpha_2 h_e \ge 0
        """
        p = np.asarray(p, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        a_nom = np.asarray(a_nom, dtype=np.float64)
        D = len(p)
        
        a_target = self._compute_tangential_guidance(p, a_nom, obstacles)
        
        alpha1, alpha2 = self.config.alpha_hocbf
        A_rows = []
        b_vals = []
        
        for obs in obstacles:
            h_val = float(np.asarray(obs.evaluate_h(p)).ravel()[0])
            grad_h = np.asarray(obs.gradient_h(p), dtype=np.float64).ravel()
            
            L_f_h = float(np.dot(grad_h, v))
            
            A_rows.append(-grad_h)
            b_vals.append((alpha1 + alpha2) * L_f_h + alpha1 * alpha2 * h_val)
            
        if include_domain:
            for d in range(D):
                h_top = self.config.domain_limit - p[d]
                L_f_top = -v[d]
                A_top = np.zeros(D)
                A_top[d] = 1.0
                A_rows.append(A_top)
                b_vals.append((alpha1 + alpha2) * L_f_top + alpha1 * alpha2 * h_top)
                
                h_bot = p[d] + self.config.domain_limit
                L_f_bot = v[d]
                A_bot = np.zeros(D)
                A_bot[d] = -1.0
                A_rows.append(A_bot)
                b_vals.append((alpha1 + alpha2) * L_f_bot + alpha1 * alpha2 * h_bot)
                
        A = np.vstack(A_rows)
        b = np.concatenate([np.atleast_1d(x) for x in b_vals])
        
        violations = A @ a_target - b
        if np.all(violations <= 1e-7):
            return np.clip(a_target, -self.config.a_max, self.config.a_max)
            
        def objective(a):
            diff = a - a_target
            return 0.5 * np.dot(diff, diff)
            
        def objective_grad(a):
            return a - a_target
            
        bounds = [(-self.config.a_max, self.config.a_max)] * D
        constraints = [{'type': 'ineq', 'fun': lambda a, A_mat=A, b_vec=b: b_vec - A_mat @ a,
                        'jac': lambda a, A_mat=A: -A_mat}]
                        
        res = minimize(
            objective,
            x0=np.clip(a_target, -self.config.a_max, self.config.a_max),
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-8, 'maxiter': 50, 'disp': False}
        )
        
        if res.success:
            return res.x
            
        worst_idx = int(np.argmax(violations))
        a_k = A[worst_idx]
        norm_sq = np.sum(a_k ** 2) + self.config.eps_reg
        a_fallback = a_target - (violations[worst_idx] / norm_sq) * a_k
        return np.clip(a_fallback, -self.config.a_max, self.config.a_max)

    def simulate_kinematic_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Any],
        dt: float = 0.01,
        max_steps: int = 500,
        goal_tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Symuluje trajektorię drona 3D / manipulatora w polu przeszkód KAN (kinematyka 1. rzędu).
        """
        p = np.asarray(start, dtype=np.float64).copy()
        goal = np.asarray(goal, dtype=np.float64)
        
        trajectory = [p.copy()]
        velocities = []
        h_min_history = []
        step_times = []
        
        success = False
        collision = False
        
        for step in range(max_steps):
            t0 = time.perf_counter()
            dist_to_goal = np.linalg.norm(goal - p)
            if dist_to_goal < goal_tolerance:
                success = True
                break
                
            u_nom = np.clip(self.config.kp_goal * (goal - p), -self.config.v_max, self.config.v_max)
            
            u_safe = self.solve_kinematic_qp(p, u_nom, obstacles)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1e6)
            
            h_vals = [float(np.asarray(obs.evaluate_h(p)).ravel()[0]) for obs in obstacles]
            if h_vals:
                min_h = min(h_vals)
                h_min_history.append(min_h)
                if min_h < -1e-4:
                    collision = True
                    
            velocities.append(u_safe)
            p = p + u_safe * dt
            trajectory.append(p.copy())
            
        return {
            "trajectory": np.array(trajectory),
            "velocities": np.array(velocities),
            "h_min_history": np.array(h_min_history),
            "success": success,
            "collision": collision,
            "steps": len(trajectory) - 1,
            "avg_latency_us": float(np.mean(step_times)) if step_times else 0.0,
            "max_latency_us": float(np.max(step_times)) if step_times else 0.0
        }

    def simulate_dynamic_trajectory(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Any],
        dt: float = 0.01,
        max_steps: int = 500,
        goal_tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Symuluje trajektorię drona 3D z dynamiką 2. rzędu (przyspieszenie / bezwładność HOCBF).
        """
        p = np.asarray(start_pos, dtype=np.float64).copy()
        v = np.asarray(start_vel, dtype=np.float64).copy()
        goal = np.asarray(goal, dtype=np.float64)
        
        trajectory = [p.copy()]
        velocities = [v.copy()]
        accelerations = []
        h_min_history = []
        step_times = []
        
        success = False
        collision = False
        
        for step in range(max_steps):
            t0 = time.perf_counter()
            dist_to_goal = np.linalg.norm(goal - p)
            if dist_to_goal < goal_tolerance and np.linalg.norm(v) < 0.5:
                success = True
                break
                
            a_nom = self.config.kp_goal * (goal - p) - 2.5 * v
            a_nom = np.clip(a_nom, -self.config.a_max, self.config.a_max)
            
            a_safe = self.solve_dynamic_hocbf_qp(p, v, a_nom, obstacles)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1e6)
            
            h_vals = [float(np.asarray(obs.evaluate_h(p)).ravel()[0]) for obs in obstacles]
            if h_vals:
                min_h = min(h_vals)
                h_min_history.append(min_h)
                if min_h < -1e-4:
                    collision = True
                    
            accelerations.append(a_safe)
            v = np.clip(v + a_safe * dt, -self.config.v_max, self.config.v_max)
            p = p + v * dt
            
            trajectory.append(p.copy())
            velocities.append(v.copy())
            
        return {
            "trajectory": np.array(trajectory),
            "velocities": np.array(velocities),
            "accelerations": np.array(accelerations),
            "h_min_history": np.array(h_min_history),
            "success": success,
            "collision": collision,
            "steps": len(trajectory) - 1,
            "avg_latency_us": float(np.mean(step_times)) if step_times else 0.0
        }

    def simulate_swarm(
        self,
        start_positions: np.ndarray,
        goal_positions: np.ndarray,
        obstacles: List[Any],
        dt: float = 0.01,
        max_steps: int = 400
    ) -> Dict[str, Any]:
        """
        Wielowątkowa / Wektorowa symulacja roju dronów (N agentów) z jednoczesnym unikaniem
        kolizji między sobą i omijaniem pól przeszkód KAN.
        """
        N, D = start_positions.shape
        P = np.asarray(start_positions, dtype=np.float64).copy()
        Goals = np.asarray(goal_positions, dtype=np.float64)
        
        trajectory_history = [P.copy()]
        inter_agent_min_dist = []
        obstacle_min_dist = []
        
        for step in range(max_steps):
            u_safe_swarm = np.zeros((N, D))
            
            for i in range(N):
                p_i = P[i]
                g_i = Goals[i]
                u_nom_i = np.clip(self.config.kp_goal * (g_i - p_i), -self.config.v_max, self.config.v_max)
                u_target_i = self._compute_tangential_guidance(p_i, u_nom_i, obstacles)
                
                A_rows = []
                b_vals = []
                
                for obs in obstacles:
                    h_val = float(np.asarray(obs.evaluate_h(p_i)).ravel()[0])
                    grad_val = np.asarray(obs.gradient_h(p_i), dtype=np.float64).ravel()
                    A_rows.append(-grad_val)
                    b_vals.append(self.config.alpha * h_val)
                    
                A_agents, b_agents = self.inter_agent.get_pairwise_constraints(i, P, self.config.alpha)
                A_rows.extend(A_agents)
                b_vals.extend(b_agents)
                
                A_dom, b_dom = self.domain_box.get_constraints(p_i, self.config.alpha)
                A_rows.append(A_dom)
                b_vals.append(b_dom)
                
                A = np.vstack(A_rows)
                b = np.concatenate([np.atleast_1d(x) for x in b_vals])
                
                violations = A @ u_target_i - b
                if np.all(violations <= 1e-7):
                    u_safe_swarm[i] = np.clip(u_target_i, -self.config.v_max, self.config.v_max)
                else:
                    def obj(u):
                        diff = u - u_target_i
                        return 0.5 * np.dot(diff, diff)
                    res = minimize(
                        obj,
                        x0=np.clip(u_target_i, -self.config.v_max, self.config.v_max),
                        bounds=[(-self.config.v_max, self.config.v_max)] * D,
                        constraints={'type': 'ineq', 'fun': lambda u, A_m=A, b_v=b: b_v - A_m @ u},
                        method='SLSQP',
                        options={'ftol': 1e-7, 'maxiter': 30, 'disp': False}
                    )
                    u_safe_swarm[i] = res.x if res.success else u_target_i
                    
            P = P + u_safe_swarm * dt
            trajectory_history.append(P.copy())
            
            diffs = P[:, None, :] - P[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            inter_agent_min_dist.append(float(np.min(dists)))
            
            if obstacles:
                obs_dists = [np.min([float(np.asarray(obs.evaluate_h(p)).ravel()[0]) for obs in obstacles]) for p in P]
                obstacle_min_dist.append(float(np.min(obs_dists)))
                
        return {
            "trajectory_history": np.array(trajectory_history),
            "min_inter_agent_dist": float(np.min(inter_agent_min_dist)) if inter_agent_min_dist else 0.0,
            "min_obstacle_dist": float(np.min(obstacle_min_dist)) if obstacle_min_dist else 0.0,
            "inter_agent_collision": bool(np.min(inter_agent_min_dist) < 2.0 * self.config.d_safe - 1e-4) if inter_agent_min_dist else False,
            "obstacle_collision": bool(np.min(obstacle_min_dist) < -1e-4) if obstacle_min_dist else False
        }
