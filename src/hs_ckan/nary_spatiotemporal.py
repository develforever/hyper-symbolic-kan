import numpy as np
from typing import List, Dict, Tuple
from src.hs_ckan.chebyshev_kan import ChebyshevKANBasis
from src.hs_ckan.closed_form_solver import ClosedFormLayerSolver

class CleanUpMemory:
    r"""
    Nieliniowa Pamięć Czyszcząca (Subspace Winner-Take-All & Block Thresholding).
    
    Operuje w czasie O(1) bez uczenia gradientowego. Dzieli wektor faktów na podprzestrzenie
    symboliczne i przestrzenne, po czym aplikuje rzutowanie prógowane WTA:
    - Predykaty & Agenci: Argmax / One-Hot clean-up.
    - Strefy Przestrzenne: Filtracja norm blokowych podprzestrzeni (zeroes out noise in inactive zones).
    """
    def __init__(self, num_predicates: int, num_entities: int, st_dim: int):
        self.num_predicates = num_predicates
        self.num_entities = num_entities
        self.st_dim = st_dim

    def cleanup(self, vec: np.ndarray) -> np.ndarray:
        # 1. Czyszczenie predykatu (WTA argmax)
        pred_part = vec[:self.num_predicates]
        pred_clean = np.zeros_like(pred_part)
        pred_clean[np.argmax(pred_part)] = 1.0

        # 2. Czyszczenie agenta (WTA argmax)
        agent_offset = self.num_predicates
        agent_part = vec[agent_offset : agent_offset + self.num_entities]
        agent_clean = np.zeros_like(agent_part)
        agent_clean[np.argmax(agent_part)] = 1.0

        # 3. Czyszczenie podprzestrzeni strefowo-przestrzennej (Block-WTA)
        spatial_offset = agent_offset + self.num_entities
        spatial_part = vec[spatial_offset:].reshape(self.num_entities, self.st_dim)
        
        row_norms = np.linalg.norm(spatial_part, axis=1)
        best_zone = np.argmax(row_norms)
        
        spatial_clean = np.zeros_like(spatial_part)
        spatial_clean[best_zone] = spatial_part[best_zone]

        return np.concatenate([pred_clean, agent_clean, spatial_clean.ravel()])

class NarySpatioTemporalEngine:
    r"""
    Rozszerzony Silnik HS-CKAN dla Predykatów N-argumentowych oraz Relacji Przestrzenno-Czasowych.
    
    Architektura:
    - Wiązanie Tensorowe (Symbolic-Spatial Tensor Outer Product Binding):
      v_bound = (e_{pred} ⊕ e_{agent}) ⊗ (e_{zone} ⊗ KAN(x, y, z, t))
    - Ciągłe Zmienne Przestrzenno-Czasowe zakodowane bazą Czebyszewa KAN T_k(x) ⊗ T_m(y).
    - Analityczne Czyszczenie Szumu (Clean-Up Memory + Ridge Projection) w czasie O(1).
    """
    def __init__(self, num_entities: int, num_predicates: int, spatial_dim: int = 4, kan_degree: int = 4, use_cleanup: bool = True):
        self.num_entities = num_entities
        self.num_predicates = num_predicates
        self.spatial_dim = spatial_dim
        self.kan_degree = kan_degree
        self.use_cleanup = use_cleanup
        
        self.entity_basis = np.eye(self.num_entities)
        self.predicate_basis = np.eye(self.num_predicates)
        self.cleaner_solver = ClosedFormLayerSolver(alpha=1e-3)
        
        num_pairs = spatial_dim * (spatial_dim - 1) // 2
        st_dim = spatial_dim * (kan_degree + 1) + num_pairs * ((kan_degree + 1) ** 2)
        self.cleanup_memory = CleanUpMemory(num_predicates, num_entities, st_dim)
        
    def _1d_chebyshev(self, x: np.ndarray) -> np.ndarray:
        x_norm = np.clip(x, -1.0, 1.0)
        N = len(x_norm)
        T = np.empty((N, self.kan_degree + 1))
        T[:, 0] = 1.0
        if self.kan_degree >= 1:
            T[:, 1] = x_norm
        for k in range(2, self.kan_degree + 1):
            T[:, k] = 2.0 * x_norm * T[:, k - 1] - T[:, k - 2]
        return T

    def encode_spatiotemporal_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Symetryczne i izotropowe kodowanie wielowymiarowe dla dowolnego wymiaru D.
        Zawiera składowe 1D dla wszystkich osi oraz pełne pary korelacji nieliniowych (d1, d2).
        """
        N, D = coords.shape
        bases_1d = [self._1d_chebyshev(coords[:, d]) for d in range(D)]
        
        linear_features = np.hstack(bases_1d) # (N, D * (K+1))
        cross_features = []
        for i in range(D):
            for j in range(i + 1, D):
                cross = (bases_1d[i][:, :, np.newaxis] * bases_1d[j][:, np.newaxis, :]).reshape(N, -1)
                cross_features.append(cross)
                
        if cross_features:
            prod_features = np.hstack([linear_features] + cross_features)
        else:
            prod_features = linear_features
            
        return prod_features

    def bind_nary_fact(self, predicate_id: int, entity_ids: List[int], coords: np.ndarray) -> np.ndarray:
        """
        Entity 0: Agent ID, Entity 1: Zone ID
        Tensor product between Zone entity vector and Spatial KAN feature vector.
        """
        agent_vec = self.entity_basis[entity_ids[0]]
        zone_vec = self.entity_basis[entity_ids[1]]
        pred_vec = self.predicate_basis[predicate_id]
        
        st_vector = self.encode_spatiotemporal_coords(coords.reshape(1, -1)).squeeze(0)
        
        # Interakcja przestrzenno-strefowa via Kronecker product
        zone_spatial_bound = np.kron(zone_vec, st_vector)
        
        # Konkatenacja z pozostałymi wskaźnikami symbolicznymi
        fact_representation = np.concatenate([pred_vec, agent_vec, zone_spatial_bound])
        return fact_representation

    def fit_knowledge_base(self, facts: List[Tuple[int, List[int], np.ndarray]], target_labels: np.ndarray):
        X_list = []
        for pred_id, ent_ids, coords in facts:
            vec = self.bind_nary_fact(pred_id, ent_ids, coords)
            X_list.append(vec)
            
        X = np.vstack(X_list)
        self.cleaner_solver.fit(X, target_labels)
        
    def query_with_noise(self, facts_noisy: List[Tuple[int, List[int], np.ndarray]], noise_level: float = 0.0) -> np.ndarray:
        X_list = []
        for pred_id, ent_ids, coords in facts_noisy:
            noisy_coords = coords.copy()
            if noise_level > 0.0:
                noisy_coords += np.random.normal(0.0, noise_level, size=coords.shape)
                noisy_coords = np.clip(noisy_coords, -1.0, 1.0)
                
            vec = self.bind_nary_fact(pred_id, ent_ids, noisy_coords)
            
            if noise_level > 0.0:
                noise_vec = np.random.normal(0.0, noise_level, size=vec.shape)
                vec = vec + noise_vec
                
            if self.use_cleanup and noise_level > 0.0:
                vec = self.cleanup_memory.cleanup(vec)
                
            X_list.append(vec)
            
        X_noisy = np.vstack(X_list)
        predictions = self.cleaner_solver.predict(X_noisy)
        return predictions

