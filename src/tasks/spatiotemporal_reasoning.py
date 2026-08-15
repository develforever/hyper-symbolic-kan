import numpy as np
from typing import List, Tuple, Dict
from src.hs_ckan.nary_spatiotemporal import NarySpatioTemporalEngine

class SpatioTemporalReasoningTask:
    """
    Zadanie Rozumowania Przestrzenno-Czasowego na Predykatach N-argumentowych z Szumem.
    
    Tworzy zbiór faktów: LocatedAt(Agent_i, Zone_j, x, y, z, t).
    Ocenia odporność algebraicznego wiązania KAN na zakłócenia szumowe wejścia.
    """
    def __init__(self, num_entities: int = 15, num_zones: int = 4, num_facts: int = 500):
        self.num_entities = num_entities
        self.num_zones = num_zones
        self.num_facts = num_facts
        self.num_predicates = 2 # 0: LocatedAt, 1: InteractsWith

    def generate_dataset(self, seed: int = 42) -> Tuple[List[Tuple[int, List[int], np.ndarray]], np.ndarray]:
        np.random.seed(seed)
        facts = []
        labels = []
        
        # Definicja stref przestrzennych w przedziale [-1, 1]^4
        for _ in range(self.num_facts):
            pred_id = 0 # LocatedAt
            agent_id = np.random.randint(0, self.num_entities)
            zone_id = np.random.randint(0, self.num_zones)
            
            # Generowanie współrzędnych (x, y, z, t)
            coords = np.random.uniform(-1.0, 1.0, size=4)
            
            # Etykieta prawdy: Czy agent w strefie spełnia określony warunek relacyjny?
            # np. zone_id określa kwadrant przestrzenny x > 0 i y > 0
            is_valid_zone = (coords[0] > 0 and coords[1] > 0) if zone_id % 2 == 0 else (coords[0] <= 0 or coords[1] <= 0)
            target = 1.0 if is_valid_zone else 0.0
            
            facts.append((pred_id, [agent_id, zone_id], coords))
            labels.append(target)
            
        return facts, np.array(labels).reshape(-1, 1)

    def evaluate_noise_robustness(self, noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]) -> Dict[float, float]:
        train_facts, train_labels = self.generate_dataset(seed=42)
        test_facts, test_labels = self.generate_dataset(seed=1337)
        
        # Inicjalizacja silnika
        engine = NarySpatioTemporalEngine(
            num_entities=max(self.num_entities, self.num_zones),
            num_predicates=self.num_predicates,
            spatial_dim=4,
            kan_degree=5
        )
        
        # Dopasowanie bazy wiedzy w czasie zamkniętym (0 gradient epochs)
        engine.fit_knowledge_base(train_facts, train_labels)
        
        results = {}
        for noise in noise_levels:
            preds = engine.query_with_noise(test_facts, noise_level=noise)
            binary_preds = (preds > 0.5).astype(float)
            acc = float(np.mean(binary_preds == test_labels)) * 100.0
            results[noise] = acc
            
        return results
