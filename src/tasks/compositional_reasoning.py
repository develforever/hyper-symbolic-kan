import numpy as np
from typing import List, Tuple, Dict

class TransitiveReasoningTask:
    """
    Zadanie Przechodniego Rozumowania Relacyjnego (Transitive Compositional Reasoning).
    Generuje łańcuchy dedukcji logicznej (np. relacje przodków, struktur organizacji, relacji przestrzennych).
    """
    def __init__(self, num_entities: int = 20, chain_depth: int = 5):
        self.num_entities = num_entities
        self.chain_depth = chain_depth
        self.entities = [f"Entity_{i}" for i in range(num_entities)]
        self.relation_name = "is_ancestor_of"

    def generate_knowledge_base(self, num_chains: int = 50) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        """
        Generuje parami relacje bezpośrednie oraz pełną macierz domknięcia przechodniego (True Ground Truth).
        """
        np.random.seed(123)
        direct_edges = []
        closure_matrix = np.zeros((self.num_entities, self.num_entities), dtype=bool)

        for _ in range(num_chains):
            # Generujemy losowy łańcuch przechodni A -> B -> C -> D -> E
            chain_nodes = np.random.choice(self.num_entities, size=self.chain_depth, replace=False)
            for i in range(len(chain_nodes) - 1):
                u, v = chain_nodes[i], chain_nodes[i+1]
                direct_edges.append((u, v))
                closure_matrix[u, v] = True

        # Warstwy domknięcia przechodniego (Warshall / Transitive Closure)
        for k in range(self.num_entities):
            for i in range(self.num_entities):
                for j in range(self.num_entities):
                    closure_matrix[i, j] = closure_matrix[i, j] or (closure_matrix[i, k] and closure_matrix[k, j])

        labels = {}
        for i in range(self.num_entities):
            for j in range(self.num_entities):
                if i != j:
                    labels[(i, j)] = 1 if closure_matrix[i, j] else 0

        return direct_edges, labels
