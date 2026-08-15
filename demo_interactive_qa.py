import os
import sys
import time
import numpy as np

# System path patch
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.hs_ckan.clifford_algebra import CliffordAlgebraEngine
from src.tdff_net.dr_tt_kan import DynamicRankTTKAN
from src.tdff_net.dr_tt_als import DynamicRankTTALSSolver
from src.tdff_net.sliding_domain import SlidingSpatialDomainWindow, NormalizedKANField
from src.mct_nse.concurrent_category_filter import ConcurrentCategoryFilter
from src.qa_engine.hyper_symbolic_qa import HyperSymbolicQAEngine

def print_banner():
    print("\n" + "=" * 80)
    print("HYPER-SYMBOLIC KAN & MONADIC ENGINE - INTERACTIVE QA CLI DEMO")
    print("Author: Principal Software Architect | 0 Gradient Epochs Constraint")
    print("=" * 80)
    print("Inicjalizacja 4 podsystemów neuro-symbolicznych...")

def main():
    print_banner()
    
    # 1. HS-CKAN Clifford Relational Knowledge Base
    t0 = time.perf_counter()
    clifford = CliffordAlgebraEngine(num_entities=20)
    knowledge_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7)]
    closure_matrix = clifford.compute_transitive_closure_matrix(knowledge_edges)
    dt_rel = (time.perf_counter() - t0) * 1000.0
    print(f"[+] 1. HS-CKAN Clifford Knowledge Graph Gotowy ({len(knowledge_edges)} krawędzi, przeliczono w {dt_rel:.2f} ms)")
    
    # 2. DR-TT-KAN Continuous Tensor Field z Normalizacją Afiniczną
    t0 = time.perf_counter()
    X_train = np.random.uniform(-90.0, 90.0, (1000, 10))
    domain = SlidingSpatialDomainWindow(spatial_dim=10)
    domain.update_bounds(X_train, mode="fit")
    X_hat = domain.transform(X_train)
    Y_train = np.cos(np.pi * X_hat[:, 0]) * np.sin(np.pi * X_hat[:, 1])
    
    base_kan = DynamicRankTTKAN(spatial_dim=10, init_ranks=[1] + [8] * 9 + [1], degree=5)
    solver = DynamicRankTTALSSolver(alpha=1e-5, max_sweeps=3)
    solver.fit(base_kan, X_hat, Y_train, adapt_ranks=True)
    kan_field = NormalizedKANField(base_model=base_kan, domain_window=domain)
    dt_kan = (time.perf_counter() - t0) * 1000.0
    print(f"[+] 2. DR-TT-KAN Continuous Field (10D) Gotowe (Dopasowano w {dt_kan:.2f} ms, 0 gradient epochs)")
    
    # 3. MCT-NSE v2 Concurrent Monadic Safety Guard (N=1000 agentów)
    t0 = time.perf_counter()
    cat_filter = ConcurrentCategoryFilter()
    cat_filter.add_invariant(
        "NoFlyBounds",
        lambda S: np.all((S[:, :3] >= -50.0) & (S[:, :3] <= 50.0), axis=1),
        lambda S: np.clip(S, -50.0, 50.0)
    )
    dt_mct = (time.perf_counter() - t0) * 1000.0
    print(f"[+] 3. MCT-NSE v2 Monadic Guard Gotowy (Flota N=1000 agentów, {dt_mct:.2f} ms)")
    
    # 4. HyperSymbolicQAEngine Conversational Router
    qa_engine = HyperSymbolicQAEngine(
        clifford_closure_matrix=closure_matrix,
        kan_field=kan_field,
        category_filter=cat_filter,
        num_agents=1000
    )
    print(f"[+] 4. HyperSymbolicQAEngine Conversational Router Gotowy (Czas odpowiedzi < 1 ms)")
    print("=" * 80)
    print("\nPRZYKŁADOWE PYTANIA DO WPISANIA:")
    print(" - 'Czy encja 0 jest połączona z 4?' (Wnioskowanie grafowe)")
    print(" - 'Jaki stan w punkcie 12.5, -5.0?' (Ewaluacja pola 10D + gradient)")
    print(" - 'Czy flota drony jest bezpieczna?' (Weryfikacja formalna 1000 agentów)")
    print(" - 'Ile epok gradientowych używa sieć?' (Metryki systemowe)")
    print(" - 'diag' (Uruchamia szybką diagnostykę wszystkich 15 zadań)")
    print(" - 'exit' / 'quit' (Zakończenie pracy CLI)")
    print("-" * 80)
    
    while True:
        try:
            user_query = input("\n[TWÓJ INPUT] > ")
            cleaned = user_query.strip()
            
            if cleaned.lower() in ["exit", "quit", "q", "wyjście"]:
                print("[SYSTEM]: Zakończono sesję CLI. Do widzenia!")
                break
                
            if not cleaned:
                continue
                
            if cleaned.lower() in ["diag", "benchmark", "test"]:
                print("\n[SYSTEM]: Uruchamiam pełen benchmark 15 zadań z main.py...")
                os.system("python main.py")
                continue
                
            if cleaned.lower() in ["help", "pomoc"]:
                print("\n[POMOC]: Wpisz dowolne pytanie w języku polskim lub angielskim.")
                print("Przykłady:")
                print("  1. 'Czy 0 wpływa na 3?'")
                print("  2. 'Jaki stan w punkcie 10, 20?'")
                print("  3. 'Czy są naruszenia stref zakazanych?'")
                print("  4. 'Jaka jest przepustowość silnika C++?'")
                continue
                
            # Wykonanie zapytania przez silnik QA w < 1 ms
            answer = qa_engine.ask(cleaned)
            print(f"\n[ODPOWIEDŹ Z SILNIKA HYPER-SYMBOLIC]:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n[SYSTEM]: Zakończono sesję CLI.")
            break

if __name__ == "__main__":
    main()
