# Strategiczna Roadmapa Rozwoju: Hyper-Symbolic KAN (Kolejne Sesje)

## Spis Treści i Kolejność Realizacji

```mermaid
flowchart TD
    A["Etap A: Natywny Build System (C++ / nanobind / scikit-build-core)"] --> B["Etap B: Zaawansowane Solvery Tensorowe (TT-Cross & DMRG)"]
    B --> C["Etap C: Natywny WebGPU WGSL Compute Shader (100k - 500k Agentów)"]
    C --> D["Etap D: Integracja z Ekosystemem PyTorch / JAX (nn.Module Custom Ops)"]
    D --> E["Etap E: Wdrożenia Branżowe (Robotyka, Bezsiatkowe PDE / PINN)"]
```

---

## Szczegółowy Zakres Prac dla Poszczególnych Etapów

### Etap A: Natywny Build System & Standaryzacja C++ / SIMD (Najbliższa sesja)
- **Cel**: Wyeliminowanie fallbacków do Pythona w runtime, zastąpienie `ctypes` nowoczesnymi, bezoverheadowymi bindingami **`nanobind`** oraz standaryzacja procesu budowania przez **`scikit-build-core`** / **`CMake`**.
- **Zadania techniczne**:
  1. Konfiguracja `CMakeLists.txt` z flagami optymalizacyjnymi AVX2 / AVX-512 oraz OpenMP dla platform Windows (MSVC), Linux (GCC/Clang) i macOS (Clang/Apple Silicon).
  2. Implementacja modułu C++ z wykorzystaniem `nanobind` eksportującego szybkie funkcje: `evaluate_tt_kan_batch`, `evaluate_tt_kan_gradient_batch`, `evaluate_cp_kan_batch`.
  3. Konfiguracja `pyproject.toml` z backendem `scikit-build-core` umożliwiającym natywną instalację: `pip install -e .`.
  4. Testy wydajnościowe porównujące narzut wywołania `ctypes` vs `nanobind` vs Python SIMD.

---

### Etap B: Zaawansowane Solvery Tensorowe: Algorytm TT-Cross & DMRG
- **Cel**: Eliminacja ograniczenia gęstego próbkowania siatki w ALS i skalowanie do wymiarów $D = 20 \dots 100$.
- **Zadania techniczne**:
  1. Implementacja algorytmu **TT-Cross (Oseledets)** opartego na macierzach interpolacji maksymalnej objętości (MaxVol).
  2. Implementacja jednowęzłowego i dwuwęzłowego **DMRG** (*Density Matrix Renormalization Group*) do bezgradientowego dopasowywania pól w wysokich wymiarach.
  3. Redukcja złożoności próbkowania z $O(N^D)$ do $O(D \cdot R^2 \cdot K)$ punktów.

---

### Etap C: Natywny WebGPU WGSL Compute Shader Pipeline (100k+ Agentów)
- **Cel**: Przeniesienie symulacji fizycznej i nawigacji roju cząstek w 100% do pamięci VRAM na karcie graficznej.
- **Zadania techniczne**:
  1. Napisanie dedykowanego kernela **WGSL Compute Shader** (`@compute @workgroup_size(64)`), który wykonuje rekurencję Czebyszewa, ewaluację gradientu $\nabla f$ i integrację kroków pozycji w GPU.
  2. Zastosowanie `GPUBuffer` typu `STORAGE | VERTEX` umożliwiającego bezpośrednie renderowanie pozycji cząstek bez transferu danych GPU $\leftrightarrow$ CPU.
  3. Zwiększenie liczby symulowanych agentów z 10 000 do **100 000 – 500 000 agentów w 60 FPS**.

---

### Etap D: Integracja z Ekosystemem PyTorch / JAX (ZREALIZOWANE)
- **Cel**: Umożliwienie bezproblemowego stosowania pól KAN w potokach Deep Learning.
- **Zrealizowane zadania**:
  1. Klasy `ContinuousKANAutograd` i `TensorTrainKANAutograd` (`torch.autograd.Function`) z analitycznymi gradientami $O(1)$ pamięci grafu i integracją z natywnym kernelem C++ nanobind/SIMD.
  2. Warstwy `ContinuousKANLayer(nn.Module)` oraz `TensorTrainKANLayer(nn.Module)` z pełną obsługą batchingu N-D, mostkami do solverów ALS/TT-Cross (0 epok) oraz fine-tuningiem gradientowym `torch.optim.Adam`.
  3. Standaryzacja zapisu i odczytu `.safetensors` (`src/torch_kan/safetensors_io.py`) z zachowaniem metadanych topologii tensorowej.
  4. Kompletny zestaw 9 testów jednostkowych i integracyjnych (`tests/test_torch_kan.py`), w tym rygorystyczny `torch.autograd.gradcheck` (100% PASS, 27/27 testów w całym repozytorium).

---

### Etap E: Aplikacje Praktyczne & Wdrożenia Branżowe
- **Cel**: Demonstracja przewagi technologii w zastosowaniach przemysłowych.
- **Zadania techniczne**:
  1. **Robotyka / Drony (Zero-Collision Trajectory Planner)**: Sprzężenie pola KAN z funkcjami barierowymi CBF (*Control Barrier Functions*) do planowania trajektorii w czasie rzeczywistym ($< 0.1\text{ ms}$).
  2. **Physics-Informed Neural Fields (Bezsiatkowe PDE)**: Rozwiązywanie stacjonarnych równań Poissona, Laplace'a i Naviera-Stokesa poprzez analityczny operator Laplace'a $\nabla^2 f$ w 0 epokach gradientowych.
