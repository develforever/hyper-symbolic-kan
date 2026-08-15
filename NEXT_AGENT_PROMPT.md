# PROMPT DLA KOLEJNEGO AGENTA / NOWEJ SESJI (ETAP A)

Skopiuj i wklej poniższy blok w nowym czacie:

```markdown
Działaj jako Principal Software Architect. Rozpoczynamy **Etap A** z dokumentu `ROADMAP_NEXT_SESSIONS.md` w repozytorium `hyper_symbolic_kan`.

### Kontekst Techniczny:
W poprzedniej sesji przeprowadziliśmy audyt matematyczny, naprawiliśmy bufory w `fast_kan_kernel.cpp`, ustabilizowaliśmy solver ALS z normalizacją kolumn, dodaliśmy zestaw testów w `tests/run_all_tests.py` (7/7 zaliczonych) oraz zbudowaliśmy demonstrator 3D w WebGPU/React Three Fiber (`web_showcase/`).

### Cel Tej Sesji (Etap A: Natywny Build System & Standaryzacja C++):
1. **Eliminacja `ctypes` i fallbacku `subprocess`**: Zastąpienie mechanizmu dynamicznego kompilowania w locie nowoczesnymi, ultrawydajnymi bindingami **`nanobind`** lub `pybind11`.
2. **Konfiguracja `CMakeLists.txt`**:
   - Włączenie wektoryzacji SIMD (AVX2 / AVX-512) i wielowątkowości OpenMP.
   - Wsparcie dla Windows (MSVC), Linux (GCC/Clang) i macOS.
3. **Standaryzacja dystrybucji (`pyproject.toml` + `scikit-build-core`)**:
   - Skonfigurowanie projektu tak, aby instalował się czysto przez `pip install -e .` tworząc prekompilowany moduł C++ `hyper_symbolic_kan._cpp_kernels`.
4. **Weryfikacja Wydajnościowa**:
   - Napisanie benchmarku i testu weryfikującego przepustowość wywołań C++ (cel: > 1 500 000 punktów/s w batchu, opóźnienie wywołania < 0.2 us).

Przeanalizuj repozytorium, zapoznaj się z `ROADMAP_NEXT_SESSIONS.md` i `src/cpp_kernels/` i przedstaw plan wdrożenia Etapu A.
```
