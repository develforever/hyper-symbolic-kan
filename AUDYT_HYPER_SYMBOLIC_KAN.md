# Audyt architektoniczny i plan skoku jakościowego — `hyper_symbolic_kan`

**Rewizja:** `d3bf28d` (TASK-F5) · **Data:** 2026-08-16 · **Rola:** Principal Software Architect / Lead AI Systems Engineer

---

## 0. Zakres i metodyka

**Przeczytane w całości:** `src/tdff_net/` (14 plików), `src/hs_ckan/` (4), `src/mct_nse/` (5), `src/cpp_kernels/` (4: `.hpp`, `.cpp`, bindings, `cpp_kan_engine.py`), `src/applications/` (2), `src/torch_kan/` (3), `src/jax_kan/` (częściowo: `autograd_ops.py` + `layers.py`), `src/facade.py`, `CMakeLists.txt`, `pyproject.toml`, `tests/*`, `benchmarks/benchmark_vs_pykan.py`, fragmenty `web_showcase/src/shaders/`.

**Nie przeczytane (poza budżetem, nie wpływa na wnioski):** pełne `web_showcase/src/components/*` (~5 kLOC TSX), `src/tasks/*` (skrypty demonstracyjne), `packages/hyper-kan-webgpu/`.

**Zweryfikowane empirycznie (nie z lektury, lecz pomiarem):**

1. Uwarunkowanie układu normalnego w `SpectralKANPoissonSolver` — odtworzona konstrukcja macierzy, policzone `cond(A)` i `cond(AᵀA)`.
2. Zakres dynamiczny iloczynu CP `∏_d φ_d` w funkcji `D` przy inicjalizacji z `TDFFNet.__init__`.
3. Wartość `T'_K(±1) = K²` — magnitudo gradientu zwracanego dla punktów poza dziedziną po `np.clip`.

**Wyniki pomiaru 1** (β_bc = 200, α_reg = 1e-9, jak w kodzie):

| D | K (degree) | M = (K+1)^D | κ(A) | κ(AᵀA) | cyfr znaczących w f64 |
|---|---|---|---|---|---|
| 2 | 6 | 49 | 2.2e+01 | 5.0e+02 | 13.3 |
| 2 | 10 | 121 | 6.6e+01 | 4.4e+03 | 12.4 |
| 2 | 18 | 361 | 1.5e+03 | 2.3e+06 | 9.6 |
| 3 | 6 | 343 | 1.7e+02 | 3.0e+04 | 11.5 |
| **3** | **10** | **1331** | **1.1e+06** | **1.3e+12** | **3.9** |

Ostatni wiersz to **domyślna konfiguracja fasady** (`PoissonSolver(dim=3, degree=10)`). Solver zwraca ~4 cyfry znaczące, raportując `pde_residual_rmse` liczone tą samą, źle uwarunkowaną macierzą — czyli metryka jakości jest ślepa na własny błąd.

**Wyniki pomiaru 2** (rank=16, degree=5, inicjalizacja jak w `TDFFNet`):

| D | mediana \|∏φ\| | max \|∏φ\| |
|---|---|---|
| 3 | 1.06e-01 | 8.69e+00 |
| 10 | 7.53e-05 | 3.93e+00 |
| 20 | 6.04e-09 | 2.83e-02 |
| 50 | 7.58e-21 | 5.98e-09 |
| 100 | **5.99e-42** | 5.22e-24 |

`FLT_MIN = 1.18e-38`. Rozkład CP w **float32** (ścieżka WebGPU i domyślna ścieżka JAX) twardo zeruje się przy D ≳ 60 i traci połowę mantysy już przy D ≈ 25. To nie jest kwestia strojenia — to własność iloczynu D czynników o medianie |φ| < 1.

---

## 1. Tabela audytu krytycznego

Priorytetyzacja: **P0** = błąd poprawności lub bezpieczeństwa pamięci; **P1** = ściana skalowalności lub fałszywa deklaracja w API; **P2** = dług techniczny o wysokim koszcie utrzymania.

### 1.1 Poprawność matematyczna (P0)

| # | Komponent | Problem / wąskie gardło | Ryzyko | Rekomendowane rozwiązanie |
|---|---|---|---|---|
| M1 | `applications/robotics_cbf_planner.py::solve_dynamic_hocbf_qp` | HOCBF pomija człon Hessianu. Warunek eksponencjalnego CBF wymaga `ḧ = ∇h·a + vᵀ∇²h v`. Kod buduje ograniczenie wyłącznie z `∇h·a`, milcząco zakładając `∇²h ≡ 0`. Dla `SyntheticSphereObstacle` `∇²h = (I − êêᵀ)/‖p−c‖ ≠ 0`; dla pola KAN krzywizna jest istotą reprezentacji. | Naruszenie bariery rośnie jak `‖v‖²·λ_max(∇²h)`. Przy `v_max=2`, `d=0.1` człon pominięty ma rząd 40 m/s² — czterokrotność `a_max`. Deklaracja „0% violations" jest fałszywa w reżimie dynamicznym. | Dodać `hessian_h(X)`. Baza istnieje: `chebyshev_derivatives_2nd` w `pde_poisson_solver.py` daje `d²T_k`. Dla CP-KAN `∂²f/∂x_i∂x_j = Σ_r λ_r · dφ_i^{(r)} dφ_j^{(r)} ∏_{m≠i,j} φ_m^{(r)}` (i≠j) oraz `d²φ_i ∏_{m≠i} φ_m` (i=j) — koszt `O(N·R·D)` z prefiksami/sufiksami, bez `O(D²)`. Ograniczenie: `−∇hᵀa ≤ (α₁+α₂)(∇hᵀv) + α₁α₂h + vᵀ∇²h v`. |
| M2 | `applications/pde_poisson_solver.py::TTPoissonSolver.fit_als` | Macierz projektowa `LapPhi_int` budowana wyłącznie z członu `m = d` (`d2T_d`). Człony `m ≠ d` laplasjanu **też są liniowe w rdzeniu d** (rdzeń d występuje w prefiksie `L` i sufiksie `R` tych członów) i są całkowicie pominięte. Komentarz w kodzie („części ze stałych pozostałych drugich pochodnych") przyznaje brak. | ALS minimalizuje residuum innego operatora niż `∇²`. Solver **nie rozwiązuje równania Poissona dla D ≥ 4**. Zwracany `rmse_res` liczony poprawnym `laplacian()`, więc raportuje wysoki błąd — ale nikt nie assertuje progu. | `LapPhi_d = Σ_{m} Φ^{(m)}_d`, gdzie dla `m = d` używamy `d²T_d`, a dla `m ≠ d` — prefiksów/sufiksów policzonych z `d²M_m` po jednej stronie i `M` po drugiej: `Φ^{(m)}_d = L^{(m,d)}_{n,r} T_{n,k} R^{(m,d)}_{n,s}`. Jedna dodatkowa para przebiegów prefiksowych na sweep → koszt `O(D)`, nie `O(D²)`. |
| M3 | `tdff_net/tucker_als.py::TuckerALSSolver.fit` | Cała logika za `if D == 2:`. Dla `D ≠ 2` pętla ALS wykonuje się bezczynnie, rdzeń i czynniki nigdy nie są aktualizowane. **Brak wyjątku.** Zwracane MSE to MSE losowej inicjalizacji. | Cicha awaria. `TuckerTDFFNet(spatial_dim=3)` „uczy się" w 0 epok i zwraca szum, a metryka nie sygnalizuje problemu. | Albo implementacja ogólna (rozwinięcie modowe `Y ≈ W_d G_(d) (⊗_{j≠d} W_j)ᵀ` przez `tensordot`), albo `raise NotImplementedError(f"TuckerALS supports D=2, got D={D}")` w pierwszej linii. Wariant drugi natychmiast, wariant pierwszy w Fazie 2. |
| M4 | `applications/robotics_cbf_planner.py` (kinematyka + rój) | Ścieżka awaryjna po niepowodzeniu SLSQP: rzut na **jedno** najbardziej naruszone ograniczenie, następnie `np.clip(u, ±v_max)`. Clipping po rzucie niszczy spełnienie CBF. W `simulate_swarm` fallback to `u_target_i` — sterowanie **całkowicie niefiltrowane**. | Dokładnie w reżimie, w którym filtr jest potrzebny (QP niedopuszczalne: róg dziedziny + przeszkoda), system zwraca sterowanie niebezpieczne. | (a) Miękkie ograniczenia ze zmienną luzu δ ≥ 0 i karą `ρδ²`, ρ ≈ 1e4 → QP zawsze dopuszczalne. (b) Rzut na **przecięcie** aktywnych półprzestrzeni (Dykstra / dual active-set), nie na jedną. (c) `v_max` jako ograniczenie pudełkowe **wewnątrz** QP, nigdy jako post-clip. (d) Zwracać `CBFResult{u, slack, feasible, active_set}` zamiast gołego wektora — wywołujący musi widzieć degradację. |
| M5 | `applications/robotics_cbf_planner.py::InterAgentCBF` | Zdecentralizowany CBF bez podziału odpowiedzialności. Agent `i` zakłada `ṗ_j = 0`. Obaj agenci jednocześnie spełniają własne ograniczenie, para narusza wspólne. | Kolizje między agentami przy zbliżeniach czołowych, mimo raportowanego „0 collisions" (metryka mierzy `min_dist` po kroku Eulera, więc łapie tylko część przypadków). | Klasyczny podział: `−∇h_ij·u_i ≤ ½·α·h_ij` dla obu stron (gwarancja przy symetrycznych priorytetach), albo pełny CBF względny na `h(p_i−p_j, v_i−v_j)`. Udokumentować wybraną konwencję jako niezmiennik. |
| M6 | `hs_ckan/clifford_algebra.py` | Nazwa i docstring deklarują algebrę geometryczną Cℓ(N) z relacją `e_ie_j + e_je_i = 2δ_ij`. Implementacja: antysymetryczna macierz sąsiedztwa N×N + mnożenie macierzowe + `setdiag(0)`. `setdiag(0)` usuwa **dokładnie część skalarną (grade-0)** iloczynu geometrycznego. Relacja Clifforda nigdzie nie jest egzekwowana ani testowana. | Fundament „hyper-symboliczny" nie realizuje deklarowanej algebry. Każda zewnętrzna weryfikacja formalna tego modułu upadnie na pierwszym pytaniu. | Decyzja architektoniczna, nie łatka: **albo** przemianować na `SparseRelationAlgebra` i usunąć terminologię Clifforda, **albo** zaimplementować rzeczywistą Cℓ(p,q) na reprezentacji multiwektorowej z rzutami gradacji (`⟨AB⟩_k`), z testem własności `e_i² = 1`, antykomutacji i asocjacyjności. Rekomendacja: pierwsze — reprezentacja rzadka jest właściwa dla grafów, terminologia jest zbędnym zobowiązaniem. |
| M7 | `hs_ckan/clifford_algebra.py::compute_transitive_closure_matrix` | `max_depth=50` obcina domknięcie: ścieżki dłuższe niż 50 krawędzi są **cicho pomijane**. Akumulacja `closure += M^k` w `float32` zlicza ścieżki — dla grafu o średnim stopniu 10 wartości osiągają 1e50 → `inf` przy k≈38. Brak wczesnego wyjścia (warunek `nnz == 0` nigdy nie zachodzi dla grafów cyklicznych). Zwracany typ: `np.ndarray` dla N≤2000, `csr_matrix` powyżej — sygnatura deklaruje `csr_matrix`. | Niepoprawne domknięcie (deklaracja „100% dokładności"). Typ niestabilny → wywołujący musi rozgałęziać. Złożoność nie jest `O(|E|·depth)` — `nnz(M^k)` rośnie do `O(N²)`. | Zamiast potęgowania: **kondensacja SCC (Tarjan) + osiągalność na DAG-u z bitsetami** — `O(V+E)` na kondensację, `O(V·E/64)` na osiągalność, wynik dokładny bez limitu głębokości. Alternatywa zachowawcza: binaryzacja po każdym kroku (`M.data[:] = 1`) + wyjście gdy `nnz` przestaje rosnąć + `dtype=bool`. Typ zwracany: zawsze `csr_matrix[bool]`, z osobnym `.to_dense()`. |
| M8 | `torch_kan/autograd_ops.py` | `_compute_chebyshev_torch` **nie klipuje** `x` do [−1,1]; wszystkie ścieżki NumPy i C++ klipują. Dla `|x|>1` rekurencja rośnie jak `cosh(K·arccosh|x|)`. | Rozbieżność wyników między backendem CPU/f64 (C++, klipuje) a GPU/f32 (torch, nie klipuje) dla tych samych wag i danych. Przepełnienie przy `|x| = 1.5, K = 16` (≈1e5). Żaden test nie porównuje backendów na danych poza dziedziną. | Klipowanie w jednym miejscu — najlepiej **usunąć je z jąder** i wprowadzić jawną warstwę `DomainWindow` (`sliding_domain.py` już to ma) jako obowiązkowy element kontraktu. Jądro dostaje wyłącznie znormalizowane wejście i **assertuje** je w trybie debug. Milczące klipowanie jest gorsze niż błąd: patrz M9. |
| M9 | wszystkie moduły z `np.clip(x, -1, 1)` | Klipowanie nie propaguje się do gradientu. Dla punktu poza dziedziną gradient = `T'_K(±1) = K²` (zmierzone: 25 / 100 / 256 / 576 dla K = 5/10/16/24) zamiast 0. | Planer CBF pytany o barierę poza `[-1,1]^D` dostaje **stałą wartość h i ogromny, kierunkowo błędny gradient**. `DomainBoxCBF(limit=0.95)` ogranicza to w happy-path, ale nie po awarii QP (M4). | Kontrakt: `evaluate`/`gradient` odrzucają wejście poza dziedziną (`ValueError`) lub zwracają `nan`, konfigurowalnie. Ekstrapolacja — jeśli potrzebna — jawnie liniowa: `f(x) ≈ f(x_c) + ∇f(x_c)·(x−x_c)`, co daje spójne h i ∇h. |

### 1.2 Bezpieczeństwo pamięci i wąskie gardła C++/CPU (P0/P1)

| # | Komponent | Problem / wąskie gardło | Ryzyko | Rekomendowane rozwiązanie |
|---|---|---|---|---|
| C1 | `fast_kan_kernel.cpp::build_dmrg_normal_equations_batch` | Każdy wątek alokuje `std::vector<double> local_A(P*P)`, gdzie `P = r_prev·K1²·r_next`. Dla domyślnych `max_rank=16, degree=5`: `P = 9216`, `P² = 8.49e7` → **679 MB na wątek**. 16 wątków = 10.9 GB. Redukcja przez `#pragma omp critical` z pętlą po 8.5e7 elementów — serializowana, dla każdego wątku. | Natychmiastowy OOM na typowej stacji przy domyślnej konfiguracji `DMRGTTKANSolver`. Redukcja krytyczna kasuje cały zysk z równoległości. | Nie akumulować `Φ` per-próbka. `Φ` jest wierszowym iloczynem Khatri-Rao — budować ją blokami po `B` wierszy (B ≈ 512) do bufora `B×P` i wołać **`cblas_dsyrk`** (`A ← A + Φ_BᵀΦ_B`) oraz `cblas_dgemv` dla `B`. Pamięć `O(B·P + P²)` **współdzielona**, nie per-wątek; równoległość i wektoryzacja z BLAS-a (≥50 GFLOP/s vs ~2 GFLOP/s pętli ręcznej). Dla P > 4096 w ogóle nie formować `A`: LSQR/CGLS na `[Φ; √α I]` z macierzą-wektor liczoną w locie. |
| C2 | `fast_kan_kernel.cpp` (ten sam) + `fast_kan_bindings.cpp` | `const int P = r_prev*K1*K1*r_next; ... P*P` w arytmetyce `int`. Dla `r=32, degree=7`: `P = 65536`, `P² = 4.29e9` > `INT32_MAX` → **przepełnienie ze znakiem (UB)**, wartość ujemna trafia do `std::memset(A_out, 0, P*P*sizeof(double))` → konwersja do olbrzymiego `size_t`. Walidacja w bindingu (`static_cast<int>(A_out.size()) < P*P`) przepełnia się identycznie i przechodzi. | Uszkodzenie sterty / segfault przy legalnej konfiguracji użytkownika. GIL zwolniony → brak jakiegokolwiek zabezpieczenia po stronie Pythona. | `std::size_t`/`std::int64_t` dla wszystkich iloczynów rozmiarów; jawny `if (P > kMaxP) throw std::invalid_argument`. Kompilacja z `-fsanitize=signed-integer-overflow` w konfiguracji Debug i uruchomienie testów pod ASan/UBSan w CI. |
| C3 | `fast_kan_bindings.cpp::py_evaluate_tt_kan_batch` / `_gradient_batch` | Waliduje `ranks.size() == D+1` i `out.size()`, ale **nigdy `cores_flat.size()` ani `core_offsets.size()`**. Jądro indeksuje `cores_flat + core_offsets[d]` i czyta `r_prev·K1·r_next` doubli. Niespójne `cores`/`ranks` (trywialne do wywołania z Pythona) → odczyt poza buforem. | Odczyt poza zakresem przy zwolnionym GIL. Wektor ataku dla plików modeli z niezaufanych źródeł (`safetensors_io` czyta `ranks` z metadanych pliku). | Walidować w bindingu: `core_offsets.size() == D`, `cores_flat.size() == Σ_d ranks[d]·K1·ranks[d+1]`, `ranks[0] == 1 && ranks[D] == 1` (jądro milcząco tego wymaga: `curr[0]=1.0`, `*out = curr[0]`). |
| C4 | `fast_kan_kernel.cpp::evaluate_tt_kan_single` (pętla gorąca) | Kolejność pętli `for s { for k { m_rs += T[k]*core_r[k*r_next + s] } }` — redukcja po `k` przy **kroku `r_next`** w pamięci. Dla `r_next=16` każdy dostęp to inna linia cache. To wzorzec AoS w kodzie deklarującym SoA. | Przepustowość ograniczona przez cache, nie ALU. Wektoryzacja praktycznie niemożliwa (gather ze stałym krokiem). To jest główne wąskie gardło CPU, nie brak intrinsics. | Zamienić kolejność: `memset(m, 0, r_next); for k { const double t = T[k]; for s { m[s] += t * core_r[k*r_next + s]; } }` — wewnętrzna pętla to ciągły `axpy`, auto-wektoryzowalny i przyjazny prefetcherowi. Analogicznie w `evaluate_tt_kan_gradient_single`. Oczekiwany zysk 3–6× bez jednej linii intrinsics. |
| C5 | `fast_kan_kernel.cpp` (cały) | `#pragma loop(ivdep)` to **składnia wyłącznie MSVC**. Na GCC/Clang jest ignorowana (ostrzeżenie `-Wunknown-pragmas`). Brak `#pragma omp simd`, brak intrinsics, brak dyspozycji runtime. Deklaracja klasy: „AVX2 & OpenMP", „SIMD". W `build/` są wyłącznie artefakty MSVC (`*.vcxproj`, `_cpp_kernels.cp312-win_amd64.pyd`). Brak CI. | Ścieżka GCC/Clang nigdy nie została zbudowana ani zmierzona. Deklaracja SIMD nie jest weryfikowalna na żadnej platformie. | `#pragma omp simd reduction(+:...)` (przenośne, część OpenMP 4.0) lub `google/highway` dla jawnej wektoryzacji z dyspozycją runtime. Włączyć `-Wall -Wextra -Wunknown-pragmas -Werror` — obecne pragmy natychmiast wypadną. CI na trzech toolchainach. |
| C6 | `CMakeLists.txt` | `-mavx2 -mfma` **bezwarunkowo** na x86_64, bez dyspozycji runtime i bez wariantu bazowego. | `SIGILL` przy imporcie modułu na każdym CPU bez AVX2 (m.in. część instancji chmurowych i procesorów Atom/Celeron). Blokada dystrybucji wheela. | Baseline `-mavx2` **tylko** dla wydzielonej jednostki translacji + `__builtin_cpu_supports("avx2")` przy wyborze implementacji, albo `highway` (robi to natywnie). Alternatywa minimalna: baseline SSE2 i AVX2 za flagą `HSKAN_ENABLE_AVX2=OFF` domyślnie wyłączoną w wheelu publicznym. |
| C7 | `CMakeLists.txt` | `-ffast-math` (GCC/Clang) i `/fp:fast` (MSVC) na **jądrze numerycznym**. Zezwala na reasocjację rekurencji Czebyszewa, zakłada brak NaN/Inf (`x != x` może zostać usunięte), łamie semantykę IEEE. | Framework deklarujący „stabilność numeryczną" i „certyfikowane bezpieczeństwo" kompiluje się z wyłączoną semantyką IEEE. Wszystkie strażniki typu `if (total_energy < 1e-14)` i detekcja NaN stają się niewiarygodne. Nie do obrony w audycie zewnętrznym. | Usunąć. Jeśli potrzebna FMA-kontrakcja: `-ffp-contract=fast` (bezpieczne, zwiększa dokładność) zamiast pełnego `-ffast-math`. `/fp:precise` na MSVC. |
| C8 | `fast_kan_kernel.cpp::evaluate_tt_kan_gradient_single` | `std::vector<int> m_offsets(D);` — alokacja sterty **przy każdym wywołaniu**, czyli raz na punkt, wewnątrz pętli OpenMP. Dla N = 50 000 to 100 000 par malloc/free z rywalizacją między wątkami. Dodatkowo ramka stosu ~66 KB (`stack_M[2048] + stack_dM[2048] + stack_L/R`) przy L1d = 32 KB — komentarz „L1 Cache resident" jest nieprawdziwy. | Alokator staje się punktem serializacji. Zestaw roboczy przekracza L1, więc gradient jest wolniejszy od ewaluacji o więcej niż wynika z operacji. | Bufor scratch per-wątek alokowany raz w `#pragma omp parallel` i przekazywany do funkcji punktowej (`struct ScratchBuffers`), albo `alloca`/tablica `int m_offsets[MAX_STACK_DIM]`. Sblokować pętlę po punktach (tile 8–16 punktów) tak, by `M`/`dM` mieściły się w L1. |
| C9 | `cpp_kernels/cpp_kan_engine.py` (wszystkie metody `*_batch`) | Przy **każdym wywołaniu**: `np.concatenate([...ravel() for c in cores])` + `np.ascontiguousarray(X)` + 3 dodatkowe alokacje. Dla D=100, r=16, K1=6 to 1.2 MB kopiowania rdzeni na wywołanie. Docstring: „zerowy narzut pamięciowy (brak alokacji na stercie per zapytanie)". | W pętli 120 FPS: ~150 MB/s czystego narzutu; przy małych `N` narzut pakowania **przewyższa czas liczenia**. Deklaracja w docstringu jest sprzeczna z kodem. | Trwały uchwyt modelu: `TTHandle`/`CPHandle` przechowujący spakowany, ciągły bufor + offsety, przepakowywany wyłącznie przy zmianie wag (`version` counter). API: `engine.bind(model) -> handle`, potem `engine.evaluate(handle, X, out=...)` z buforem wyjściowym dostarczanym przez wywołującego. |
| C10 | `fast_kan_kernel.cpp` (wszystkie `omp parallel for`) | `schedule(static)` przy zmiennym koszcie punktu (gałąź heap vs stack) → niezbalansowanie. Zapisy `Y_out[i]` / `grad_out + i*D` przy `D=3` dzielą linie cache na granicach chunków. | Utrata 10–30% skalowalności; przy `D=3` mierzalny false sharing. | `schedule(static, 64)` (chunk = wielokrotność linii cache dzielona przez `sizeof(double)*D`) lub `guided`. Zmierzyć, nie zgadywać — dodać mikrobenchmark skalowalności do CI. |

### 1.3 Skalowalność i uwarunkowanie (P1)

| # | Komponent | Problem / wąskie gardło | Ryzyko | Rekomendowane rozwiązanie |
|---|---|---|---|---|
| N1 | wszystkie solvery ALS (`closed_form_als`, `tt_kan::TTALSSolver`, `dr_tt_als`, `tucker_als`, `hs_ckan/closed_form_solver`, `pde_poisson_solver`) | Jednolity wzorzec `A = Φᵀ Φ + α·I; np.linalg.solve(A, Φᵀ y)`. Układ normalny **podnosi uwarunkowanie do kwadratu**: `κ(ΦᵀΦ) = κ(Φ)²`. Zmierzone: κ(A)=1.1e6 → κ(AᵀA)=1.3e12, pozostają 4 cyfry (D=3, K=10). `α` jest stałą absolutną (1e-4 … 1e-9), nieskalowaną względem `trace(A)/n` — dla danych o dużej amplitudzie regularyzacja jest nieistotna, dla małej dominuje rozwiązanie. | Utrata precyzji rosnąca wykładniczo z `degree`. Cicha, bo residuum liczone jest tą samą macierzą. Deklaracja „0 epok = dokładne rozwiązanie analityczne" nie ma pokrycia. | Trzy warstwy, w tej kolejności: **(a)** nie formować `ΦᵀΦ` — rozwiązywać LS na rozszerzonej `[Φ; √α·I]` przez `scipy.linalg.lstsq` (QR/DGELSD) → koszt `κ(Φ)`, nie `κ(Φ)²`, odzyskane ~6 cyfr; **(b)** ekwilibracja kolumnowa (skalowanie Jacobiego `Φ ← Φ D⁻¹`, `D = diag(‖Φ_:,j‖)`) — bloki laplasjanu (`T''_K(1) = K²(K²−1)/3`) i brzegowe (O(1)) są obecnie nieskalowalne względem siebie; **(c)** `α` względne: `α_eff = α_rel · trace(ΦᵀΦ)/n`, `α_rel` domyślnie 1e-10. Dodatkowo **zawsze raportować** `cond` i `rank` z SVD w słowniku wynikowym — metryka, która nie widzi własnego uwarunkowania, jest bezwartościowa. |
| N2 | `applications/pde_poisson_solver.py::SpectralKANPoissonSolver` | Baza pełnego iloczynu tensorowego `M = (K+1)^D`. Dla D=4, K=10: M = 14 641 → `AᵀA` zajmuje 1.7 GB. Pętla po `self.multi_indices` w czystym Pythonie: M iteracji × D mnożeń wektorowych. Brak jakiegokolwiek strażnika na D. Punkty kolokacji z `np.random.uniform` (globalny RNG) zamiast węzłów Czebyszewa-Gaussa-Lobatto. | Klątwa wymiarowości w module reklamowanym jako bezsiatkowy. D ≥ 4 nieużywalne. Losowa kolokacja niszczy zbieżność spektralną (rząd `O(N^{-1/2})` zamiast wykładniczego) i czyni wyniki niereprodukowalnymi. | Krótkoterminowo: `Φ` przez jeden `np.einsum('nk,nl,nm->nklm', T0,T1,T2).reshape(N,-1)` (≈100× szybciej), `LapPhi` jako suma D takich kontrakcji; siatka tensorowa CGL zamiast losowej; twardy strażnik `D ≤ 3`. Docelowo: **RFC §2** — QTT + metoda ultrasferyczna, która usuwa zarówno `(K+1)^D`, jak i κ=1e12. |
| N3 | `tdff_net/dmrg_kan.py`, `dr_tt_als.py`, `pde_poisson_solver.py::TTPoissonSolver` | Prefiksy `L`/`R` przeliczane **od zera dla każdego węzła `d`** wewnątrz sweepu: `_update_prefixes` iteruje po wszystkich D wymiarach, a jest wołane D razy → `O(D²)` kontrakcji i `O(D²)` ewaluacji bazy Czebyszewa na sweep. `TTPoissonSolver.fit_als` woła `get_prefixes` dwukrotnie na węzeł. | Dla D=100: 10 000 kontrakcji na sweep zamiast 200. Deklarowana złożoność TT `O(D·R²·K)` nie jest osiągana przez żaden solver w repozytorium. | Kanoniczny wzorzec DMRG: utrzymywać **stos** prefiksów. Przy przesuwaniu okna w prawo dokładać jeden `L`, zdejmować jeden `R` — `O(1)` amortyzowanej pracy na węzeł, `O(D)` na sweep. Bazy Czebyszewa `T_list` liczyć raz na `fit()` (już tak jest w `dmrg_kan`, nie jest w `dr_tt_als`). |
| N4 | `tdff_net/tensor_field.py::TDFFNet.gradient`, `cpp_kan_engine.gradient_cp_batch` (fallback NumPy) | `for dim in range(D): term_evals = phi_evals.copy(); ...; np.prod(term_evals, axis=2)` — kopia tablicy `(N,R,D)` **na każdy wymiar** → `O(N·R·D²)` pamięci i czasu. Jądro C++ robi to poprawnie (prefiks/sufiks, `O(N·R·D)`); NumPy nie. | Fallback NumPy jest `D` razy wolniejszy niż konieczne. Przy D=100, N=10⁴, R=16 to 1.6e9 elementów przetworzonych zamiast 1.6e7. | Przenieść algorytm prefiks/sufiks z `evaluate_cp_kan_gradient_batch` (C++, linie „2./3./4.") do NumPy: `pref = cumprod` wyłączny, `suff` analogicznie odwrócony, `grad[:,m] = (λ · dφ_m · pref_m · suff_m).sum(1)`. |
| N5 | `tdff_net/tensor_field.py` (CP jako całość) | **Zmierzone:** mediana `∏_d φ_d` spada do 6e-42 przy D=100 (float64 wytrzymuje, float32 nie — `FLT_MIN = 1.2e-38`). Rozkład CP jest z natury źle skalowany dla dużych D bez normalizacji per-mod. | CP-KAN w float32 (WebGPU, domyślny JAX) zeruje się przy D ≳ 60, traci połowę mantysy przy D ≈ 25. Deklaracja „D = 3…50" w benchmarku jest poza reżimem poprawności f32. | Normalizacja per-mod z jawnym eksponentem: przechowywać `φ_d = s_d · φ̂_d`, `‖φ̂_d‖ ≈ 1`, oraz akumulator `log`-skali; rekonstrukcja `f = exp(Σ log s_d) · ∏ φ̂_d`. Alternatywnie **nie używać CP dla D > 20** — TT jest tu strukturalnie właściwy (kontrakcja macierzowa nie kumuluje iloczynu skalarów) i już jest w repo. Wyegzekwować to w fasadzie. |
| N6 | `tdff_net/tt_cross.py::TTCrossSolver.fit_function` | (a) Potrójne pętle Pythona budujące `pts` — `O(r²K)` iteracji interpretera na węzeł na sweep. (b) `P_inv = np.linalg.pinv(inter_evals)` — jawne odwracanie macierzy przecięcia, mimo że `maxvol` **już zwraca `Z = A·A[I,:]⁻¹`**, czyli dokładnie macierz interpolacji; `Z` jest odrzucane (`I_piv, _ = maxvol(...)`). (c) Brak deduplikacji indeksów w losowej inicjalizacji `J_sets` → możliwe wiersze zdegenerowane. (d) Ostateczna konstrukcja rdzeni **ponownie ewaluuje** punkty policzone w sweepach — brak memoizacji. (e) `V_inv = np.linalg.inv(V)` zamiast DCT-I. (f) Brak raportowania `κ(A(I,J))` — jedynej wielkości determinującej stabilność TT-Cross. | Liczba wywołań funkcji celu ~2× powyżej optimum (krytyczne, gdy `func` to symulacja CFD). Stabilność niekontrolowana. Przy D=1000 pętle Pythona dominują czas. | (a) Wektoryzacja przez `np.repeat`/`np.tile` + jedno `np.concatenate`. (b) Użyć `Z` z `maxvol` bezpośrednio — eliminuje `pinv` i macierze przecięcia (punkt (d) znika przy okazji). (c) `np.unique(..., axis=0)` + uzupełnienie do rangi. (e) `scipy.fft.dct(type=1)` — `O(K log K)`, lepiej uwarunkowane. (f) Zwracać `TTCrossReport{n_evals, ranks, max_kappa_intersection, residual_on_holdout}`; **residuum na niezależnym zbiorze walidacyjnym** jest jedyną uczciwą metryką dla krzyżowej aproksymacji. |
| N7 | `tdff_net/streaming_als.py::StreamingALSSolver` | Docstring deklaruje RLS z macierzą precyzji `P` i wzorami Shermana-Morrisona. **`self.P_matrices` jest alokowane w `__init__` i nigdy nie używane. `self.forget_factor` nigdy nie używany.** Implementacja to Normalized LMS ze stałym `learning_rate`. Aktualizuje `factors` bez korekty `lambdas` → dryf skali CP. | Deklarowana zbieżność `O(1)` per próbka i śledzenie kowariancji nie istnieją. Martwa pamięć `D·(K1·R)²·8` bajtów. Nazwa klasy wprowadza w błąd na poziomie API. | Decyzja: **albo** zaimplementować RLS (`k = Pφ/(λ+φᵀPφ)`, `P ← (P − kφᵀP)/λ`) i wtedy nazwa jest uczciwa, **albo** przemianować na `StreamingNLMSSolver`, usunąć `P_matrices`/`forget_factor` i udokumentować rzeczywistą charakterystykę zbieżności. RLS jest tu wartościowy (dostarcza kowariancję → przedziały ufności dla CBF), więc rekomendacja: zaimplementować. |
| N8 | `jax_kan/autograd_ops.py`, `layers.py` | (a) Pętle `for d in range(D)` i `for k in range(1, degree)` w kodzie śledzonym przez JAX → **pełne rozwinięcie grafu**: `O(D·K)` węzłów HLO. Dla D=100, K=10 to ~1000 węzłów tylko na bazę. (b) `jax.config.update("jax_enable_x64", True)` **nie występuje nigdzie w repo**, a `layers.py` domyślnie ustawia `dtype=jnp.float64` → JAX cicho degraduje do float32. | (a) Czas kompilacji XLA rośnie liniowo z D·K; przy D≥50 minuty na `jit`. (b) Backend JAX liczy w **float32**, deklarując float64. Brak testu porównującego torch-f64 z jax; rozbieżność ~1e-7 pozostaje niewykryta. | (a) `jax.lax.scan` po wymiarach i po stopniu, ze spakowaną tablicą `factors: (D,R,K+1)` — graf `O(1)` względem D. (b) Jawny `jax.config.update("jax_enable_x64", True)` w `jax_kan/__init__.py` **z ostrzeżeniem**, jeśli nie zadziała, albo domyślny `dtype=jnp.float32` i uczciwa dokumentacja. Test parzystości `torch(f64) ↔ jax(x64)` przy `atol=1e-12` jako bramka CI. |

### 1.4 Nieszczelne abstrakcje i kontrakt API (P1/P2)

| # | Komponent | Problem | Ryzyko | Rozwiązanie |
|---|---|---|---|---|
| A1 | `facade.py::TensorField` | Docstring: „High-throughput C++ SIMD inference (< 0.3 µs per point) (`.predict()`)" — `predict()` woła `self._model.evaluate()`, czyli **czysty NumPy**; `FastCPPKANEngine` nie jest importowany w fasadzie. Docstring: „Zero-copy conversion to PyTorch (`.to_torch()`)" — implementacja robi `torch.from_numpy(...).to(dtype, device)` **plus** `copy_()`, czyli dwie kopie. | Główny publiczny interfejs deklaruje wydajność, której nie dostarcza. Użytkownik mierzy 30× gorszy wynik niż obiecany i traci zaufanie do wszystkich pozostałych liczb. | Albo podłączyć `FastCPPKANEngine` z trwałym uchwytem (C9) i utrzymać deklarację, albo usunąć deklarację. Trzecia opcja, najlepsza: `predict(..., backend: Literal["auto","numpy","cpp"] = "auto")` z `TensorField.backend_info()` raportującym, co faktycznie działa. |
| A2 | `facade.py::TensorField.save` vs `tdff_net/serializer.py::KANSerializer` | Dwie niezależne implementacje serializacji JSON z **różnymi tagami typu**: `"TDFFNet_CP_KAN"` (fasada) vs `"TDFFNet_CP"` (serializer). `KANSerializer.load_json` rzuca `ValueError` na plikach zapisanych przez fasadę. Trzeci format: `safetensors_io` (`"TDFFNet_CP"`). Brak pola wersji formatu w którymkolwiek. | Pliki zapisane jednym API nie wczytują się drugim. Brak wersjonowania → każda zmiana schematu cicho psuje stare artefakty. | Jedno źródło prawdy: `hs_kan.io` z `save(model, path, format="safetensors"|"json")` i `load(path)`. Obowiązkowe `{"format_version": 1, "model_type": ...}`. JSON wyłącznie do inspekcji (25 bajtów na float — dla TT z D=100 to ~4 MB tekstu); safetensors jako format domyślny. |
| A3 | `facade.py::PoissonSolver.solve` | Dopasowanie sygnatury funkcji użytkownika przez `try: source_fn(X[:,0], X[:,1]) except TypeError: source_fn(X)`. `TypeError` **wyrzucony wewnątrz** funkcji użytkownika zostaje przechwycony i funkcja jest wołana **drugi raz** z inną sygnaturą. | Maskowanie prawdziwych błędów użytkownika jako niezgodności sygnatury; podwójne wykonanie efektów ubocznych; komunikaty błędów wskazujące złe miejsce. | `inspect.signature(fn).parameters` — decyzja przed wywołaniem, jednokrotnie. Alternatywnie: wymagać jednej konwencji (`f(X: np.ndarray) -> np.ndarray`) i udokumentować ją. |
| A4 | `tt_kan.py`, `dr_tt_kan.py`, `pde_poisson_solver.py::TTPoissonSolver`, 16 plików w `src/tasks/` | `np.random.seed(42)` **w `__init__`** modelu. Konstrukcja modelu resetuje **globalny stan RNG całego procesu**. | Utworzenie `TensorTrainKAN` w środku eksperymentu niszczy reprodukowalność wszystkiego, co losuje później — w tym `InterAgentCBF` (`np.random.randn` przy zdegenerowanym gradiencie) i próbkowania kolokacji w PDE. Determinizm, będący centralną deklaracją projektu, jest łamany przez konstruktor. | `rng: np.random.Generator | int | None = None` jako parametr; `self._rng = np.random.default_rng(seed)`. **Zero** `np.random.seed` w bibliotece — dopuszczalne wyłącznie w skryptach `src/tasks/` i testach. Reguła do wymuszenia w lincie (`ruff` NPY002). |
| A5 | `facade.py` (`predict`, `gradient`), `pde_poisson_solver` (`evaluate`, `laplacian`), `robotics_cbf_planner` (`evaluate_h`, `gradient_h`) | Heurystyka `if X.ndim == 1 and len(X) == spatial_dim: return scalar else: return array` — **typ zwracany zależy od wartości wejścia**. Dla `D=1` przypadek jest niejednoznaczny. | Kod wywołujący musi rozgałęziać lub `np.atleast_1d` wszędzie. Źródło błędów kształtów w warstwie CBF, gdzie każde `evaluate_h` opakowane jest w `float(np.asarray(...).ravel()[0])` — obejście widoczne w 8 miejscach. | Zawsze `(N,)` / `(N,D)`. Osobne, jawne `predict_one(x: (D,)) -> float`. Usunąć 8 obejść. |
| A6 | `facade.py` | `try: import torch ... except ImportError` — uszkodzona instalacja torcha (np. brak `libcudart`) rzuca `OSError`, nie `ImportError`, i **wywala import całej fasady**. `_HAS_TORCH=False` nie zapisuje przyczyny. | Nieodporna izolacja opcjonalnych zależności. Diagnoza „PyTorch is required" myli, gdy torch jest zainstalowany, ale zepsuty. | `except Exception as e: _TORCH_IMPORT_ERROR = e`. W `to_torch()`: `raise ImportError(...) from _TORCH_IMPORT_ERROR`. |
| A7 | `tt_cross.py`, `dmrg_kan.py` | `from src.cpp_kernels.cpp_kan_engine import FastCPPKANEngine` **wewnątrz funkcji**, w czysto matematycznej ścieżce. Kierunek zależności odwrócony: warstwa algorytmiczna zależy od warstwy akceleracji. | Warstwa matematyczna niezdatna do testowania w izolacji. Import w gorącej ścieżce (cache modułów łagodzi, ale ukrywa zależność przed narzędziami). | Wstrzykiwanie zależności: `TTCrossSolver(..., engine: KernelBackend = NumpyBackend())`. Protokół `KernelBackend` z implementacjami `NumpyBackend` / `CppBackend` / `TritonBackend`. To jest też warunek wstępny Wektora 2. |
| A8 | `torch_kan/layers.py::from_tt_kan` | Przy zmianie kształtu: `self.cores[d] = nn.Parameter(...)` — podmiana obiektu w `ParameterList`. Optymalizator utworzony wcześniej trzyma referencje do **starych** tensorów. | Po `fit_tt_cross()` (który zmienia rangi) istniejący `optimizer` cicho przestaje aktualizować model. Workflow „dopasuj analitycznie → dostrój gradientowo" jest zepsuty dokładnie w miejscu, które jest sprzedażowym argumentem projektu. | Ostrzeżenie + wymuszenie: metoda zwraca `bool` (`shapes_changed`), a dokumentacja wymaga odtworzenia optymalizatora. Lepiej: `TensorTrainKANLayer.from_tt_kan` jako `classmethod` tworząca nową warstwę, nigdy mutująca istniejącą. |
| A9 | `torch_kan/safetensors_io.py::load_kan_safetensors` (`as_torch=False`) | `f.get_tensor(...)` wewnątrz `with safetensors.safe_open(...)`, następnie `.cpu().numpy()` i zwrot **po wyjściu z kontekstu**. Tensory z `safe_open` mogą referować mapowanie pliku. | Potencjalne use-after-unmap: nieokreślone dane lub segfault. Do potwierdzenia dla używanej wersji safetensors — ale wzorzec jest niebezpieczny niezależnie od wersji. | `.clone()` (lub `np.array(..., copy=True)`) przed opuszczeniem bloku `with`. Test: wczytać, wymusić GC, zweryfikować sumę kontrolną. |
| A10 | `hs_ckan/closed_form_solver.py` + `nary_spatiotemporal.py` | `A = Φᵀ Φ + αI` o rozmiarze `K×K`, gdzie `K = num_pred + num_ent + num_ent · st_dim`. Dla `num_ent=1000, D=4, kan_degree=4`: `st_dim = 170`, `K ≈ 1.7e5` → `A` zajmuje **231 GB**. Liczba faktów `N` jest typowo o rzędy mniejsza od `K` (układ silnie niedookreślony). | Moduł pamięci symbolicznej nie skaluje się poza kilkadziesiąt encji. Problem jest strukturalny (iloczyn Kroneckera `zone ⊗ st_vector`), nie parametryczny. | Postać dualna (kernelowa): dla `N < K` rozwiązanie ridge to `W = Φᵀ(ΦΦᵀ + αI)⁻¹Y` — koszt `O(N²K + N³)` zamiast `O(NK² + K³)`, pamięć `O(N² + NK)`. Przy N=1000, K=1.7e5: 1.4 GB zamiast 231 GB. Dodatkowo `Φ` jest rzadka i strukturalna (Kronecker) — `scipy.sparse.linalg.LinearOperator` z jawnym `matvec` eliminuje jej materializację. |
| A11 | `hs_ckan/nary_spatiotemporal.py::CleanUpMemory.cleanup` | Twardy `argmax` bez marginesu ufności i bez ścieżki abstynencji. Stan dwuznaczny (dwa niemal równe predykaty) daje pewną, arbitralną odpowiedź. | Deklaracja „0% halucynacji" (Wektor 4) nie ma pokrycia: moduł zawsze odpowiada, nigdy nie odmawia. | Zwracać `CleanupResult{vector, margin, confident: bool}`, gdzie `margin = (s₁ − s₂)/s₁`. Poniżej progu — `confident=False` i jawne `ABSTAIN` w górę stosu. To jest warunek konieczny dla Wektora 4 (koprocesor LLM musi umieć powiedzieć „nie wiem"). |
| A12 | `mct_nse/category_filter.py`, `concurrent_category_filter.py` | `filter_state(..., max_iters=5)`: pętla punktu stałego wychodzi po 5 iteracjach **bez sygnalizacji**, że punkt stały nie został osiągnięty. Zwracane `violations` to lista nazw, nie dowód zbieżności. | Klasa deklaruje „w 100% algebraiczne spełnienie wszystkich inwariantów (0% violation rate)". Przy morfizmach naprawczych, które ze sobą kolidują (typowe: „wewnątrz pudełka" vs „poza przeszkodą"), pętla nie zbiega, a wynik jest zwracany jako bezpieczny. | `filter_state -> FilterResult{state, converged: bool, iters, violations}`. `converged=False` **musi** propagować się do wywołującego (w CBF: przejście w tryb awaryjny). Dodatkowo: test własności — dla losowych stanów i zbioru inwariantów sprawdzić, że po `filter_state` **wszystkie** predykaty są spełnione; obecnie taki test nie istnieje. |

### 1.5 Weryfikacja: testy i benchmarki (P0 dla wiarygodności)

| # | Komponent | Problem | Ryzyko | Rozwiązanie |
|---|---|---|---|---|
| V1 | `benchmarks/benchmark_vs_pykan.py` | Klasa `SimulatedBSplineKANLayer` deklaruje „Simplified Cox-de Boor spline basis", a implementuje `np.exp(-0.5*((X-center)/0.2)**2)` — **funkcje Gaussa, nie B-splajny**. Ewaluacja w czystym Pythonie/NumPy. Biblioteka `pykan` nie jest importowana. | Benchmark „vs PyKAN" porównuje się z własnoręcznie napisaną, nieoptymalną atrapą o innej bazie funkcyjnej. **Każda liczba przyspieszenia z tego pliku jest bezwartościowa** i przy pierwszym zewnętrznym przeglądzie zdyskredytuje pozostałe, uczciwe wyniki. | Zainstalować `pykan`, `efficient-kan`, `fastkan` jako zależności benchmarkowe i mierzyć rzeczywisty kod. Jeśli to niemożliwe — usunąć plik i nazwę „PyKAN". Raportować wersje bibliotek, model CPU, flagi kompilacji, `n_repeat`, medianę i IQR (nie średnią). |
| V2 | `tests/test_cpp_kernels.py` | `assert throughput >= 1_500_000` i `assert latency_us <= 0.67` — asercje wydajnościowe zależne od sprzętu w teście jednostkowym. `assert engine.is_native_available()` bez `pytest.skipif` — pakiet nie przechodzi testów bez zbudowanego rozszerzenia C++. | Flaki na innym sprzęcie i w CI; test suite nieuruchamialny na czystym klonie. | Benchmarki wydzielić do `pytest-benchmark` z porównaniem do zapisanego baseline'u (regresja > 20% = fail), nie do progu absolutnego. `@pytest.mark.skipif(not _HAS_CPP)` na testach natywnych + osobny job CI, który **wymaga** obecności rozszerzenia. |
| V3 | `tests/` (całość, 134 asercje) | Testy porównują implementację C++ z fallbackiem NumPy — **oba napisane przez tego samego autora z tego samego wzoru**. To test spójności, nie poprawności. Jedyny niezależny oracle to różnice skończone w `test_cpp_kernels.py:79` (tol 1e-5). Brak: porównania z `scipy.special.eval_chebyt`, testów własnościowych (`hypothesis`), testów na znanych rozwiązaniach analitycznych z progiem błędu, testów niezmienników bezpieczeństwa CBF. | Klasa błędów „wzór jest zły, ale zaimplementowany spójnie w obu backendach" jest niewykrywalna. Dokładnie taka jest natura M1, M2, M3 — **żaden z tych trzech błędów nie jest łapany przez 134 istniejące asercje**. | Trzy warstwy: **(a)** oracles zewnętrzne — `scipy.special.eval_chebyt/eval_chebyu` dla bazy, `numpy.polynomial.chebyshev.chebder` dla pochodnych, rozwiązania analityczne z `PoissonAnalyticalSolution` z **twardym progiem** `L2_rel < 1e-8`; **(b)** testy własnościowe (`hypothesis`) — niezmienniki: `∫T_iT_j w = 0` dla i≠j, `‖TT_evaluate − dense_contract‖ < ε` dla małych D, `filter_state` osiąga punkt stały, `∀t: h(p_t) ≥ −ε` na całej trajektorii CBF; **(c)** testy parzystości backendów: numpy ↔ cpp ↔ torch(f64) ↔ jax(x64) ↔ WGSL(f32, tol 1e-5) na jednym zestawie wag. |
| V4 | repozytorium | **Brak `.github/`** — zero CI. `build/` zawiera wyłącznie artefakty MSVC (`*.vcxproj`, `_cpp_kernels.cp312-win_amd64.pyd`). Ścieżka GCC/Clang nigdy nie zbudowana. Brak lintera, brak type-checkingu w bramce (jest `pyrightconfig.json`, ale nic go nie egzekwuje). | Wszystkie deklaracje przenośności i wydajności są nieweryfikowalne. C5, C6, C7 istnieją, bo nikt nigdy nie skompilował tego kodu poza MSVC. | Matryca CI: `{ubuntu, macos, windows} × {gcc-13, clang-17, msvc} × {py3.11, py3.12}`. Bramki: `ruff` + `pyright --strict` na `src/`, `-Wall -Wextra -Werror` na C++, testy pod ASan/UBSan (job osobny, Debug), `pytest-benchmark` z baseline'em. To jest **warunek wstępny dla wszystkich czterech wektorów** — bez tego każda optymalizacja jest niesprawdzalna. |
| V5 | `web_showcase/src/shaders/kanComputeShaders.ts` | Rozmiary tablic wpisane na sztywno w WGSL: `array<f32, 6>` (degree = 5), `array<vec4<f32>, 12>` (48 floatów = rank 8 × K1 6). Strona Pythona pozwala na dowolne `rank`/`degree`. `KANSerializer.to_webgpu_buffers` deklaruje wyrównanie do 16 bajtów, ale zwraca płaską tablicę bez paddingu — i jako listę Pythona serializowaną do JSON. | Model z `rank=16` lub `degree=7` cicho generuje błędne wyniki albo przepełnia bufor. Transport wag jako tekst JSON zamiast binarnego `ArrayBuffer` — kilkukrotny narzut rozmiaru i czasu parsowania. Cały tor przeglądarkowy w f32, podczas gdy „certyfikaty" ustalono w f64. | WGSL generowany z szablonu parametryzowanego `(R, K1)`; nagłówek bufora z `{magic, version, rank, degree, dim}` i walidacją po stronie JS przed utworzeniem pipeline'u. Eksport binarny (`.bin` + osobny JSON z metadanymi). Test parzystości f32: WGSL vs NumPy przy `atol=1e-5` (V3c). |

---

## 2. Specyfikacja architektoniczna (Design RFC) — Wektor 1: silnik QTT

### RFC-001: Quantized Tensor Train na indeksach ultrasferycznych

**Status:** Draft · **Zastępuje:** `SpectralKANPoissonSolver`, `TTPoissonSolver` · **Zależy od:** V4 (CI), N1 (rezygnacja z układów normalnych), A7 (wstrzykiwanie backendu)

#### 2.1 Uzasadnienie problemowe

Audyt pokazuje dwie niezależne ściany, które QTT usuwa **jedną konstrukcją**:

1. **Ściana uwarunkowania.** κ(AᵀA) = 1.3e12 przy D=3, K=10 (zmierzone). Wynika z tego, że macierz różniczkowania Czebyszewa w bazie nodalnej ma normę rosnącą jak `K⁴` (`T''_K(1) = K²(K²−1)/3`). Zwiększanie `degree` — jedyny sposób na ostre fronty i wysokie częstotliwości — pogarsza uwarunkowanie szybciej, niż poprawia aproksymację.
2. **Ściana pamięci.** Baza pełnego iloczynu `(K+1)^D`, oraz — nawet w TT — liniowy koszt `O(K)` na mod. Dla pól 3D/4D o rozdzielczości wymagającej `K ~ 10³` reprezentacja `O(D·K·R²)` jest wciąż nieakceptowalna.

QTT atakuje obie: **kwantyzacja indeksu współczynnika** daje `O(log K)` zamiast `O(K)`, a **przejście do bazy ultrasferycznej** zamienia gęsty, źle uwarunkowany operator różniczkowania na pasmowy i dobrze uwarunkowany, który w reprezentacji QTT ma **dokładną, stałą rangę**.

#### 2.2 Reprezentacja

Niech `K + 1 = 2^L`. Indeks współczynnika `k ∈ {0,…,2^L−1}` rozkładamy binarnie:

```
k = Σ_{l=0}^{L-1} b_l · 2^l ,   b_l ∈ {0,1}
```

Wektor współczynników `c ∈ R^{2^L}` staje się tensorem `C[b_0,…,b_{L−1}]` o `L` modach binarnych i jest reprezentowany łańcuchem TT:

```
C[b_0,…,b_{L−1}] = Q^(0)[:, b_0, :] Q^(1)[:, b_1, :] ⋯ Q^(L−1)[:, b_{L−1}, :]
Q^(l) ∈ R^{ρ_{l−1} × 2 × ρ_l} ,  ρ_0 = ρ_L = 1
```

Pole D-wymiarowe to konkatenacja: `D · L` rdzeni binarnych w jednym łańcuchu (uporządkowanie **przeplatane bitowo** — patrz §2.7).

| Wielkość | Obecnie (TT nodalne) | QTT |
|---|---|---|
| Pamięć na wymiar | `O(K · R²)` | `O(log₂K · ρ²)` |
| Pamięć pola D-wym. | `O(D · K · R²)` | `O(D · log₂K · ρ²)` |
| Ewaluacja w punkcie | `O(D · K · R²)` | `O(D · log₂K · ρ²)` |
| Baza `T_k(x)`, k=0..K | `O(K)` | **ranga dokładnie 2** (§2.3) |
| Różniczkowanie | gęste, `‖D‖ ~ K²` | **MPO rangi ≤ 4**, pasmowe (§2.4) |
| Laplasjan D-wym. | `(K+1)^D` bazy | **MPO rangi ≤ D+1** wzdłuż wymiarów |

Dla `K = 2^20 ≈ 10⁶` i `ρ = 8`: 20·64 = 1280 liczb na wymiar zamiast 10⁶. Współczynnik kompresji ~800×, przy zachowaniu dokładnej reprezentacji funkcji gładkich i quasi-optymalnej dla frontów.

#### 2.3 Ewaluacja: baza Czebyszewa ma dokładną rangę QTT 2

To jest własność, która czyni całą konstrukcję praktyczną. Podstawiamy `x = cos θ`, wtedy `T_k(x) = cos(kθ)`. Ponieważ

```
exp(i k θ) = Π_{l=0}^{L-1} exp(i · b_l · 2^l · θ)
```

para `(cos(kθ), sin(kθ))` propaguje się przez łańcuch binarny **macierzą obrotu**:

```
G^(l)[b] = [[ cos(b·2^l·θ), −sin(b·2^l·θ) ],
            [ sin(b·2^l·θ),  cos(b·2^l·θ) ]]      ∈ R^{2×2}
```

czyli wektor `[T_0(x),…,T_{2^L−1}(x)]` ma **dokładną (nie przybliżoną) rangę QTT równą 2**, niezależnie od `L`. Ewaluacja pola w punkcie to kontrakcja łańcucha rangi 2 z łańcuchem rangi ρ:

```
koszt = O(L · ρ² · 2²)  flopów na wymiar,  bez jednej operacji trygonometrycznej poza początkowym arccos
```

Dla `K = 10⁶`, `ρ = 8`: ~2560 flopów zamiast 10⁶ operacji Clenshawa. **Rdzenie `G^(l)` zależą wyłącznie od `θ`**, więc dla wsadu `N` punktów są to `N·L` macierzy 2×2 — idealny wzorzec dla GEMM wsadowego (i dla Wektora 2: jeden kernel Triton).

Stabilność: `arccos` traci precyzję przy `|x| → 1` (`dθ/dx = −1/√(1−x²)`). Mitygacja: dla `|x| > 1 − 2⁻²⁶` przejść na rozwinięcie `θ ≈ √(2(1∓x))`. Do udokumentowania jako znany limit i pokrycia testem.

#### 2.4 Operatory różniczkowe: baza ultrasferyczna

W bazie nodalnej/modalnej Czebyszewa operator różniczkowania jest **górnotrójkątny i gęsty** (`D[j,k] = 2k/c_j` dla `k>j`, `k−j` nieparzyste) — ranga QTT rośnie z `L`, konstrukcja nie działa. Rozwiązaniem jest metoda ultrasferyczna (Olver–Townsend): różniczkowanie mapuje bazę `C^(λ)` na `C^(λ+1)` i jest **operatorem diagonalno-przesuniętym**:

```
D_λ : C^(λ) → C^(λ+1) ,   (D_λ)_{k, k+1} = 2^{λ−1} (λ−1)! · k        (λ ≥ 1)
D_0 : C^(0) → C^(1) ,     (D_0)_{k, k+1} = k
S_λ : C^(λ) → C^(λ+1) ,   pasmowy, szerokość pasma 2
```

W reprezentacji QTT (indeks binarny):

| Operator | Konstrukcja | Ranga QTT |
|---|---|---|
| `diag(k)` | `k = Σ_l b_l 2^l` — suma L członów rangi 1 → standardowy MPO licznikowy | **dokładnie 2** |
| `shift(±1)` | MPO przeniesienia bitowego (carry) | **dokładnie 2** |
| `D_λ = shift ∘ diag(k)` | złożenie | **≤ 4** |
| `S_λ` (pasmowy, bw=2) | suma diag + shift | **≤ 4** |
| `D_λ D_{λ−1}` (2. pochodna) | złożenie | **≤ 8** |
| `∇² = Σ_d I⊗…⊗L_d⊗…⊗I` | standardowy MPO sumy operatorów lokalnych | **≤ 2** wzdłuż granic wymiarów, ≤ 8 wewnątrz |

**Konsekwencja dla N1:** operator ultrasferyczny jest pasmowy, a po standardowym prekondycjonowaniu diagonalnym ma `κ = O(1)` **niezależnie od K** — w przeciwieństwie do `κ = O(K⁴)` bazy nodalnej. Zmierzone κ(AᵀA)=1.3e12 spada do rzędu 10¹–10². Rozwiązywanie odbywa się przez DMRG/AMEn **bezpośrednio na układzie `Au = f` w formacie MPO/TT**, bez formowania `AᵀA` — problem podniesienia uwarunkowania do kwadratu przestaje istnieć konstrukcyjnie, nie przez łatkę.

#### 2.5 Splot

Dwa reżimy, oba w `O(polylog N)`:

**(a) Splot cykliczny na siatce jednorodnej** — przez QTT-FFT (Dolgov–Khoromskij–Savostyanov). DFT rozmiaru `2^L` rozkłada się na `L` warstw motylkowych; koszt `O(L² ρ³)` zamiast `O(N log N)`. Splot: `FFT → mnożenie Hadamarda w QTT (ranga ρ_a·ρ_b, po zaokrągleniu ρ) → IFFT`.

**(b) Splot w przestrzeni współczynników Czebyszewa** — iloczyn dwóch funkcji odpowiada operatorowi `½(Toeplitz(a) + Hankel(a))` na współczynnikach `b`. Macierze Toeplitza i Hankela mają w QTT rangę `O(log(1/ε))` przy standardowej konstrukcji przez rozkład symbolu. Koszt zastosowania: `O(L · ρ³ · log(1/ε))`.

Interfejs jednolity:

```python
def qtt_convolve(a: QTTField, b: QTTField, *, mode: Literal["cyclic","chebyshev"],
                 eps: float = 1e-10, max_rank: int = 64) -> QTTField: ...
```

#### 2.6 Struktury danych i API

```python
# src/qtt/core.py

@dataclass(frozen=True, slots=True)
class QTTCore:
    """Rdzeń binarny (ρ_prev, 2, ρ_next). Niemutowalny — mutacja przez rebuild."""
    data: np.ndarray                      # C-contiguous, float64, shape (ρp, 2, ρn)
    level: int                            # l ∈ [0, L)
    dim: int                              # d ∈ [0, D)

class QTTField:
    """
    Pole f: [-1,1]^D -> R jako QTT nad indeksami współczynników ultrasferycznych C^(λ).

    Niezmienniki (sprawdzane w __post_init__ i po każdej operacji):
      I1: len(cores) == D * L
      I2: cores[0].shape[0] == 1 and cores[-1].shape[2] == 1
      I3: cores[i].shape[2] == cores[i+1].shape[0]
      I4: wszystkie rdzenie C-contiguous, float64
      I5: canonical_center ∈ [0, D*L) lub None
    """
    __slots__ = ("cores", "spatial_dim", "levels", "lam", "domain", "_canon")

    spatial_dim: int                      # D
    levels: int                           # L, gdzie 2^L = K+1
    lam: int                              # λ — indeks bazy ultrasferycznej (0 = Czebyszew T)
    domain: DomainWindow                  # afiniczne odwzorowanie [a,b]^D -> [-1,1]^D (istnieje: sliding_domain.py)

    # --- konstrukcja ---
    @classmethod
    def from_function(cls, fn: Callable[[np.ndarray], np.ndarray], *, spatial_dim: int,
                      levels: int, eps: float = 1e-10, max_rank: int = 64,
                      sampler: Literal["tt_cross","dmrg_cross","amen_cross"] = "amen_cross",
                      rng: np.random.Generator) -> "QTTField": ...
                      # O(D·L·ρ²) wywołań fn — logarytmicznie względem K

    @classmethod
    def from_tt(cls, tt: TensorTrainKAN, *, eps: float = 1e-10) -> "QTTField": ...
                      # ścieżka migracji z istniejących modeli

    def to_tt(self) -> TensorTrainKAN: ...        # de-kwantyzacja, O(D·K·R²) pamięci — tylko dla małych K

    # --- ewaluacja ---
    def evaluate(self, X: np.ndarray) -> np.ndarray: ...        # (N,D) -> (N,);  O(N·D·L·ρ²)
    def gradient(self, X: np.ndarray) -> np.ndarray: ...        # (N,D) -> (N,D); O(N·D·L·ρ²)
    def hessian(self, X: np.ndarray) -> np.ndarray: ...         # (N,D) -> (N,D,D); wymagane przez M1
    def laplacian(self, X: np.ndarray) -> np.ndarray: ...       # (N,D) -> (N,)

    # --- algebra ---
    def derivative(self, axis: int, order: int = 1) -> "QTTField": ...
                      # zastosowanie MPO rangi ≤4^order; λ rośnie o `order`
    def __add__(self, other: "QTTField") -> "QTTField": ...     # ranga ρa+ρb, potem round()
    def __mul__(self, other: "QTTField") -> "QTTField": ...     # Hadamard, ranga ρa·ρb, potem round()
    def round(self, eps: float = 1e-10, max_rank: int | None = None) -> "QTTField": ...
                      # TT-SVD w formie kanonicznej; O(D·L·ρ³)

    # --- diagnostyka (obowiązkowa, patrz N1/N6) ---
    def report(self) -> QTTReport: ...
        # ranks: list[int], max_rank, memory_bytes, canonical_center,
        # frobenius_norm, truncation_error_estimate

class QTTOperator:
    """MPO nad tymi samymi indeksami binarnymi."""
    @staticmethod
    def diag_k(levels: int) -> "QTTOperator": ...          # ranga 2, dokładny
    @staticmethod
    def shift(levels: int, by: int = 1) -> "QTTOperator": ...  # ranga 2, dokładny
    @staticmethod
    def derivative(levels: int, lam: int) -> "QTTOperator": ...  # D_λ, ranga ≤4
    @staticmethod
    def conversion(levels: int, lam: int) -> "QTTOperator": ...  # S_λ, ranga ≤4
    @staticmethod
    def laplacian(spatial_dim: int, levels: int) -> "QTTOperator": ...  # ranga ≤ D+1

    def __matmul__(self, f: QTTField) -> QTTField: ...     # O(D·L·ρ²·χ²)

# src/qtt/solve.py
def amen_solve(A: QTTOperator, f: QTTField, *, eps: float = 1e-10,
               max_rank: int = 64, max_sweeps: int = 20,
               preconditioner: Literal["none","jacobi","block"] = "jacobi"
               ) -> tuple[QTTField, AMEnReport]: ...
    # AMEn (Dolgov–Savostyanov): DMRG z wzbogaceniem residuum.
    # Koszt/sweep: O(D·L·ρ³·χ + D·L·ρ²·χ²);  NIE formuje A^T A.
    # AMEnReport{sweeps, residual_history, ranks_history, converged: bool}
```

#### 2.7 Uporządkowanie modów

Kolejność `D·L` rdzeni determinuje rangi. Dwa warianty, oba wspierane, wybór przez parametr:

- **`"serial"`** (`d₀b₀…d₀b_{L−1}, d₁b₀…`): niskie rangi dla funkcji **separowalnych** (`f = ∏ f_d`); rangi rosną dla silnie sprzężonych.
- **`"interleaved"`** (`b₀^{(0)}b₀^{(1)}…b₀^{(D−1)}, b₁^{(0)}…`): grupuje bity o tej samej skali; niskie rangi dla funkcji **izotropowych i samopodobnych** — fronty, wiry, pola SDF. To jest wariant właściwy dla zastosowań w `applications/`.

Decyzja projektowa: `mode_order: Literal["serial","interleaved"] = "interleaved"` jako domyślna, z benchmarkiem obu na zestawie referencyjnym (§4, KPI-1.3).

#### 2.8 Analiza złożoności (podsumowanie)

Oznaczenia: `D` wymiar, `K+1 = 2^L` liczba współczynników na wymiar, `ρ` ranga QTT, `χ` ranga MPO, `N` liczba punktów, `n` liczba próbek.

| Operacja | Obecnie | QTT | Warunek przewagi |
|---|---|---|---|
| Pamięć pola | `O(D K R²)` | `O(D L ρ²)` | `ρ² L < R² K` → praktycznie `K ≳ 64` |
| `evaluate` (N pkt) | `O(N D K R²)` | `O(N D L ρ²)` | jw. |
| `gradient` (N pkt) | `O(N D K R²)` | `O(N D L ρ²)` | jw. |
| `hessian` (N pkt) | brak | `O(N D² L ρ²)` | — |
| Dopasowanie do black-box | TT-Cross `O(D R² K)` wywołań | AMEn-Cross `O(D L ρ²)` wywołań | `K/L` ≈ 5·10⁴ przy K=10⁶ |
| Rozwiązanie `∇²u = f` | `O(M³)`, `M=(K+1)^D`, κ=1e12 | `O(sweeps · D L ρ³ χ)`, κ=O(1) | każde D ≥ 2 |
| Splot | `O(K^D log K)` | `O(L² ρ³)` | każde D ≥ 2 |

**Uczciwe ograniczenie:** przy obecnych `degree = 5` (L = 3) QTT to czysty narzut — 3 rdzenie binarne rangi ρ vs 6 współczynników. **Próg opłacalności to `K ≳ 64` (L ≥ 6).** QTT nie jest optymalizacją istniejących ścieżek; jest warunkiem wejścia w reżim, w którym obecny kod strukturalnie nie działa: ostre fronty SDF, pola objętościowe 3D/4D, sygnały o szerokim paśmie. Ten warunek trzeba zapisać w dokumentacji i wyegzekwować w fasadzie (`QTTField.from_function` ostrzega dla `levels < 6`).

#### 2.9 Punkty ryzyka

| Ryzyko | Prawdopodobieństwo | Mitygacja |
|---|---|---|
| Ranga QTT eksploduje dla funkcji nieseparowalnych i niesamopodobnych (np. szum) | wysokie dla danych, niskie dla pól fizycznych | Twardy `max_rank` + `AMEnReport.converged=False`. Bramka akceptacyjna KPI-1.2 mierzy to na zestawie referencyjnym. Jawnie udokumentować klasy funkcji poza zakresem. |
| Utrata precyzji `arccos` przy `\|x\| → 1` | średnie | Rozwinięcie asymptotyczne (§2.3) + test na `x = 1 − 2⁻⁵²`. |
| AMEn nie zbiega dla źle uwarunkowanych operatorów | średnie | Prekondycjoner Jacobiego jako domyślny; fallback na blokowy; `converged: bool` propagowany do wywołującego (nigdy cicha porażka — patrz A12). |
| `L` musi być potęgą 2 | pewne | Dopełnienie zerami do `2^L` + jawny `effective_degree` w metadanych. |
| Złożoność implementacyjna (MPO, formy kanoniczne, AMEn) | wysokie | Nie pisać od zera: `ttpy` / `torchtt` jako referencja poprawności w testach (oracle zewnętrzny, V3a), własna implementacja optymalizowana dopiero po przejściu testów parzystości. |

---

## 3. Wektory 2–4: specyfikacje skrócone

### Wektor 2 — Kernele sprzętowe (Triton/CUDA + rdzeń C++20)

**Warunek wstępny:** C1–C10 naprawione, V4 (CI) działa. Pisanie kerneli Triton przed naprawą kolejności pętli (C4) i przepełnienia `int` (C2) to optymalizacja błędnego kodu.

**2a. Fused Tensor-Train Basis Kernel (Triton).** Jeden kernel łączy: rekurencję Czebyszewa (lub — po RFC-001 — rekurencję obrotową QTT rangi 2), kontrakcję `M_d = T_d · G^(d)` i kontrakcję łańcucha `curr ← curr · M_d`. Kluczowa własność: rdzenie `G^(d)` (`O(D·K1·R²)` ≈ dziesiątki KB) mieszczą się w **shared memory** SM; punkty strumieniowane z HBM. Zamienia obecny wzorzec ograniczony pasmem pamięci na ograniczony przez ALU.

```python
@triton.jit
def tt_kan_fused_fwd(X_ptr, cores_ptr, offs_ptr, ranks_ptr, Y_ptr,
                     N, D, K1, BLOCK_N: tl.constexpr, MAX_R: tl.constexpr):
    # BLOCK_N punktów na program; pętla po D rozwijana statycznie;
    # T[k] w rejestrach (K1 ≤ 16), core kafelkowany do shared;
    # akumulator curr[MAX_R] w rejestrach.
```

Autotuning po `(BLOCK_N, num_warps, num_stages)`. Wariant `_bwd` z tą samą strukturą prefiks/sufiks co `evaluate_tt_kan_gradient_single`, ale z `L`/`R` w rejestrach.

**2b. Rdzeń C++20.** `std::span<const double>` zamiast par `(ptr, size)` — usuwa całą klasę błędów C3. `concepts` na `KernelBackend`. `std::mdspan` (C++23, fallback: własny `Mdspan`) dla układu `(r,k,s)` — czyni C4 niewyrażalnym. Wektoryzacja: `google/highway` z dyspozycją runtime (usuwa C5, C6). Zamiast „bez GIL": jawne API bufora wyjściowego (`out=`) + trwały uchwyt modelu (C9), tak by zwolnienie GIL faktycznie coś dawało.

**KPI:** przepustowość TT-KAN fwd na A100 vs obecny C++/MSVC na 16 rdzeniach; cel ≥ 50× przy D=20, R=16, N=10⁶. Roofline: osiągnąć ≥ 60% szczytu ALU (obecnie CPU ~5% szczytu przez C4).

### Wektor 3 — Weryfikacja formalna (SMT/Z3)

Wektor jest wykonalny **tylko po naprawie M1, M4, M5, A12** — nie ma sensu dowodzić własności systemu, którego warunek bezpieczeństwa jest matematycznie niepełny.

**Podział na trzy warstwy, według tego, co da się faktycznie udowodnić:**

| Warstwa | Narzędzie | Co jest dowodzone | Kiedy |
|---|---|---|---|
| L1 — niezmienniki dyskretne | Z3 (QF_LRA / QF_LIA) | Domknięcie tranzytywne, spójność bazy faktów, niesprzeczność zbioru inwariantów `CategoryFilter`, terminacja pętli punktu stałego | offline, przy budowie bazy wiedzy |
| L2 — dopuszczalność QP | Z3 (QF_LRA) | Dla danego zbioru aktywnych ograniczeń liniowych CBF: czy przecięcie półprzestrzeni ∩ pudełko `‖u‖_∞ ≤ v_max` jest niepuste. To jest **czysto liniowe** i Z3 rozstrzyga to w µs. | runtime, przy zmianie zbioru aktywnego |
| L3 — niezmienniczość przód | dReal / Lean 4 + interwały | `∀x ∈ S: h(x) ≥ 0 ⟹ h(x + u*·dt) ≥ 0` dla pola KAN. **Nierozstrzygalne dla Z3** (wielomiany wysokiego stopnia + arytmetyka rzeczywista). | offline, certyfikacja modelu |

**Kluczowa decyzja architektoniczna:** L3 nie jest zadaniem dla SMT-solvera nad rzeczywistymi. Właściwe narzędzie to **arytmetyka interwałowa na formie Czebyszewa**: dla `h` w postaci CP/TT ograniczenie `h` na pudełku `[a,b]^D` liczy się przez propagację interwałów przez rekurencję Czebyszewa w czasie `O(D·K·R)` — deterministycznie, bez solvera, w mikrosekundach. To daje **certyfikat pokrycia** (`h ≥ ε` na całym pudełku), a nie punktową ewaluację. Rekomendacja: zaimplementować `IntervalKAN.bound_h(box) -> Interval` jako fundament L3, a Lean 4 wykorzystać do sformalizowania **dowodu poprawności samego propagatora interwałowego** (jednorazowo), nie każdego zapytania.

**Realistyczny budżet czasowy:** L1+L2 to praca inżynierska (~4 tygodnie). L3 przez interwały: ~3 tygodnie. L3 przez Lean: to projekt badawczy na kwartały — nie umieszczać w roadmapie produktowej.

### Wektor 4 — Koprocesor neuro-symboliczny dla LLM

**Warunek wstępny:** A11 (ścieżka abstynencji) i A12 (sygnalizacja niezbieżności) — bez nich moduł „bezhalucynacyjny" zawsze udziela pewnej odpowiedzi, co jest gorsze niż halucynacja modelu językowego, bo jest opatrzone certyfikatem.

**Most dwukierunkowy:**

```
LLM  ──(1) wyrażenie Kleisli w JSON/DSL──►  Parser + walidator schematu
                                                    │
                                            (2) typecheck: czy morfizm
                                                jest dobrze typowany
                                                w kategorii stanów?
                                                    │
                                        ┌───────────┴───────────┐
                                        │                       │
                                  ODRZUĆ + kontrprzykład   WYKONAJ
                                        │                       │
                                        │            (3) domknięcie tranzytywne (SCC, M7)
                                        │            (4) weryfikacja fizyczna: IntervalKAN
                                        │            (5) CategoryFilter → FilterResult
                                        └───────────┬───────────┘
                                                    │
LLM  ◄──(6) GroundingResult{answer | ABSTAIN, proof_trace, confidence}
```

**Newralgiczne punkty projektowe:**

1. **DSL, nie kod.** LLM emituje deklaratywne wyrażenie w zamkniętej gramatyce (predykaty, encje, ograniczenia przestrzenne), nie wykonywalny Python. Walidacja schematem przed jakimkolwiek wykonaniem. To jest granica bezpieczeństwa — wejście z LLM jest z definicji niezaufane.
2. **`O(1)` w deklaracji jest nieprawdziwe.** Domknięcie tranzytywne to `O(V·E/64)` po naprawie M7, nie `O(1)`. Zapytanie o osiągalność w **prekomputowanym** domknięciu to `O(1)`. Rozdzielić te dwa koszty w dokumentacji i API (`KnowledgeBase.compile()` vs `.query()`).
3. **`ABSTAIN` jest produktem, nie porażką.** Kontrakt: koprocesor zwraca `ABSTAIN` zawsze, gdy (a) margines WTA poniżej progu, (b) `FilterResult.converged == False`, (c) interwał `h` obejmuje zero. Miara jakości to nie „accuracy", lecz **precyzja przy pokryciu** — para `(precision@answered, coverage)`. Deklaracja „0% halucynacji" ma sens wyłącznie jako „precision@answered = 1.0 przy coverage = X%", gdzie X trzeba zmierzyć i podać.

---

## 4. Harmonogram wdrożenia

Fazy sekwencyjne. **Faza 0 jest blokująca dla wszystkich pozostałych** — bez CI i oracles zewnętrznych żadna z poniższych zmian nie jest weryfikowalna, a trzy najpoważniejsze błędy (M1, M2, M3) przeszły przez 134 istniejące asercje.

### Faza 0 — Fundament weryfikacji (2 tyg.)

| Zadanie | Kryterium akceptacji |
|---|---|
| CI: `{ubuntu, macos, windows} × {gcc-13, clang-17, msvc} × {py3.11, py3.12}` | Zielony build na wszystkich 12 kombinacjach; `-Wall -Wextra -Werror` na C++ |
| Job ASan/UBSan (Debug) | 0 wykrytych błędów po naprawie C2, C3 |
| `ruff` + `pyright --strict` na `src/` jako bramka | 0 błędów; `ruff` reguła NPY002 wymusza A4 |
| Oracles zewnętrzne (V3a) | `scipy.special.eval_chebyt`, `numpy.polynomial.chebyshev.chebder`; `L2_rel < 1e-8` na `PoissonAnalyticalSolution` (2D wielomianowy i trygonometryczny) |
| Testy parzystości backendów (V3c) | numpy↔cpp `atol=1e-12`; torch(f64)↔jax(x64) `atol=1e-12`; WGSL(f32)↔numpy `atol=1e-5` |
| Testy własnościowe (`hypothesis`, V3b) | ≥ 8 niezmienników, w tym: `∀t: h(p_t) ≥ −ε` na trajektorii CBF; `filter_state` osiąga punkt stały |
| `pytest-benchmark` z zapisanym baseline'em (V2) | Progi absolutne zastąpione detekcją regresji > 20% |

**KPI-0:** pokrycie `src/` (bez `src/tasks/`) ≥ 75% linii; **3 z 3** znanych błędów (M1, M2, M3) wykrywane przez nowe testy **przed** ich naprawą (test-first — to jest dowód, że bramka działa).

### Faza 1 — Poprawność (3 tyg.)

| Zadanie | Kryterium akceptacji |
|---|---|
| M1: człon Hessianu w HOCBF | `∀t: h ≥ −1e-9` na 1000 losowych scenariuszy przy `v_max = 2`; obecnie mierzalne naruszenia |
| M2: pełne człony laplasjanu w `TTPoissonSolver` | `L2_rel < 1e-6` vs rozwiązanie analityczne dla D = 4, 6, 8 |
| M3: `TuckerALS` — implementacja ogólna lub jawny `NotImplementedError` | Test dla D=3 albo przechodzi z `MSE < 1e-8`, albo rzuca wyjątek |
| M4: miękkie ograniczenia + rzut na przecięcie + `CBFResult` | 0 przypadków, w których zwrócone `u` narusza aktywne ograniczenie; `feasible=False` zawsze raportowane |
| M5: podział odpowiedzialności między agentami | 0 kolizji przy 100 losowych scenariuszach czołowych, N = 50 |
| M7: SCC + osiągalność bitsetowa | Zgodność z `networkx.transitive_closure` na 500 losowych grafach, w tym ze ścieżkami > 50 |
| M8, M9: jednolity kontrakt dziedziny | Test parzystości backendów przechodzi dla `\|x\| > 1`; wejście poza dziedziną rzuca `ValueError` |
| C2, C3: `size_t` + walidacja rozmiarów | UBSan czysty; fuzz na niespójnych `cores`/`ranks` nie powoduje odczytu poza zakresem |
| C7: usunięcie `-ffast-math` / `/fp:fast` | Testy dokładności zaostrzone o rząd wielkości nadal przechodzą |
| A4: eliminacja `np.random.seed` z biblioteki | `ruff NPY002` zielony; test: dwukrotne uruchomienie pipeline'u daje bit-identyczne wyniki |
| A2: jedno `hs_kan.io` z wersjonowaniem | Round-trip dla wszystkich 4 typów modeli; stare pliki wczytywane z ostrzeżeniem o migracji |

**KPI-1:** 0 naruszeń bariery na zestawie 1000 scenariuszy CBF (kinematyka **i** dynamika); `L2_rel < 1e-6` dla Poissona przy D ≤ 8; UBSan/ASan czyste.

### Faza 2 — Wydajność i uwarunkowanie (4 tyg.)

| Zadanie | Kryterium akceptacji |
|---|---|
| C1: `dsyrk` blokowy w DMRG | Pamięć szczytowa `< 2 GB` przy `max_rank=32, degree=7` (obecnie: OOM lub UB) |
| C4: zamiana kolejności pętli | ≥ 3× przyspieszenie `evaluate_tt_kan_batch` (D=20, R=16, N=10⁵), zmierzone `pytest-benchmark` |
| C5, C6: `omp simd` + `highway` z dyspozycją runtime | Moduł importuje się i przechodzi testy na CPU bez AVX2 (wymuszone `QEMU`/`-mno-avx2`) |
| C8: scratch per-wątek | 0 alokacji sterty w gorącej pętli (weryfikacja `heaptrack`) |
| C9: trwały uchwyt modelu | 0 alokacji na wywołanie `evaluate` po `bind()`; ≥ 5× przyspieszenie dla N < 1000 |
| N1: `lstsq` na `[Φ; √αI]` + ekwilibracja + `α` względne | κ(A) zamiast κ(AᵀA): dla D=3, K=10 błąd L2 spada z ~1e-4 do `< 1e-10` |
| N3: stos prefiksów w DMRG/ALS | Czas sweepu liniowy w D (współczynnik determinacji dopasowania liniowego R² > 0.98 dla D ∈ [10,100]) |
| N4: prefiks/sufiks w gradiencie NumPy CP | ≥ D/2 przyspieszenie przy D = 50 |
| N6: wektoryzacja TT-Cross + użycie `Z` z maxvol | Liczba wywołań `func` ≤ 55% obecnej przy tej samej dokładności na zbiorze walidacyjnym |
| N8: `lax.scan` + `enable_x64` w JAX | Czas `jit` niezależny od D (< 5 s dla D = 100); parzystość z torch f64 przy `atol=1e-12` |
| A10: postać dualna dla `N < K` | `NarySpatioTemporalEngine` działa dla 1000 encji przy `< 4 GB` (obecnie: 231 GB) |
| V1: benchmark vs prawdziwy `pykan`/`efficient-kan` | Raport z wersjami bibliotek, modelem CPU, medianą i IQR z ≥ 20 powtórzeń |

**KPI-2:** ≥ 3× przepustowość TT-KAN na CPU; ≥ 6 odzyskanych cyfr znaczących w solverach ALS/PDE; ≤ 55% wywołań funkcji celu w TT-Cross; zero regresji poprawności z Fazy 1.

### Faza 3 — RFC-001: silnik QTT (8 tyg.)

| Etap | Zadanie | Kryterium akceptacji |
|---|---|---|
| 3.1 (2 tyg.) | `QTTField`, `QTTCore`, `round()`, forma kanoniczna, `evaluate` przez rekurencję obrotową rangi 2 | Parzystość z `to_tt().evaluate()` przy `atol=1e-12` dla L ∈ [3,12]; zgodność z `ttpy`/`torchtt` jako oracle zewnętrzny |
| 3.2 (2 tyg.) | `QTTOperator`: `diag_k`, `shift`, `D_λ`, `S_λ`, `laplacian` | Ranga **dokładnie** 2 dla `diag_k` i `shift` (weryfikacja przez TT-SVD, `atol=1e-14`); `‖D_λ f − analytical‖ < 1e-11` na wielomianach |
| 3.3 (2 tyg.) | `amen_solve` z prekondycjonerem Jacobiego + `AMEnReport` | `κ(operator) < 10²` **niezależnie od L ∈ [6,20]** — to jest główna teza RFC; `L2_rel < 1e-10` dla Poissona 2D/3D |
| 3.4 (1 tydz.) | `qtt_convolve` (cykliczny + Czebyszew) | Zgodność ze splotem gęstym przy `atol=1e-10` dla L ≤ 14 |
| 3.5 (1 tydz.) | `from_function` (AMEn-Cross), `from_tt`, migracja `PoissonSolver` w fasadzie | Ścieżka migracji: istniejące modele TT wczytywalne jako QTT bez utraty dokładności |

**KPI-3 (bramki twarde, każda mierzona na zestawie referencyjnym z §4.1):**

- **KPI-3.1 — pamięć:** ≥ 100× redukcja vs TT nodalne przy `K = 2^16`, `D = 3`, przy `L2_rel < 1e-8`.
- **KPI-3.2 — ranga:** `max_rank ≤ 16` dla ≥ 80% funkcji zestawu referencyjnego przy `eps = 1e-10`. Poniżej tego progu QTT nie jest opłacalne i decyzja o wdrożeniu wymaga rewizji.
- **KPI-3.3 — uwarunkowanie:** `κ` operatora Poissona `< 10²` dla `L ∈ [6,20]`, wobec zmierzonych 1.3e12 obecnie. **To jest główny wynik do wykazania.**
- **KPI-3.4 — wywołania funkcji celu:** `O(D·L·ρ²)` potwierdzone empirycznie — liczba wywołań rośnie **logarytmicznie** z K (dopasowanie `log`, R² > 0.95 dla K ∈ [2^6, 2^16]).
- **KPI-3.5 — próg opłacalności:** udokumentowany punkt przecięcia z TT nodalnym (przewidywany `K ≈ 64`), zmierzony, zapisany w README.

**Zestaw referencyjny (§4.1)** — 12 funkcji z uzasadnieniem doboru: gładkie separowalne (`exp`, `sin` — oczekiwana ranga 1–2, weryfikacja dokładności konstrukcji), gładkie sprzężone (Rosenbrock, Ackley), ostry front (`tanh(x/δ)`, δ = 1e-3 — główny przypadek użycia), SDF sfery i torusa 3D, pole wirowe 2D, rozwiązanie Poissona 3D, funkcja nieciągła (schodek), szum (przypadek negatywny — ranga **musi** eksplodować, inaczej implementacja jest błędna).

### Faza 4 — Wektory 2–4 (równolegle, po Fazie 3)

| Tor | Czas | Główny KPI |
|---|---|---|
| Wektor 2 (Triton/CUDA + C++20) | 6 tyg. | ≥ 50× vs CPU na A100 (D=20, R=16, N=10⁶); ≥ 60% szczytu ALU (roofline) |
| Wektor 3 (L1+L2 Z3, L3 interwałowy) | 7 tyg. | 100% zapytań L2 < 100 µs; `IntervalKAN.bound_h` daje certyfikat pokrycia dla ≥ 99% pudełek testowych bez fałszywych negatywów |
| Wektor 4 (koprocesor LLM) | 6 tyg. | `precision@answered = 1.0` przy `coverage ≥ 70%` na zestawie 500 zapytań; 100% niepoprawnie typowanych wyrażeń odrzuconych z kontrprzykładem |

---

## 5. Trzy rzeczy, które trzeba zrobić w tym tygodniu

Niezależnie od roadmapy, w kolejności ryzyka:

1. **Wyłączyć `-ffast-math` / `/fp:fast`** (C7) i dodać strażnik `size_t` na `P*P` (C2). Pierwsze unieważnia każdą deklarację stabilności numerycznej, drugie to nieokreślone zachowanie przy legalnej konfiguracji użytkownika.
2. **Naprawić lub jawnie oznaczyć M3** (`TuckerALS` cicho nie robi nic dla D≠2). Cicha awaria zwracająca metrykę sukcesu jest najgorszą kategorią błędu — jedna linia `raise NotImplementedError` usuwa problem natychmiast.
3. **Usunąć lub przepisać `benchmarks/benchmark_vs_pykan.py`** (V1). Porównanie z własnoręcznie napisaną atrapą opisaną jako „B-spline", która w rzeczywistości używa funkcji Gaussa, jest pojedynczym elementem repozytorium, który przy zewnętrznym przeglądzie podważy wiarygodność wszystkich pozostałych — w tym tych uczciwych.

---

*Wszystkie liczby oznaczone „zmierzone" pochodzą z odtworzenia odpowiedniego fragmentu kodu i uruchomienia go; pozostałe stwierdzenia pochodzą z lektury źródeł na rewizji `d3bf28d`.*
