# Plan rozwoju — 12 tygodni, cel: portfolio / dowód kompetencji

**Kontekst decyzyjny:** kierunek = „wszystkie powyższe", artefakt = **portfolio / dowód kompetencji**.
**Rewizja bazowa:** `d3bf28d` · **Podstawa:** `AUDYT_HYPER_SYMBOLIC_KAN.md`

---

## 0. Rozstrzygnięcie konfliktu w celach

Cztery tory naraz (konsolidacja + ekspansja + pivot QTT + równolegle) to ~9 miesięcy pracy solo. Drugi wybór — portfolio — narzuca jednak jednoznaczną kolejność, bo zmienia funkcję celu:

> Recenzent techniczny poświęca temu repozytorium **20–60 minut**. Czyta README, otwiera jeden demo, przegląda jeden moduł źródłowy. Wygrywa nie ten projekt, który ma najwięcej funkcji, tylko ten, w którym **nie znajdzie nic, co podważy zaufanie do reszty**.

Z tego wynika ranking, który jest inny niż w `ROADMAP_PHASE_2.md`:

| Tor | Wartość dla portfolio | Decyzja |
|---|---|---|
| Konsolidacja (poprawność + CI + uczciwe deklaracje) | **Najwyższa.** Usuwa to, co kończy przegląd w 10. minucie. Dodatkowo sam proces audytu jest materiałem portfolio. | **Rdzeń planu, tyg. 0–7** |
| Pivot QTT | **Wysoka, ale tylko jako jeden obroniony wynik.** Jedna figura „κ: 1.3e12 → 1e2" jest warta więcej niż cała Faza II. | **Zawężony PoC, tyg. 8–11** |
| Ekspansja funkcji (Faza II F1–F5) | **Ujemna.** Każdy nowy showcase oparty na `TTPoissonSolver` (M2) i HOCBF bez Hessianu (M1) powiększa powierzchnię, którą trzeba będzie obronić. | **Wstrzymane** |
| Wszystko równolegle | Rozprasza jedyny zasób, którym jest Twój czas. | **Odrzucone** |

**Kontrintuicyjny wniosek:** dla portfolio naprawa własnych błędów jest mocniejszym sygnałem niż nowa funkcjonalność. Zdanie „przeprowadziłem audyt własnego kodu, znalazłem że mój HOCBF pomija człon Hessianu, naprawiłem i oto test, który to wykrywa" świadczy o dojrzałości inżynierskiej lepiej niż szósty scenariusz w `web_showcase/`.

---

## 1. Sprint 0 — zatrzymanie krwawienia (jeden weekend, ~6 h)

Trzy rzeczy, które kończą przegląd zanim recenzent dojdzie do wartościowej części. Każda to zmiana poniżej godziny.

- [ ] **`benchmarks/benchmark_vs_pykan.py`** — usunąć plik albo przepisać na prawdziwy `pykan`. W obecnej formie `SimulatedBSplineKANLayer` używa `np.exp(-0.5*((X-center)/0.2)**2)` (funkcje Gaussa) i opisuje to jako „Cox-de Boor spline basis", a `pykan` nie jest importowany. **To jest pojedynczy plik o największym ujemnym wpływie w całym repo.**
  *DoD:* plik usunięty, albo `pip install pykan` w `requirements-bench.txt` i mierzony rzeczywisty kod, z wersjami bibliotek i modelem CPU w nagłówku raportu.
- [ ] **`CMakeLists.txt`** — usunąć `-ffast-math` i `/fp:fast`. Zostawić `-ffp-contract=fast`.
  *DoD:* istniejące testy dokładności przechodzą z progami zaostrzonymi o rząd wielkości.
- [ ] **`tucker_als.py::TuckerALSSolver.fit`** — pierwsza linia: `if D != 2: raise NotImplementedError(f"TuckerALS supports D=2, got D={D}")`.
  *DoD:* test dla D=3 rzuca wyjątek zamiast zwracać MSE losowej inicjalizacji.
- [ ] **`facade.py`** — usunąć z docstringów liczby, których kod nie dostarcza: „< 0.3 µs per point" (`predict()` woła czysty NumPy), „Zero-copy conversion" (są dwie kopie).
  *DoD:* `grep -rn "µs\|us per point\|zero-copy\|0%\|100%" src/` — każde trafienie albo ma test, albo znika.

**Artefakt:** repozytorium, które przetrwa pierwsze 10 minut przeglądu.

---

## 2. Sprint 1 — infrastruktura weryfikacji (tyg. 1–2)

Dla portfolio zielony badge CI na trzech platformach to widoczny sygnał o bardzo dobrym stosunku wartości do nakładu.

- [ ] **`.github/workflows/ci.yml`** — matryca `{ubuntu-latest, macos-latest, windows-latest} × {py3.11, py3.12}`.
  *DoD:* zielony build; badge w README. Uwaga: `build/` zawiera dziś wyłącznie artefakty MSVC — **ścieżka GCC/Clang nigdy nie została zbudowana**, więc spodziewaj się błędów kompilacji przy pierwszym uruchomieniu (m.in. `#pragma loop(ivdep)` to składnia wyłącznie MSVC).
- [ ] **Flagi kompilatora** — `-Wall -Wextra -Wunknown-pragmas -Werror`.
  *DoD:* zero ostrzeżeń; wszystkie `#pragma loop(ivdep)` zamienione na `#pragma omp simd` lub usunięte.
- [ ] **`-mavx2` za dyspozycją runtime** — obecnie bezwarunkowo na x86_64 → `SIGILL` przy imporcie na CPU bez AVX2.
  *DoD:* moduł importuje się pod `-mno-avx2` / QEMU.
- [ ] **Job ASan + UBSan** (Debug, tylko Linux).
  *DoD:* czysty po naprawie C2 (`P*P` w `int` przepełnia się przy `rank=32, degree=7`) i C3 (bindingi nie walidują `cores_flat.size()`).
- [ ] **`ruff` + `pyright --strict`** jako bramka. Reguła `NPY002` wymusza eliminację `np.random.seed` z biblioteki.
  *DoD:* zero błędów; `np.random.seed` wyłącznie w `src/tasks/` i testach.
- [ ] **Oracles zewnętrzne** — `scipy.special.eval_chebyt`, `numpy.polynomial.chebyshev.chebder`, rozwiązania analityczne z twardym progiem.
  *DoD:* ≥ 6 nowych testów porównujących z niezależną implementacją, nie z własnym fallbackiem.
- [ ] **Testy parzystości backendów** — numpy↔cpp `atol=1e-12`, torch(f64)↔jax(x64) `atol=1e-12`, WGSL(f32)↔numpy `atol=1e-5`.
  *DoD:* przechodzą **także dla `|x| > 1`** — dziś `_compute_chebyshev_torch` nie klipuje, a NumPy/C++ klipują, więc ten test na starcie padnie. To jest zamierzone.
- [ ] **`pytest-benchmark` z zapisanym baseline'em** zamiast progów absolutnych (`assert throughput >= 1_500_000` flakuje na innym sprzęcie).

**Artefakt:** badge CI × 3 platformy + test suite, który wykrywa realne błędy.

---

## 3. Sprint 2 — poprawność matematyczna (tyg. 3–5)

**Kolejność jest istotna: najpierw test, który dowodzi błędu, potem naprawa.** To jest sedno materiału portfolio — dowód, że bramka działa.

- [ ] **M1 — człon Hessianu w HOCBF** (`robotics_cbf_planner.py::solve_dynamic_hocbf_qp`)
  Warunek eksponencjalnego CBF wymaga `ḧ = ∇h·a + vᵀ∇²h v`; kod buduje ograniczenie wyłącznie z `∇h·a`. Dla sfery `∇²h = (I − êêᵀ)/‖p−c‖ ≠ 0`. Przy `v_max=2`, `d=0.1` pominięty człon ma rząd 40 m/s² — czterokrotność `a_max`.
  Baza jest: `chebyshev_derivatives_2nd` w `pde_poisson_solver.py`. Dla CP-KAN: `∂²f/∂x_i∂x_j = Σ_r λ_r dφ_i dφ_j ∏_{m≠i,j} φ_m` — prefiks/sufiks, `O(N·R·D)`.
  *DoD:* test na 1000 losowych scenariuszy dynamicznych **pada przed naprawą**, przechodzi po (`∀t: h ≥ −1e-9`).
- [ ] **M2 — pełne człony laplasjanu** (`pde_poisson_solver.py::TTPoissonSolver.fit_als`)
  Macierz projektowa budowana tylko z członu `m=d`; człony `m≠d` też są liniowe w rdzeniu `d` (rdzeń występuje w ich `L`/`R`). Solver **nie rozwiązuje Poissona dla D≥4**.
  *DoD:* `L2_rel < 1e-6` vs rozwiązanie analityczne dla D = 4, 6, 8.
- [ ] **M4 — bezpieczna ścieżka awaryjna CBF**
  Po niepowodzeniu SLSQP: rzut na *jedno* ograniczenie + `np.clip` (niszczy spełnienie CBF). W `simulate_swarm` fallback to sterowanie **całkowicie niefiltrowane**.
  Naprawa: miękkie ograniczenia ze zmienną luzu `δ ≥ 0` i karą `ρδ²` (ρ≈1e4) → QP zawsze dopuszczalne; `v_max` jako ograniczenie pudełkowe *wewnątrz* QP; zwracać `CBFResult{u, slack, feasible, active_set}`.
  *DoD:* zero przypadków, w których zwrócone `u` narusza aktywne ograniczenie; `feasible=False` zawsze propagowane.
- [ ] **M5 — podział odpowiedzialności między agentami** (`InterAgentCBF`)
  Agent `i` zakłada `ṗ_j = 0`; obaj spełniają własne ograniczenie, para narusza wspólne. Naprawa: `−∇h_ij·u_i ≤ ½·α·h_ij` po obu stronach.
  *DoD:* 0 kolizji na 100 scenariuszach czołowych, N=50.
- [ ] **M7 — domknięcie tranzytywne przez SCC** (`clifford_algebra.py`)
  `max_depth=50` cicho obcina ścieżki; akumulacja `float32` przepełnia się do `inf`; typ zwracany niestabilny (`ndarray` vs `csr_matrix`). Naprawa: kondensacja Tarjana + osiągalność bitsetowa.
  *DoD:* zgodność z `networkx.transitive_closure` na 500 grafach, w tym ze ścieżkami > 50.
- [ ] **M8/M9 — jednolity kontrakt dziedziny**
  `np.clip` nie propaguje się do gradientu: dla punktu poza `[-1,1]^D` zwracane jest `T'_K(±1) = K²` (25/100/256/576 dla K=5/10/16/24) zamiast 0. Planer CBF dostaje ogromny, kierunkowo błędny gradient.
  Naprawa: klipowanie **poza** jądrami, w jawnej warstwie `DomainWindow` (`sliding_domain.py` już istnieje); jądro assertuje wejście.
  *DoD:* test parzystości dla `|x|>1` przechodzi; wejście poza dziedziną rzuca `ValueError`.
- [ ] **A4 — eliminacja globalnego RNG** — `np.random.seed(42)` w `__init__` modeli TT resetuje RNG całego procesu.
  *DoD:* dwukrotne uruchomienie pipeline'u daje bit-identyczne wyniki.

**Artefakt portfolio (najważniejszy w całym planie):** dokument `docs/BUG_HUNT.md` — „Trzy błędy matematyczne, których nie wykryło 134 asercji". Dla każdego: objaw, przyczyna, dlaczego testy go nie łapały, wykres przed/po, test który go teraz łapie. **To jest materiał, który realnie odróżnia to repozytorium od setek innych projektów ML na GitHubie.**

---

## 4. Sprint 3 — uwarunkowanie i wydajność (tyg. 6–7)

- [ ] **N1 — koniec z układami normalnymi.** Wzorzec `A = ΦᵀΦ + αI; solve(A, Φᵀy)` w 6 solverach podnosi uwarunkowanie do kwadratu. Zmierzone: κ(A)=1.1e6 → κ(AᵀA)=1.3e12 przy D=3, K=10 (**domyślna konfiguracja fasady**), zostają 4 cyfry.
  Naprawa: `scipy.linalg.lstsq` na `[Φ; √α·I]` + ekwilibracja kolumnowa + `α` względne (`α_rel · trace/n`). Zawsze raportować `cond` i `rank` w słowniku wynikowym.
  *DoD:* błąd L2 dla D=3, K=10 spada z ~1e-4 do < 1e-10.
- [ ] **C4 — kolejność pętli w jądrze C++.** `for s { for k { m += T[k]*core[k*r_next+s] } }` — redukcja po `k` z krokiem `r_next`. Zamiana na `for k { t=T[k]; for s { m[s] += t*core[k*r_next+s] } }` daje ciągły `axpy`.
  *DoD:* ≥ 3× przyspieszenie `evaluate_tt_kan_batch` (D=20, R=16, N=1e5), zmierzone `pytest-benchmark`.
- [ ] **C1 — `dsyrk` blokowy w DMRG.** `local_A(P*P)` per wątek = **679 MB przy domyślnych parametrach**, ×16 wątków = 10.9 GB.
  *DoD:* pamięć szczytowa < 2 GB przy `max_rank=32, degree=7`.
- [ ] **C9 — trwały uchwyt modelu.** `np.concatenate([...])` przepakowuje rdzenie przy **każdym** wywołaniu (1.2 MB dla D=100), mimo docstringa „brak alokacji na stercie per zapytanie".
  *DoD:* zero alokacji po `bind()`; ≥ 5× przyspieszenie dla N < 1000.
- [ ] **N3 — stos prefiksów** zamiast przeliczania `L`/`R` od zera dla każdego węzła (`O(D²)` → `O(D)` na sweep).
  *DoD:* czas sweepu liniowy w D (R² > 0.98 dla D ∈ [10,100]).
- [ ] **N8 — `lax.scan` + `jax_enable_x64`.** Backend JAX liczy dziś w **float32**, deklarując float64 (`jax.config.update` nie występuje nigdzie w repo).

**Artefakt:** `docs/BENCHMARKS.md` z wykresami: κ przed/po, przyspieszenie C4, i uczciwym porównaniem z `pykan`/`efficient-kan`.

---

## 5. Sprint 4 — QTT proof-of-concept, zawężony (tyg. 8–11)

**Nie budujemy silnika z RFC-001.** Budujemy najmniejszy fragment, który dowodzi tezy i daje jedną zrozumiałą figurę.

Zakres — wyłącznie 1D, bez `D`-wymiarowego łańcucha, bez AMEn, bez splotu:

- [ ] **`QTTVector`** — łańcuch `L` rdzeni binarnych `(ρ,2,ρ)`, `round()` przez TT-SVD, forma kanoniczna.
  *DoD:* parzystość z gęstą reprezentacją `atol=1e-12` dla L ∈ [3,14]; oracle: `ttpy` lub `torchtt`.
- [ ] **Baza Czebyszewa jako QTT rangi 2** — przez `T_k(cos θ) = cos kθ` i macierze obrotu `G^(l)[b] = R(b·2^l·θ)`.
  *DoD:* ranga **dokładnie** 2 zweryfikowana przez TT-SVD (`atol=1e-14`); ewaluacja zgodna z `scipy.special.eval_chebyt`.
- [ ] **`diag_k`, `shift`, `D_λ`, `S_λ`** jako MPO.
  *DoD:* ranga dokładnie 2 dla `diag_k` i `shift`; `‖D_λ f − pochodna analityczna‖ < 1e-11` na wielomianach.
- [ ] **Pomiar κ: baza nodalna vs ultrasferyczna-QTT, dla K ∈ [2^3, 2^16].**
  *DoD:* wykres pokazujący `κ ~ O(K⁴)` dla nodalnej i `κ = O(1)` dla ultrasferycznej.

**Artefakt:** notebook + **jedna figura**: κ w funkcji K, dwie krzywe, skala log. Recenzent rozumie ją w 5 sekund. To jest najmocniejszy pojedynczy element techniczny, jaki możesz z tego projektu wyprodukować.

**Punkt kontrolny na koniec tyg. 9:** jeśli `QTTVector` + baza rangi 2 nie działają, **przerwij tor QTT** i przeznacz tyg. 10–11 na Sprint 5. Figura, której nie ma, jest lepsza niż figura, która kłamie.

---

## 6. Sprint 5 — narracja (tyg. 12)

Dla portfolio README *jest* produktem. Większość recenzentów nie dojdzie dalej.

- [ ] **Tabela deklaracji** zamiast marketingu — dla każdej: warunki, zmierzona wartość, test który ją weryfikuje.

  | Deklaracja | Warunki | Zmierzone | Test |
  |---|---|---|---|
  | Dopasowanie bez epok gradientowych | CP, D≤10, K≤8 | 12 ms, L2_rel 3e-11 | `test_als_accuracy` |
  | Analityczny gradient | wszystkie backendy | zgodność z FD 1e-9 | `test_gradient_parity` |
  | ~~0% naruszeń CBF~~ | kinematyka, dt≤0.01, QP dopuszczalne | 0/1000 scenariuszy | `test_cbf_invariance` |

  Trzeci wiersz pokazuje wzorzec: deklaracja zawężona do warunków, w których jest prawdziwa, jest **mocniejsza** niż deklaracja ogólna, bo jest sprawdzalna.
- [ ] **`docs/LIMITATIONS.md`** — jawnie: CP w float32 zeruje się przy D≳60 (mediana `∏φ` = 6e-42 dla D=100); `SpectralKANPoissonSolver` nieużywalny dla D>3; QTT opłacalne dopiero od K≳64. Sekcja ograniczeń jest sygnałem dojrzałości, nie słabości.
- [ ] **Jeden demo, nie pięć.** Wybrać najmocniejszy scenariusz z `web_showcase/`, dopracować, resztę oznaczyć jako eksperymentalne.
- [ ] **README:** co to jest → jedna figura κ → tabela deklaracji → link do `BUG_HUNT.md` → jak uruchomić.

---

## 7. Lista cięć (świadomie NIE robimy)

| Element | Dlaczego nie |
|---|---|
| Nowe scenariusze `web_showcase/` (Faza II) | Każdy opiera się na solverach z M1/M2. Powiększanie powierzchni przed naprawą fundamentu. |
| Publikacja npm `@hyper-kan/webgpu` | WGSL ma na sztywno wpisane `rank=8, degree=5`; publikacja utrwala błędny kontrakt. |
| Kernele Triton/CUDA (Wektor 2) | Optymalizacja przed naprawą C4 (kolejność pętli) to optymalizacja błędnego kodu. Dla portfolio CPU 3× wystarcza jako dowód kompetencji. |
| Z3 / Lean (Wektor 3) | Nie ma sensu dowodzić własności systemu, którego warunek bezpieczeństwa jest niepełny (M1). L3 i tak nie jest zadaniem dla SMT — właściwe narzędzie to arytmetyka interwałowa. |
| Koprocesor LLM (Wektor 4) | Wymaga ścieżki abstynencji (A11) i sygnalizacji niezbieżności (A12), których nie ma. Bez nich „0% halucynacji" jest gorsze niż halucynacja — jest halucynacją z certyfikatem. |
| Pełny silnik QTT z RFC-001 | 8 tygodni na PoC vs 6 miesięcy na silnik. Dla portfolio liczy się figura, nie API. |

---

## 8. Wariant minimalny (jeśli masz 3 weekendy, nie 12 tygodni)

W tej kolejności, każdy punkt niezależnie wartościowy:

1. **Sprint 0 w całości** (~6 h) — usuwa to, co kończy przegląd.
2. **CI + testy parzystości backendów** (~8 h) — badge i dowód, że coś weryfikujesz.
3. **M1 + `docs/BUG_HUNT.md` tylko dla HOCBF** (~10 h) — jeden dobrze opisany, znaleziony i naprawiony błąd matematyczny.
4. **Tabela deklaracji w README** (~3 h) — zamiana marketingu na sprawdzalne stwierdzenia.

To ~27 h i daje 80% wartości portfolio całego planu. Reszta jest optymalizacją.

---

## 9. Ryzyka planu

| Ryzyko | Mitygacja |
|---|---|
| CI pada masowo przy pierwszym uruchomieniu na GCC/Clang | Spodziewane — ścieżka nie-MSVC nigdy nie była budowana. Zarezerwować cały tydzień 1 na to, nie 2 dni. |
| Naprawa M4 (miękkie ograniczenia) zmienia zachowanie showcase'ów | Trajektorie się zmienią. Zaktualizować zrzuty/wideo, nie cofać naprawy. |
| QTT PoC nie domyka się w 4 tygodnie | Twardy punkt kontrolny na koniec tyg. 9 (§5). |
| Zniechęcenie przy usuwaniu własnych deklaracji | Zawężona deklaracja z testem jest mocniejsza niż ogólna bez testu. `LIMITATIONS.md` czyta się jako pewność siebie, nie jako przyznanie porażki. |
