# CLAUDE.md — instrukcje dla agentów pracujących w tym repozytorium

## 0. Cel repozytorium

To repozytorium jest **portfolio / dowodem kompetencji**, nie produktem. Recenzent
techniczny poświęci mu 20–60 minut: przeczyta README, otworzy jedno demo, przejrzy
jeden moduł źródłowy. Wygrywa nie ten projekt, który ma najwięcej funkcji, tylko ten,
w którym recenzent **nie znajdzie nic, co podważy zaufanie do reszty**.

Z tego wynika kryterium akceptacji dla każdej zmiany:

> Usuwamy rzeczy, które podważają zaufanie do reszty kodu. Zawężona deklaracja
> z testem jest mocniejsza niż deklaracja ogólna bez testu.

---

## 1. Punkt wejścia — przeczytaj w tej kolejności

1. **`AUDYT_HYPER_SYMBOLIC_KAN.md`** — audyt techniczny na rewizji `d3bf28d`. Stan
   faktyczny kodu: znalezione błędy matematyczne (M*), problemy C++ (C*),
   numeryczne (N*), API (A*) i weryfikacyjne (V*). To jest źródło prawdy o tym, co
   w tym repozytorium działa, a co nie.
2. **`PLAN_ROZWOJU_12_TYGODNI.md`** — plan prac wynikający z audytu, z rankingiem
   i listą świadomych cięć (§7 „Lista cięć — świadomie NIE robimy").

### Dokumenty przestarzałe — nie startuj z nich

`NEXT_AGENT_PROMPT.md` i `ROADMAP_NEXT_SESSIONS.md` opisują stan, który audyt
częściowo obalił. W szczególności:

- `NEXT_AGENT_PROMPT.md` opisuje Etap A jako „najbliższą sesję" — Etapy A–E zostały
  już zaimplementowane; ten prompt jest historyczny.
- Deklarowane w nich cele wydajnościowe (`< 0.2 us` opóźnienia) nie są tym, co
  mierzą dzisiejsze testy (`tests/test_cpp_kernels.py` dopuszcza `<= 0.67 us`).
- `ROADMAP_NEXT_SESSIONS.md` / `ROADMAP_PHASE_2.md` planują dalszą **ekspansję
  funkcji**. Plan 12-tygodniowy ocenia jej wartość dla portfolio jako **ujemną**,
  dopóki nie zostaną naprawione M1/M2/M4/M5/M7 — każdy nowy showcase oparty na
  wadliwym solverze powiększa powierzchnię, którą trzeba będzie obronić.

Traktuj te trzy pliki jako zapis historii, nie jako listę zadań.

---

## 2. Reguła: każda deklaracja liczbowa wymaga testu

**Każda liczbowa deklaracja w docstringu, komentarzu, stringu wypisywanym
użytkownikowi lub w README musi mieć test, który ją weryfikuje. Jeżeli testu nie
ma — deklaracja znika.** Trzeciej opcji nie ma.

Dotyczy to w szczególności:

- progów wydajności (`< 0.3 us / punkt`, `> 750 000 punktów/s`),
- deklaracji dokładności (`100% dokładności`, `0% violation rate`, `błąd 0.00000000`),
- deklaracji o właściwościach pamięciowych (`zero-copy`, `brak alokacji per zapytanie`),
- notacji złożoności (`O(|E|·depth)`, `O(1)`) — jeżeli nie odpowiada rzeczywistemu
  kosztowi, jest deklaracją liczbową jak każda inna.

Poprawny wzorzec — deklaracja **zawężona do warunków, w których jest prawdziwa**,
z nazwą testu:

```
Zakres gwarancji: brak kolizji jest sprawdzany na pojedynczych scenariuszach
w `tests/test_applications.py`. NIE jest to gwarancja w reżimie dynamicznym --
warunek HOCBF pomija człon Hessianu (audyt M1, otwarty).
```

Przed zamknięciem zadania uruchom:

```
grep -rn "us per point\|zero-copy\|0%\|100%\|1e-14" src/
```

Każde trafienie musi być albo pokryte testem, albo strażnikiem numerycznym w kodzie
(np. `if total_energy < 1e-14`), albo usunięte. Rozszerzaj ten grep o nowe wzorce,
gdy je wprowadzasz.

---

## 3. Zakaz porównań z atrapami napisanymi we własnym repo

**Nie wolno benchmarkować ani walidować tego kodu względem implementacji
referencyjnej napisanej w tym repozytorium.** Porównanie z własnoręcznie napisaną,
nieoptymalną atrapą nie mierzy niczego, a przy pierwszym zewnętrznym przeglądzie
dyskredytuje również uczciwe wyniki.

Konkretny precedens (audyt V1): `benchmarks/benchmark_vs_pykan.py` deklarował
porównanie z PyKAN, a w rzeczywistości zawierał klasę `SimulatedBSplineKANLayer`
opisaną jako „Cox-de Boor spline basis", która liczyła funkcje Gaussa
(`np.exp(-0.5*((X-center)/0.2)**2)`); `pykan` nie był w ogóle importowany. Plik
został usunięty.

Zasady dla benchmarków porównawczych:

- Zależność instalowana z zewnątrz (`requirements-bench.txt`), realny import,
  mierzony rzeczywisty kod biblioteki — `pykan`, `efficient-kan`, `fastkan`, `torchtt`.
- Nagłówek raportu: wersje bibliotek, model CPU, flagi kompilacji, `n_repeat`.
- Raportuj **medianę i IQR**, nie średnią.
- Jeżeli zależności nie da się zainstalować — nie ma benchmarku i nie ma jego nazwy
  w repozytorium.

Ta sama zasada dotyczy testów poprawności (audyt V3): porównanie backendu C++
z fallbackiem NumPy napisanym przez tego samego autora z tego samego wzoru jest
**testem spójności, nie poprawności**. Oracle musi być zewnętrzny —
`scipy.special.eval_chebyt`, `numpy.polynomial.chebyshev.chebder`,
`networkx.transitive_closure`, rozwiązanie analityczne z twardym progiem.

---

## 4. Znane, jeszcze NIENAPRAWIONE błędy

**Nie buduj nowych funkcji na tych ścieżkach.** Każdy nowy scenariusz oparty na
poniższych solverach dziedziczy ich błąd i powiększa powierzchnię do obrony. Jeżeli
zadanie wymaga któregoś z nich — najpierw napraw błąd (test-first: najpierw test,
który dowodzi błędu, potem naprawa), albo wybierz inną ścieżkę.

| Id | Lokalizacja | Na czym polega | Konsekwencja |
|---|---|---|---|
| **M1** | `applications/robotics_cbf_planner.py::solve_dynamic_hocbf_qp` | HOCBF pomija człon Hessianu. Warunek eksponencjalnego CBF wymaga `ḧ = ∇h·a + vᵀ∇²h v`; kod buduje ograniczenie wyłącznie z `∇h·a`, milcząco zakładając `∇²h ≡ 0`. Dla sfery `∇²h = (I − êêᵀ)/‖p−c‖ ≠ 0`. | Przy `v_max=2`, `d=0.1` pominięty człon ma rząd 40 m/s² — czterokrotność `a_max`. Każda deklaracja bezpieczeństwa w reżimie **dynamicznym** jest nieważna. |
| **M2** | `applications/pde_poisson_solver.py::TTPoissonSolver.fit_als` | Macierz projektowa budowana wyłącznie z członu `m = d` laplasjanu. Człony `m ≠ d` też są liniowe w rdzeniu `d` i są całkowicie pominięte. | ALS minimalizuje residuum **innego operatora niż `∇²`**. Solver nie rozwiązuje równania Poissona dla D ≥ 4. |
| **M4** | `applications/robotics_cbf_planner.py` (kinematyka + rój) | Ścieżka awaryjna po niepowodzeniu SLSQP: rzut na **jedno** ograniczenie, potem `np.clip(u, ±v_max)` — clipping niszczy spełnienie CBF. W `simulate_swarm` fallback to sterowanie **całkowicie niefiltrowane**. | Dokładnie w reżimie, w którym filtr jest potrzebny, system zwraca sterowanie niebezpieczne — i nie sygnalizuje degradacji wywołującemu. |
| **M5** | `applications/robotics_cbf_planner.py::InterAgentCBF` | Zdecentralizowany CBF bez podziału odpowiedzialności. Agent `i` zakłada `ṗ_j = 0`. | Obaj agenci spełniają własne ograniczenie, para narusza wspólne → kolizje przy zbliżeniach czołowych, mimo raportowanego braku kolizji (metryka mierzy `min_dist` dopiero po kroku Eulera). |
| **M7** | `hs_ckan/clifford_algebra.py::compute_transitive_closure_matrix` | `max_depth=50` **cicho** obcina ścieżki dłuższe niż 50 krawędzi. Akumulacja `closure += M^k` w `float32` zlicza ścieżki i przepełnia się do `inf` (k≈38 przy średnim stopniu 10). Typ zwracany niestabilny: `ndarray` dla N≤2000, `csr_matrix` powyżej. | Domknięcie przechodnie jest niepoprawne. Cały `qa_engine` odczytuje z tej macierzy — odpowiedzi „są połączone / brak relacji" mogą być fałszywe. |

Pozostałe otwarte pozycje (M8/M9, N1, N3, N8, C1–C9, A4, V3, V4) — patrz tabela
w `AUDYT_HYPER_SYMBOLIC_KAN.md`.

### Naprawione w Sprint 0

- **V1** — `benchmarks/benchmark_vs_pykan.py` usunięty (atrapa opisana jako B-spline).
- **C7** — `-ffast-math` / `/fp:fast` usunięte z `CMakeLists.txt`; został
  `-ffp-contract=fast` (GCC/Clang) i `/fp:precise` + `/fp:contract` (MSVC). Progi
  dokładności w `tests/test_cpp_kernels.py` i `tests/test_tt_cross.py` zaostrzone
  o rząd wielkości. **Nie przywracaj `-ffast-math`** — unieważnia strażniki
  `total_energy < 1e-14` i detekcję NaN w jądrach.
- **M3** — `tdff_net/tucker_als.py::TuckerALSSolver.fit` rzuca `NotImplementedError`
  dla D ≠ 2 zamiast cicho zwracać MSE losowej inicjalizacji
  (`tests/test_tucker_als.py`).
- **A1** — deklaracje bez pokrycia usunięte z `src/facade.py` i pozostałych modułów
  `src/` (patrz §2).

### Naprawione w Sprint 1 (przenośność i CI)

- **V4** — `.github/workflows/ci.yml`. Matryca `{ubuntu, macos, windows} ×
  {py3.11, py3.12}` plus joby: `native-required` (wymaga rozszerzenia),
  `test-no-extension` (czysty klon bez rozszerzenia), `sanitizers` (ASan+UBSan,
  Debug, Linux), `lint` (ruff + pyright), `benchmark` (artefakt, bez bramki).
- **C5** — `#pragma loop(ivdep)` (składnia wyłącznie MSVC) zastąpiona makrami
  `HS_OMP` / `HS_OMP_SIMD` / `HS_OMP_SIMD_REDUCE` w `fast_kan_kernel.hpp`.
  Makra rozwijają się do pustego ciągu, gdy kompilator nie zgłasza odpowiedniego
  poziomu OpenMP (`_OPENMP >= 201307` dla `simd`), więc build bez OpenMP
  (AppleClang bez libomp) jest czysty pod `-Wunknown-pragmas`. Target
  `_cpp_kernels` kompiluje się z `-Wall -Wextra -Wunknown-pragmas -Werror`
  (MSVC: `/W4 /WX`), sterowane opcją `HSKAN_WERROR` (domyślnie ON).
- **C6** — `-mavx2 -mfma` / `/arch:AVX2` już nie są bezwarunkowe. Domyślny build
  to baseline x86-64 (`-mno-avx2 -mno-fma`, jawnie, żeby było widać w logu);
  AVX2 za opcją `HSKAN_ENABLE_AVX2=OFF`. Job `test` na Linuksie weryfikuje
  `objdump -d ... | grep -c '%ymm' == 0`. **Nie włączaj `HSKAN_ENABLE_AVX2` w
  wheelu publicznym** — nie ma dyspozycji runtime (`__builtin_cpu_supports`),
  więc byłby to SIGILL przy imporcie na CPU bez AVX2.
- **C2** — wszystkie iloczyny rozmiarów w `fast_kan_kernel.cpp` i
  `fast_kan_bindings.cpp` liczone w `std::size_t`; walidacja `P * P` przez
  dzielenie (`P > A_out.size() / P`), odrzucanie niedodatnich wymiarów.
  Test regresji: `tests/test_cpp_size_guards.py`.
- **V2** — `tests/_native.py::requires_native` (skipif) + zmienna
  `HSKAN_REQUIRE_NATIVE=1`, która zamienia skip w twardy błąd; progi
  wydajnościowe zależne od sprzętu usunięte z `tests/` (patrz niżej).

### Do rozstrzygnięcia (otwarte po Sprint 1)

- **Sprzeczność progu opóźnienia.** Usunięty próg testowy mówił
  `latency_us <= 0.67`, a docstring `src/cpp_kernels/cpp_kan_engine.py:24`
  deklaruje `< 0.2 us / pkt` — te dwie liczby nie mogą być jednocześnie
  właściwym opisem tego samego kodu. Po usunięciu testu wydajnościowego
  deklaracja z docstringa **nie ma żadnego pokrycia** (§2). Pomiar baseline'u
  bez AVX2 na Ryzen 7 5700X: 100 000 punktów TT-KAN (D=10, rank=8, degree=5)
  w medianie 24.3 ms, czyli ok. 0.24 us/pkt jednowątkowo-równolegle na 16
  wątkach — bliżej 0.2 niż 0.67, ale w innym reżimie niż oba progi.
  Decyzja (zawęzić deklarację do zmierzonej konfiguracji albo ją usunąć) jest
  świadomie odłożona; nie rozstrzygaj jej przy okazji innego zadania.
- **Bramka lintera jest wąska.** `[tool.ruff.lint] select` to dziś tylko
  `E9, F63, F7, F82` (0 trafień). Pełny zestaw `E,F,W,I,B,UP,NPY,SIM,RUF` daje
  **1660 trafień w `src/`**, w tym **66 × NPY002** (`np.random.seed` w kodzie
  biblioteki — audyt A4), 691 × W293, 445 × E501, 141 × UP006. Job `lint`
  raportuje te liczby jako wyjście nieblokujące. Rozszerzanie `select` to osobna
  zmiana — masowy autofix w jednym commicie z pracą nad C++ czyni oba
  nierecenzowalnymi.
- **`pyright --strict` nie jest bramką.** Tryb podstawowy: **199 błędów** w
  `src/` (najliczniejsze: 46 × `reportPossiblyUnbound`, 37 ×
  `reportOptionalMemberAccess`, 24 × `reportInvalidTypeForm`). Job `lint`
  uruchamia `pyright` i `pyright -p pyrightconfig.strict.json` jako wyjście
  nieblokujące.
- **Bramka regresji benchmarków działa tylko lokalnie.** Kryterium „regresja
  mediany > 20% względem zapisanego baseline'u" wymaga tej samej maszyny;
  runnery GitHub-hosted są współdzielone i ich rozrzut przekracza 20%. W CI
  benchmarki są artefaktem, nie bramką (`benchmarks/README.md`).
- **C4 (kolejność pętli) celowo nietknięta** w Sprint 1 — zmiana wydajnościowa
  zmieszana z przenośnościową uczyniłaby obie nieweryfikowalnymi.

---

## 5. Praktyki pracy w tym repozytorium

- **Test-first przy naprawie błędu.** Najpierw test, który **pada** na obecnym
  kodzie i dowodzi błędu; dopiero potem naprawa. To jest materiał portfolio
  (`docs/BUG_HUNT.md` w planie) — dowód, że bramka działa.
- **Rebuild C++ po zmianie flag.** `CMakeLists.txt` jest konsumowany przez
  `scikit-build-core`; po zmianie flag uruchom
  `python -m pip install -e . --no-build-isolation --no-deps --force-reinstall`,
  a następnie zweryfikuj flagi w wygenerowanym projekcie (`build/_cpp_kernels.vcxproj`
  na MSVC), zanim uwierzysz wynikom testów.
- **`build/` zawiera wyłącznie artefakty MSVC.** Ścieżka GCC/Clang nigdy nie została
  zbudowana — spodziewaj się błędów kompilacji przy pierwszym uruchomieniu na Linuksie
  (m.in. `#pragma loop(ivdep)` to składnia wyłącznie MSVC).
- **RNG.** `np.random.seed(...)` w bibliotece resetuje RNG całego procesu (audyt A4).
  Nowy kod w `src/` (poza `src/tasks/` i testami) używa `np.random.default_rng(seed)`.
- **Testy natywne.** Nowy test dotykający `_cpp_kernels` dostaje dekorator
  `@requires_native` z `tests/_native.py` (albo `pytestmark = requires_native`).
  Bez niego czysty klon bez zbudowanego rozszerzenia znowu przestanie przechodzić
  testy (audyt V2).
- **Brak progów wydajnościowych w `tests/`.** Pomiary czasu należą do
  `benchmarks/` (pytest-benchmark, porównanie do baseline'u), nie do testów
  jednostkowych — próg absolutny mierzy sprzęt, nie kod.
- **Nie commituj bez zgody użytkownika.** Pokaż diff i zaproponuj wiadomość commita.
