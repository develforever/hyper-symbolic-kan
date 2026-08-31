# Benchmarki jądra natywnego

Zakres: pomiar przepustowości natywnych jąder TT-KAN / CP-KAN.
**To nie są testy** — `pytest tests/` ich nie zbiera.

## Dlaczego nie próg absolutny

Audyt V2: `tests/test_cpp_kernels.py` zawierał

```python
assert throughput >= 1_500_000     # punktów/s
assert latency_us <= 0.67          # us/punkt
```

Obie liczby opisują jedną maszynę, nie kod: na wolniejszym CPU test pada bez
regresji, na szybszym przechodzi mimo regresji. Zostały usunięte.

Kryterium przyjęte w zamian: **regresja mediany > 20% względem baseline'u
zapisanego na tej samej maszynie**. Porównanie mediany dwóch przebiegów na tym
samym sprzęcie mierzy zmianę w kodzie; próg absolutny mierzy sprzęt.

## Zapisany baseline

`benchmarks/baselines/Windows-CPython-3.12-64bit/0001_baseline.json`

Plik JSON pytest-benchmark zawiera pełny `machine_info` (marka CPU, liczba
rdzeni, wersja Pythona, kompilator). Baseline zapisany na:

| | |
|---|---|
| CPU | AMD Ryzen 7 5700X (8C/16T, 3.4 GHz) |
| OS | Windows 11, 64-bit |
| Python | 3.12.3, MSC v.1938 |
| Kompilator rozszerzenia | MSVC 19.51, `/O2 /fp:precise /fp:contract /openmp`, **bez** `/arch:AVX2` (baseline x86-64) |
| Konfiguracja CMake | `HSKAN_ENABLE_AVX2=OFF`, `HSKAN_WERROR=ON` |

## Uruchomienie

Zapis nowego baseline'u (raz na maszynę):

```
pytest benchmarks/ --benchmark-storage=benchmarks/baselines \
                   --benchmark-save=baseline --benchmark-min-rounds=20
```

Bramka regresji względem zapisanego baseline'u:

```
pytest benchmarks/ --benchmark-storage=benchmarks/baselines \
                   --benchmark-compare=0001 \
                   --benchmark-compare-fail=median:20%
```

## Zakres gwarancji

Bramka 20% jest wiarygodna **tylko** przy porównaniu przebiegów na tej samej
maszynie. Katalog baseline'u jest wybierany przez pytest-benchmark na podstawie
`system-implementacja-wersja-bity`, więc baseline z Windows/CPython 3.12 nie
zostanie użyty do porównania na Linuksie — takie porównanie nie ma sensu i
narzędzie go nie wykona.

Job `benchmark` w `.github/workflows/ci.yml` uruchamia te same benchmarki na
runnerze GitHuba i **publikuje wynik jako artefakt, bez bramki**: runnery
GitHub-hosted są współdzielone, ich rozrzut czasów przekracza 20%, więc bramka
na nich sygnalizowałaby regresje, których nie ma. Egzekwowalna bramka wymaga
runnera dedykowanego — dopóki go nie ma, w CI mamy pomiar, a nie gwarancję.
