---
name: hyper-symbolic-kan-development
description: Rules, mathematical paradigms, and development workflows for the Hyper-Symbolic Clifford-KAN (HS-CKAN), Tensor-Decomposed Functional Fields (TDFF-Net), and Monadic Category-Theoretic Neuro-Symbolic Engine (MCT-NSE) codebase.
---

# Hyper-Symbolic & Algebraic Neural Architectures (HS-ANA) Skill

## Overview
Ta umiejętność (Skill) definiuje żelazne zasady architektoniczne, wzorce projektowe oraz procedury weryfikacyjne przy rozwijaniu repozytorium `hyper_symbolic_kan`.

---

## 1. Filozofia i Zasady Projektowe (Core Principles)

1. **Zakaz Wykorzystywania Standardowej Propagacji Wstecznej (No Traditional Backprop)**:
   - Wszystkie moduły wnioskowania logicznego i dopasowania relacji muszą korzystać z algebry geometrycznej w zamkniętej formie ($O(1)$) lub analitycznych solverów regresji macierzowej (SVD / QR / Ridge).

2. **Gwarancja Precyzji (Zero-Error Invariance)**:
   - Moduły dedukcji przechodniej na predykatach i relacjach muszą zachowywać 100.00% dokładności (zero błędów dedukcji / zero halucynacji).
   - Przechodniość relacji $R(A, B) \circ R(B, C) = R(A, C)$ musi być realizowana przez kontrakcję geometryczną Algebry Clifforda:
     $$(e_i e_k) \cdot (e_k e_j) = e_i (e_k^2) e_j = e_i e_j$$

3. **Czystość Funkcjonalna i Determinizm**:
   - Stan sieci i uaktualnienia muszą zachowywać niezmienniczość monadyczną (Zero Side-Effects).

---

## 2. Mapa Repozytorium i Odpowiedzialności Plików

```
C:\Users\robert\code\hyper_symbolic_kan\
├── README.md                          # Dokumentacja architektoniczna projektu
├── PROJECT_GOALS.md                   # Wskaźniki KPI, metryki wydajności i roadmapa
├── pyproject.toml / requirements.txt   # Deklaracja środowiska i zależności
├── main.py                            # Główny runner i benchmark systemowy
├── .skills/
│   └── hyper_symbolic_kan_development/
│       └── SKILL.md                   # Niniejszy skrypt zasad i specyfikacji
└── src/
    ├── hs_ckan/
    │   ├── clifford_algebra.py        # Silnik Algebry Geometrycznej Cℓ_N (Kontrakcja iloczynu)
    │   ├── chebyshev_kan.py           # Wielomiany Czebyszewa T_k(x) na krawędziach KAN
    │   └── closed_form_solver.py      # Analityczny solver macierzowy (Zero epok backprop)
    ├── tdff_net/                      # [Planned] Niskorządowy rozkład tensorowy dla SDF
    ├── mct_nse/                       # [Planned] Monadyczny silnik kategoryczny z filtrem Boola
    └── tasks/
        └── compositional_reasoning.py # Benchmark rozumowania przechodniego na tekście
```

---

## 3. Workflow Weryfikacyjny (Verification Workflow)

Każda zmiana w kodzie lub nowa architektura **MUSI** zostać poddana natychmiastowej weryfikacji przed zatwierdzeniem:

1. **Uruchomienie Benchmarku**:
   ```bash
   python main.py
   ```
2. **Kryteria Akceptacji (Pass Criteria)**:
   - Czas uczenia warstwy: $< 5.0 \text{ ms}$ (Rozwiązanie analityczne).
   - Czas inferencji per zapytanie: $< 1.0 \text{ us}$.
   - Dokładność (Accuracy): **Dokładnie 100.00%**.

---

## 4. Dobre Praktyki Kodowania (Coding Standards)

- **Brak Pętli Uczenia**: Nie stosować optimizerów typu Adam/SGD dla podprzestrzeni algebraicznych.
- **Macierze Zawsze Ortogonalne**: Bazy wektorowe Algebry Clifforda muszą być generowane przy użyciu rozkładu QR dla zachowania czystej ortonormalności.
- **Odporność na Szum**: Przy wprowadzaniu szumu do wejścia (Continuous Noise) stosować transformację bazową wielomianami Czebyszewa w celu wygładzenia aproksymacyjnego przed projekcją SVD.
