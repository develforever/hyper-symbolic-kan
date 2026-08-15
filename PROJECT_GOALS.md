# Cel Projektu & Kryteria Sukcesu Technicznego

## 1. Cele Główne (Core Objectives)

1. **Wydajność Obliczeniowa**:
   - Eliminacja propagacji wstecznej (Backpropagation) w modułach wnioskowania logicznego.
   - Skrócenie czasu trenowania warstw o min. **2 rzędy wielkości (100x)** w stosunku do MLP/Transformerów o porównywalnej liczbie parametrów.

2. **Precyzja Dedykowana (100% Accuracy on Symbolic Rules)**:
   - Zero halucynacji w zadaniach zawierających twarde relacje przechodnie.
   - Zachowanie determinizmu dla zapytań logicznych przy jednoczesnym zachowaniu reprezentacji ciągłej.

3. **Nowe Paradygmaty Architektoniczne**:
   - Etap I: Implementacja i weryfikacja **HS-CKAN** na wnioskowaniu tekstowym.
   - Etap II: Implementacja **TDFF-Net** dla ciągłych pól funkcjonalnych.
   - Etap III: Integracja **MCT-NSE** z filtrami monadycznymi.

---

## 2. Metryki Sukcesu dla Zadania 1 (Compositional Reasoning)

| Metryka | Klasyczne Transformer / LLM | Dążenie HS-CKAN |
| :--- | :--- | :--- |
| **Błąd Dedukcji (Zł złożoność = 10)** | > 15-30% błędów | **0.0% (Dokładność 100%)** |
| **Liczba epok trenowania** | 100 - 1000 epok | **0 epok (Rozwiązanie analityczne)** |
| **Zużycie Pamięci w czasie uczenia** | $O(N \cdot B \cdot L)$ (Zapis aktywacji) | $O(D^2)$ (Rozmiar macierzy kowariancji) |
| **Odporność na katastrofalne zapominanie** | Podatna (wymaga LoRA/Replay) | Odporna (Algebraiczne dopisywanie podprzestrzeni) |
