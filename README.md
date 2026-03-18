# ML — Eksploracja Rejestru Cen Nieruchomości w mieście Warszawa 🧠📊

To repozytorium zawiera analizę cen nieruchomości w mieście Warszawa w okresie 2015 - 2025

---

# 📁 Zawartość

-  **inflacja.ipynb** — eksploracja danych inflacyjnych
-  **rcn.ipynb** — esklporacja danych sprzedażowych
-  **inflacja.csv** — dane inflacyjne
-  **inflacja_prepared.csv** — przekształcone dane inflacyjne
-  **sales_random.csv** — dane sprzedażowe

**Wyniki (RMSE / R²):**
| Model             | Train RMSE | Train R² | Test RMSE | Test R² |
|-------------------|-----------|----------|-----------|---------|
| Linear Regression | 57 349    | 0.9390   | 51 666    | 0.9435  |
| Random Forest     | 6 204     | 0.9993   | 12 961    | 0.9964  |
| XGBoost           | 2 075     | 0.9999   | 16 199    | 0.9944  |

Random Forest i XGBoost wykazują oznaki przeuczenia — widoczne głównie w RMSE, 
gdzie błąd na danych testowych jest kilkukrotnie wyższy niż na treningowych 
(XGBoost: Train RMSE 2k vs Test RMSE 16k, Random Forest: 6k vs 13k). W dalszym etapie
próba regularyzacji
