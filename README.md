# E-Commerce Stream Mining: Next-Category & Purchase Intent Prediction

This repository contains the implementation of a streaming machine learning project based on the **RetailRocket e-commerce dataset**. The project evaluates adaptive streaming models and baseline heuristics in an online, prequential (test-then-train) setting using the `river` library. 

A key focus of this project is understanding **concept drift** and handling realistic deployment constraints (e.g., hiding item identifiers at prediction time to prevent data leakage).

## Project Objectives
The analysis focuses on two main streaming tasks:
1. **Task A (Next Parent Category Prediction):** Predicting the product category a user will interact with next (283 classes).
2. **Task B (Purchase Intent Detection):** Predicting whether the current browsing event will result in a transaction (highly imbalanced binary classification).

## Methodology & Feature Engineering
* **Point-in-Time Merging:** Item properties (category, price) were merged using backward and forward `asof` joins to ensure no future information was leaked into the training stream.
* **Streaming Feature Extraction:** Engineered features include user session trackers (cart-to-view ratios, time gaps), pattern miners (recent category diversity), transition matrices (first-order Markov models), and bipartite graph interactions.
* **Anomaly Detection:** Integration of Half-Space Trees (HST) to score session anomalies on the fly.
* **Cold-Start Policy:** A strict global trending fallback was implemented for users with no prior history (affecting ~51.3% of events) to ensure fair model evaluation.

## Models Evaluated
* **Baselines:** Last-seen category, Global trending, Transition Matrix, Cart-presence heuristic.
* **Machine Learning:** Adaptive Random Forest (ARF Big & Lean), Hoeffding Adaptive Tree (HAT).
* **Drift Testing:** Models were stress-tested against 5 custom concept drift scenarios (repeat-row, category-shuffle, distribution-shift, intent-suppression, intent-boost).

## Key Findings
1. **Heuristics vs. ML in Session Data:** For next-category prediction, simple session autocorrelation (the "Last seen" baseline) significantly outperformed tree-based models, achieving ~44% Top-1 Accuracy.
2. **Value of ML in Intent Prediction:** While a simple cart-presence heuristic achieved the highest ROC-AUC (0.93), the oversampled Adaptive Random Forest provided a better F1 score (0.19 vs 0.16) by selectively identifying *when* a cart-positive user would actually transact.
3. **Drift Resilience:** Model robustness is highly task-specific. ML models adapted better to sudden conversion surges (intent-boost), whereas baselines were robust to consistent label permutations.

## Repository Structure
* `data/` - Contains the RetailRocket dataset subset used for this analysis.
* `01_data_preprocessing.ipynb` - Notebook for point-in-time data merging and stream preparation.
* `02_stream_pipeline.ipynb` - Prequential evaluation of Task A and Task B models.
* `03_stream_drift_test.ipynb` - Implementation and testing of the 5 concept drift scenarios.
* `eval_plots/` - Visualizations of standard model performance and metrics.
* `drift_plots/` - Visualizations of model performance before and after drift injections.
* `report.pdf` - Detailed academic report on the methodology and findings.

## How to Run
This project is built using Jupyter Notebooks. To replicate the findings:
1. Ensure you have the required libraries installed: `pip install pandas numpy river matplotlib imbalanced-learn jupyter`
2. Run `01_data_preprocessing.ipynb` to generate the streaming tables from the raw data.
3. Execute `02_stream_pipeline.ipynb` for the main model evaluation.
4. Run `03_stream_drift_test.ipynb` to visualize how models react to synthetic concept drift.