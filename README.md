# Housing Prices: Categorical Encoding and Prediction Pipeline

**Author:** Bridget  
**Course:** DAS  

## Project Overview
This project focuses on predicting residential home sales prices . The primary goal was to explore, implement, and evaluate various **categorical encoding techniques** to see how different data transformation methods impact the predictive performance of a regularized linear model (`Ridge Regression`).

---

## Project Pipeline Structure

The notebook is divided into clear machine learning steps:

*   **Step One: Library Imports & Data Loading** — Initializing core data analysis tools (`pandas`, `numpy`) and importing the training and test sets.
*   **Step Two: Target Variable Distribution Check** — Visualizing the distribution of `SalePrice` to analyze target skewness and potential data anomalies.
*   **Step Three: Automated Data Processing** — Separating the data into features (`X`) and targets (`y`), identifying variable types, and filling missing values for both categorical and numerical columns.
*   **Step Four: Validation Framework** — Creating an `evaluate_model` function utilizing an 80/20 train/validation split to benchmark predictive consistency via the $R^2$ metric.
*   **Step Five & Six: Feature Engineering & Encoding** — Benchmarking six unique categorical data transformation techniques.
*   **Step Seven: Inference & Test Predictions** — Compiling a performance summary table and exporting final test set predictions.

---

## Categorical Encoding Methods Evaluated

We systematically tested six different approaches to transform the text features into numeric values for our machine learning model:

1.  **Label Encoding:** Converting each unique category into a basic sequential integer.
2.  **One-Hot Encoding (OHE):** Creating separate binary (0 or 1) columns for every unique category value present.
3.  **Feature Hashing:** Using a dictionary-based hashing function to map high-cardinality categorical attributes into a fixed 1,000-dimensional space.
4.  **Frequency Encoding:** Mapping categories directly to their statistical frequency percentage within the training data.
5.  **Target Encoding:** Mapping categories to the mean value of the target variable (`SalePrice`).
6.  **K-Fold Target Encoding:** An advanced version of target encoding that uses out-of-fold averages to limit training data leakage.

---

## Performance Summary

The evaluation metric used to score the models is the **Validation $R^2$ Score** (Coefficient of Determination). The results across the models are as follows:

| Encoding Method | Validation $R^2$ Score |
| :--- | :---: |
| **Method 3: Feature Hashing** | **0.8863 (Best)** |
| Cyclic Encoding (OHE + Month Transformation) | 0.8848 |
| Method 5: Target Encoding | 0.8718 |
| Method 2: One-Hot Encoding | 0.8648 |
| Method 6: K-Fold Target Encoding | 0.8548 |
| Method 4: Dataset Statistic (Frequency) | 0.8447 |
| Method 1: Label Encoding Baseline | 0.8444 |

---

## Key Conclusions

*   **Feature Hashing provided the highest predictive accuracy (0.8863).** It successfully compressed complex text categories (like neighborhoods or exterior building materials) into a dense, manageable layout without creating an exploding number of columns that could confuse a linear model.
*   **Plain One-Hot Encoding created a minor bottleneck.** While it outperformed the simple baseline, creating hundreds of loose binary columns led to slight overfitting when dealing with highly skewed house prices and raw numerical features. 
*   **Combining Hashing with Ridge Regression was highly effective.** The model remained highly stable and resistant to scale disparities, proving to be the most robust approach for generating our final test set submission.

---

## Repository Deliverables
*   `Houseprices.ipynb` — The primary workspace notebook containing all documented steps and code blocks.
*   `my_submission.csv` — The final formatted output file mapping home IDs to predicted sales prices, exported cleanly without row index counts.
