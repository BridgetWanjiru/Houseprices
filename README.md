# Housing Prices: Categorical Encoding and Prediction Pipeline

**Author:** Bridget
**Course:** DAS

## Project Overview
This project focuses on predicting residential home sale prices. The goal 
was to encode the dataset's categorical features by reasoning about what 
each variable represents rather than applying one method uniformly or 
choosing whichever encoder produced the best score and evaluate the 
result using a regularized linear model (Ridge Regression).

## Project Pipeline Structure
The notebook follows these steps:

- **Step One: Library Imports & Data Loading** — Initializing core data 
  analysis tools (pandas, numpy) and importing the training and test sets.
- **Step Two: Target Variable Distribution Check** — Visualizing the 
  distribution of `SalePrice` to check for skew and anomalies.
- **Step Three: Fix `MSSubClass`'s Data Type** — Converting `MSSubClass` 
  from integer to string, since its values are building-class codes, not 
  quantities.
- **Step Four: Handle Missing Values** — Separating columns where NaN means 
  "feature doesn't exist" (filled with `'None'` as a real category) from 
  columns where NaN means genuinely missing data (imputed with median/mode).
- **Step Five: Validation Framework** — An `evaluate_model` function using 
  an 80/20 train/validation split to score predictions via R².
- **Step Six: Ordinal Encoding** — Manually ranked categorical columns 
  (e.g. quality ratings from Poor to Excellent) encoded in their true order.
- **Step Seven: One-Hot Encoding** — Categorical columns with no natural 
  order (e.g. Neighborhood, SaleType) encoded as separate binary columns.
- **Step Eight: Validate the Full Encoding** — Evaluating the combined 
  ordinal + one-hot feature set.
- **Step Nine: Generate Final Test Predictions** — Fitting the final Ridge 
  model and exporting the submission file.

## Encoding Approach
Each column was encoded according to what it actually represents:

- **Ordinal encoding** for columns with a genuine rank (quality ratings, 
  basement finish type, functional deductions, etc.), so the model can use 
  the order between categories. Each order was defined manually from the 
  data dictionary,not assigned automatically or alphabetically.
- **One-hot encoding** for columns with no natural order (Neighborhood, 
  MSZoning, Foundation, SaleType, MSSubClass, etc.), so no false ranking is 
  introduced.

## Performance Summary
| Step | Validation R² Score |
|---|---|
| Ordinal encoding only (quality columns + numeric features) | 0.8393 |
| Full pipeline: ordinal + one-hot combined | 0.8778 |

## Key Conclusions
- **`MSSubClass`** is stored as a number but represents categorical building 
  codes. Left unconverted, a linear model would wrongly treat larger codes 
  as "more" of something — correcting its type was necessary for the 
  encoding to be valid, independent of its effect on the score.
- **NaN doesn't mean the same thing in every column.** Columns like 
  `PoolQC` or `GarageType` use NaN to indicate the feature is absent; 
  columns like `LotFrontage` or `Electrical` use NaN to indicate a value 
  wasn't recorded. Each required different handling.
- **Matching encoding to variable meaning**, rather than sweeping several 
  encoders and reporting the highest score, was the actual goal of this 
  assignment — the R² score here is a validation check, not the basis for 
  selecting the method.

## Repository Deliverables
- `Houseprices.ipynb` — The primary workspace notebook containing all 
  documented steps and code blocks.
- `my_submission.csv` — The final formatted output file mapping home IDs to 
  predicted sale prices, exported cleanly without row index counts.
