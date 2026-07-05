# Part 2: Bayesian Probability — Sentiment Analysis
### ML Formative 3 | Assigned to: Karega Aurele

---

## Overview

This section implements **Bayes' Theorem** from scratch in Python to determine the probability that an IMDB movie review is positive, given that a specific keyword appears in it. No external machine learning libraries were used — only Python's built-in `csv` and `re` modules.

---

## Dataset

**Source:** IMDB Movie Reviews Dataset  
**File:** `IMDB Dataset.csv`  
**Size:** 50,000 reviews  
**Structure:**

| Column | Description |
|--------|-------------|
| `review` | Full text of the movie review |
| `sentiment` | Label — either `positive` or `negative` |

The dataset is perfectly balanced:
- **25,000 positive** reviews (50%)
- **25,000 negative** reviews (50%)

This balance is important because it means our **prior probability P(Positive) = 0.5** — before seeing any keyword, there is a 50/50 chance a review is positive.

---

## The Mathematical Foundation

### Bayes' Theorem

$$P(\text{Positive} \mid \text{keyword}) = \frac{P(\text{keyword} \mid \text{Positive}) \times P(\text{Positive})}{P(\text{keyword})}$$

Each term has a specific meaning in this context:

### Prior — P(Positive)
The baseline probability that any review is positive, before we look at any words. Since the dataset is balanced, this is always **0.5 (50%)**.

> *"Before I read anything, there's a 50% chance this review is positive."*

### Likelihood — P(keyword | Positive)
The probability that the keyword appears in a review, **given that the review is positive**. Computed as:

$$P(\text{keyword} \mid \text{Positive}) = \frac{\text{Number of positive reviews containing the keyword}}{\text{Total positive reviews}}$$

A high likelihood means the keyword is commonly used in positive reviews. A low likelihood means it rarely appears there.

### Marginal — P(keyword)
The overall probability that the keyword appears in **any** review, regardless of sentiment. Computed as:

$$P(\text{keyword}) = \frac{\text{Total reviews containing the keyword}}{\text{Total reviews}}$$

This acts as a normalising factor. If a keyword is very common across all reviews, it carries less information. If it's rare, it carries more weight.

### Posterior — P(Positive | keyword)
The updated probability that a review is positive **after** observing the keyword. This is what Bayes' Theorem computes — it combines the prior belief with the strength and rarity of the keyword clue.

---

## Keyword Selection

We chose **4 positive keywords** and **4 negative keywords** based on intuition about movie review language:

| Group | Keywords |
|-------|----------|
| ✅ Positive | `excellent`, `masterpiece`, `brilliant`, `heartwarming` |
| ❌ Negative | `terrible`, `awful`, `boring`, `waste` |

**Chosen direction:** P(Positive | keyword) — we compute the probability of positivity given each keyword.

---

## Results

| Keyword | Prior P(+) | Likelihood P(kw\|+) | Marginal P(kw) | Posterior P(+\|kw) |
|---------|-----------|---------------------|----------------|---------------------|
| excellent | 0.5000 | 0.1147 | 0.0710 | **0.8074 (80.7%)** |
| masterpiece | 0.5000 | 0.0351 | 0.0241 | **0.7274 (72.7%)** |
| brilliant | 0.5000 | 0.0635 | 0.0418 | **0.7601 (76.0%)** |
| heartwarming | 0.5000 | 0.0046 | 0.0027 | **0.8382 (83.8%)** |
| terrible | 0.5000 | 0.0153 | 0.0540 | **0.1418 (14.2%)** |
| awful | 0.5000 | 0.0114 | 0.0577 | **0.0985 (9.9%)** |
| boring | 0.5000 | 0.0236 | 0.0610 | **0.1937 (19.4%)** |
| waste | 0.5000 | 0.0070 | 0.0507 | **0.0691 (6.9%)** |

---

## Worked Example — "excellent"

Starting with the raw counts from the dataset:

- Total reviews: **50,000**
- Positive reviews containing "excellent": **2,868**
- Total reviews containing "excellent": **3,552**

Applying the formula step by step:

```
Prior      = 25,000 / 50,000        = 0.5000
Likelihood = 2,868  / 25,000        = 0.1147
Marginal   = 3,552  / 50,000        = 0.0710

Posterior  = (0.1147 × 0.5000) / 0.0710
           = 0.0574 / 0.0710
           = 0.8074  →  80.7%
```

**Interpretation:** Seeing the word "excellent" raises our confidence that the review is positive from 50% all the way to 80.7%. The word appears roughly 4× more often in positive reviews than negative ones, making it a strong positive signal.

---

## Key Insight — Why Keywords Move the Probability

The posterior shifts away from 50% based on two factors:

1. **How imbalanced the keyword is** between positive and negative reviews — the bigger the gap, the bigger the shift
2. **How rare the keyword is overall** — a rare but lopsided keyword is more informative than a common one

This is why "heartwarming" (83.8%) outperforms "excellent" (80.7%) despite being less common — it appears almost exclusively in positive reviews, making it an extremely strong signal when it does appear.

---

## Implementation Details

### File Structure
```
ML Formative 3/
├── IMDB Dataset.csv              # Raw dataset (50,000 reviews)
├── bayesian_sentiment.py         # Standalone Python script
├── Part2_Bayesian_Sentiment.ipynb # Jupyter/Colab notebook
└── README_Part2.md               # This file
```

### How to Run

**Option 1 — Python script (VS Code):**
```bash
python bayesian_sentiment.py
```
Make sure `IMDB Dataset.csv` is in the same folder.

**Option 2 — Jupyter Notebook (Google Colab):**
1. Upload `Part2_Bayesian_Sentiment.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Upload `IMDB Dataset.csv` via the folder icon in the left sidebar
3. Change the dataset path in Cell 1 to `/content/IMDB Dataset.csv`
4. Run all cells top to bottom

### Dependencies
```
csv    — built-in Python module (no install needed)
re     — built-in Python module (no install needed)
matplotlib — for visualisation only (pre-installed in Colab)
```
No external machine learning libraries were used.

---

## Interpretation Summary

| Keyword | Posterior | Signal Strength |
|---------|-----------|-----------------|
| heartwarming | 83.8% | Strongly positive |
| excellent | 80.7% | Strongly positive |
| brilliant | 76.0% | Strongly positive |
| masterpiece | 72.7% | Strongly positive |
| boring | 19.4% | Strongly negative |
| terrible | 14.2% | Strongly negative |
| awful | 9.9% | Strongly negative |
| waste | 6.9% | Strongly negative |

Every keyword behaved exactly as expected — positive keywords pushed the posterior well above 50%, and negative keywords pulled it well below. This validates Bayes' Theorem as an effective, interpretable tool for keyword-based sentiment classification without any machine learning.

---

## Contribution

**Karega Aurele** — Responsible for Part 2:
- Keyword selection and justification
- Python implementation of Bayes' Theorem
- Computation of all probabilities
- Jupyter notebook and visualisation
- This README


# Part 3: Gradient Descent Manual Calculation
### ML Formative 3 | Assigned to: Gakwaya Ineza Ketia

---

## Overview

This section demonstrates the manual implementation of the **Gradient Descent** optimization algorithm for a multiple linear regression model. The objective was to iteratively update the model parameters (**m** and **b**) by minimizing the **Mean Squared Error (MSE)** cost function. All calculations were performed manually using **matrix multiplication**, following the instructions provided in the assignment.

---

## Problem Statement

The linear regression model is defined as:

$begin:math:display$
y \= m\_1x\_1 \+ m\_2x\_2 \+ b
$end:math:display$

### Initial Parameters

- **m = [-1, 2]**
- **b = [1, 1]**

### Dataset

| x₁ | x₂ | y |
|----|----|---|
| 1 | 3 | 5 |
| 4 | 10 | 6 |

---

## Objective

The following tasks were completed:

- Computed the predicted values (ŷ) using the initial values of **m** and **b**.
- Derived the gradients of the Mean Squared Error (MSE) cost function with respect to **m** and **b**.
- Performed four manual Gradient Descent iterations using matrix multiplication.
- Updated the model parameters after each iteration.
- Recorded all intermediate calculations, including predictions, errors, gradients, updated parameters, and cost values.

---

## Mathematical Foundation

### Prediction

$begin:math:display$
\\hat\{y\}\=Xm\+b
$end:math:display$

### Mean Squared Error (MSE)

$begin:math:display$
J\(m\,b\)\=\\frac\{1\}\{n\}\\sum\_\{i\=1\}\^\{n\}\(y\_i\-\\hat\{y\}\_i\)\^2
$end:math:display$

### Gradient Descent Update Rule

$begin:math:display$
m\=m\-\\alpha\\frac\{\\partial J\}\{\\partial m\}
$end:math:display$

$begin:math:display$
b\=b\-\\alpha\\frac\{\\partial J\}\{\\partial b\}
$end:math:display$

where **α** represents the learning rate.

---

## Results

Four Gradient Descent iterations were completed successfully. During each iteration, the predictions, errors, gradients, and updated parameter values were calculated manually. After every update, the Mean Squared Error decreased, showing that the model was gradually learning and moving toward the optimal parameter values.

---

## Observation

Throughout the iterations, both the weight vector (**m**) and bias (**b**) changed in the direction that minimized the error. The consistent decrease in the cost function demonstrated that Gradient Descent was converging correctly and improving the model's predictions with each iteration.

---

## Implementation Details

### Files

```text
ML Formative 3/
├── Part3_Gradient_Descent.ipynb
├── Manual_Calculations.pdf
└── README_Part3.md
```

### Implementation Summary

- Manual matrix multiplication
- Manual MSE gradient derivation
- Four Gradient Descent iterations
- Parameter updates after every iteration
- Cost calculation after every update

No machine learning libraries were used.

---

## Conclusion

The manual implementation demonstrated how Gradient Descent optimizes a multiple linear regression model by repeatedly calculating gradients and updating the parameters. The reduction in the Mean Squared Error across all iterations confirmed that the algorithm was successfully converging toward a better-fitting model.

---

## Contribution

**Gakwaya Ineza Ketia** — Responsible for Part 3:

- Manual prediction calculations
- Gradient derivation
- Four Gradient Descent iterations
- Parameter updates
- Analysis of convergence
- README documentation
