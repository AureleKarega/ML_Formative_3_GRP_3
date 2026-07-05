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

This section demonstrates the manual implementation of the **Gradient Descent** optimization algorithm for a multiple linear regression model. The objective was to iteratively update the model parameters (**m** and **b**) using the gradients of the **Mean Squared Error (MSE)** cost function. All calculations were performed manually using **matrix multiplication**, without relying on machine learning libraries.

---

## Problem Statement

The regression model is defined as:

\[
y = m_1x_1 + m_2x_2 + b
\]

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

- Computed the predicted values (ŷ) using the initial parameters.
- Calculated the Mean Squared Error (MSE).
- Derived the gradients of the cost function with respect to **m** and **b**.
- Updated the parameters using the Gradient Descent update rule.
- Repeated the process for **four iterations**, recording all intermediate calculations after every update.

---

## Mathematical Foundation

### Prediction

\[
\hat{y} = Xm + b
\]

where:

- **X** is the feature matrix
- **m** is the weight vector
- **b** is the bias vector

---

### Mean Squared Error (MSE)

\[
J(m,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

The MSE measures how far the predicted values are from the actual target values.

---

### Gradient Descent Update Rule

The parameters are updated after each iteration using:

\[
m = m - \alpha \frac{\partial J}{\partial m}
\]

\[
b = b - \alpha \frac{\partial J}{\partial b}
\]

where:

- **α** is the learning rate
- **∂J/∂m** is the gradient with respect to the weights
- **∂J/∂b** is the gradient with respect to the bias

---

## Results

Four manual Gradient Descent iterations were successfully completed. During each iteration:

- Predictions were computed.
- Errors between predicted and actual values were determined.
- Gradients were calculated using matrix operations.
- The parameters **m** and **b** were updated using the Gradient Descent formula.
- The new cost value was evaluated after every update.

The calculations showed a consistent reduction in the cost function across the iterations, indicating that the model parameters were moving toward values that better fit the training data.

---

## Observation

The behaviour of the algorithm demonstrated the expected characteristics of Gradient Descent:

- The values of **m** and **b** gradually changed after every iteration.
- Prediction errors became smaller as the parameters were updated.
- The Mean Squared Error continuously decreased, showing that each update improved the model.
- The optimization process moved steadily toward convergence.

This confirms that Gradient Descent effectively minimizes prediction error by adjusting the model parameters in the direction of the negative gradient.

---

## File Structure

```text
ML Formative 3/
├── Part3_Gradient_Descent.pdf
├── Part3_Gradient_Descent.ipynb
├── Manual Calculations.pdf
└── README_Part3.md
```

---

## Implementation Details

The implementation included:

- Manual matrix multiplication for predictions
- Manual derivation of gradients
- Manual parameter updates
- Four complete Gradient Descent iterations
- Step-by-step calculations with intermediate results

No machine learning libraries were used during the calculations.

---

## Conclusion

The manual implementation provided a clear understanding of how Gradient Descent optimizes a multiple linear regression model. By repeatedly computing gradients and updating the parameters, the algorithm progressively reduced the prediction error and minimized the Mean Squared Error. The results confirmed that Gradient Descent is an effective optimization technique for training linear regression models.

---


# Part 4 — Gradient Descent in Code

What this is

This converts the Part 3 manual gradient descent calculations into working Python code, using SciPy and Matplotlib as required.

Files


Part4_Gradient_Descent.ipynb — Jupyter notebook (recommended). Already run once, so outputs and plots show immediately on open. Can also be re-run live during the presentation.
part4_gradient_descent.py — same code as a plain Python script.


How to run

Option A — Jupyter / Google Colab (recommended)


Go to https://colab.research.google.com
File → Upload notebook → select Part4_Gradient_Descent.ipynb
Run cells top to bottom (Shift + Enter on each), or just view the saved outputs.


Option B — Local Python

pip install numpy scipy matplotlib
python part4_gradient_descent.py

What the code does


Starts from the same data, m, b, and learning rate (0.01) used in Part 3.
Defines the Mean Squared Error cost function.
Uses scipy.optimize.approx_fprime to compute the derivative of that cost function with respect to m and b — the SciPy requirement for this part.
Also computes the same gradient analytically (by hand-derived formula) and prints both side by side to show they match.
Updates m and b using gradient descent, once per iteration, in matrix form — every intermediate value (y_hat, error, MSE, gradient) is printed, nothing is hidden inside one function call.
Runs 4 iterations — one per group member.
Computes final predictions using the final m and b.
Plots:

m and b values across iterations
MSE (error) across iterations





Result summary

IterationmbMSE0 (init)[-1.00, 2.00][1.00, 1.00]61.001[-1.45, 0.87][0.99, 0.89]6.502[-1.33, 1.18][1.02, 0.91]2.503[-1.37, 1.10][1.04, 0.90]2.164[-1.36, 1.12][1.06, 0.89]2.10

Error drops sharply after the first iteration, then keeps decreasing but flattens out. Both m and b move steadily in the direction that reduces error, confirming gradient descent is working — it slows down as it approaches a minimum, which is expected with a fixed learning rate.

Contribution

Part 4 (this notebook/script) completed by: Loice Teta


## Contribution

**Gakwaya Ineza Ketia** — Responsible for Part 3:

- Manual calculation of predicted values
- Derivation of MSE gradients
- Manual Gradient Descent parameter updates
- Completion of four Gradient Descent iterations
- Analysis of parameter convergence and cost reduction
- Documentation and README preparation
