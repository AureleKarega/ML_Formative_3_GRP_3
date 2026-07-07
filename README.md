# Expectation-Maximization (EM) Algorithm for Gaussian Mixture Models
### Probability Formative Assessment

## Overview

This project implements the **Expectation-Maximization (EM) algorithm from scratch in Python** to separate a mixed dataset of **father and child heights** into two Gaussian distributions **without using labels**.

The model assumes that every height belongs to one of two hidden groups:

- Distribution 1
- Distribution 2

Since the labels are unknown, EM repeatedly estimates which distribution each height most likely belongs to and updates the Gaussian parameters until the model converges.

---

# Project Structure

```
probability formative/
│
├── FORMATIVE_TEST.py      # EM algorithm implementation
├── GaltonFamilies.csv     # Height dataset
└── README.md              # Project documentation
```

---

# Dataset

Dataset used:

**GaltonFamilies.csv**

Columns used:

- `father`
- `childHeight`

The program combines both columns into one unlabeled dataset.

Example:

| Height |
|---------|
| 70 |
| 67 |
| 71 |
| 64 |
| ... |

The EM algorithm has **no knowledge** of which values belong to fathers or children.

---


### Problems

- The two distributions overlap.
- Many fathers are shorter than the global mean.
- Many children are taller than the global mean.
- Heights close to the mean would be misclassified.
- Every point is forced into a single group with complete certainty.

Example:

```
Children

58 60 61 63 64 66 68

Parents

65 66 68 69 71 73 75
```

A height of **66** could realistically belong to either group.

Splitting at the global mean would incorrectly force it into only one group.

---

# Why EM is Better

Instead of making a hard decision, EM makes a **soft assignment**.

Example

Height = 66

Instead of saying

```
100% Parent
```

EM may estimate

```
Child = 42%

Parent = 58%
```

These probabilities are called **posterior probabilities** (responsibilities).

This makes EM much more accurate when the distributions overlap.

---

# Gaussian Distribution

Each hidden group is modeled using a Gaussian (Normal) distribution.

The Gaussian probability density function is

\[
\frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\]

Where

- **μ** = mean
- **σ²** = variance
- **x** = observed height

---

# EM Algorithm

The EM algorithm alternates between two steps.

---

## Step 1 — Initialization

The algorithm begins with initial guesses.

In this implementation:

```
mean1 = overall mean − 5

mean2 = overall mean + 5

variance1 = 25

variance2 = 25

pi1 = 0.5

pi2 = 0.5
```

Reasons for initialization
```
The average used for the **mean** gives a location where the data is generally centered unless if random numbers were picked , they can make EM converge to a poor solution.

25 used as the **variance** is used because most heights are roughly within 5 units of the mean.
1 is not used because the Gaussian probability becomes almost zero.
A large number is not picked because every height would get almost the same probability.

0.5 is used for the **mixin coeffecient** just to assume that each guassian represents half the weight of the data


---

## Step 2 — Expectation (E-Step)

For every height:

Compute the probability that it came from Gaussian 1.

Compute the probability that it came from Gaussian 2.

These probabilities are weighted using the mixing coefficients.

```
prob1 = π1 × Gaussian1

prob2 = π2 × Gaussian2
```

Normalize them

```
r1 = prob1 / (prob1 + prob2)

r2 = prob2 / (prob1 + prob2)
```

These are called **responsibilities**.

Each responsibility tells us how much that data point belongs to each Gaussian.

Example

Height = 67

```
Gaussian 1 = 0.34

Gaussian 2 = 0.66
```

Meaning

```
34% Distribution 1

66% Distribution 2
```

# Mathematical Example of Expectation step

Assume one height observation:

```
x = 68
```

Current parameters:

| Parameter | Gaussian 1 | Gaussian 2 |
|-----------|-----------:|-----------:|
| μ | 65 | 72 |
| σ² | 9 | 16 |
| π | 0.5 | 0.5 |

## E-Step

Calculate the Gaussian probabilities:

```
P1 ≈ 0.0807
P2 ≈ 0.0605
```

Multiply by the mixing coefficients:

```
0.5 × 0.0807 = 0.04035
0.5 × 0.0605 = 0.03025
```

Normalize to obtain the responsibilities:

```
Total = 0.04035 + 0.03025 = 0.07060

r1 = 0.04035 / 0.07060 = 0.572
r2 = 0.03025 / 0.07060 = 0.428
```

**Interpretation:** Height **68** belongs **57.2%** to Gaussian 1 and **42.8%** to Gaussian 2. This is a **soft assignment**.
---

## Step 3 — Maximization (M-Step)

After calculating responsibilities for every height, update the Gaussian parameters.

### Update the means

Each height contributes according to its responsibility.

Instead of every point counting equally, points contribute proportionally.

---

### Update the variances

The variance becomes the weighted spread around the new mean.

---

### Update the mixing coefficients

```
π = effective number of points
    -------------------------
       total observations
```

This estimates the proportion of the dataset belonging to each Gaussian.

## M-Step

Suppose the responsibilities for Gaussian 1 are:

| Height | Responsibility |
|-------:|---------------:|
| 65 | 0.90 |
| 68 | 0.57 |
| 72 | 0.10 |

Effective number of observations:

```
N1 = 0.90 + 0.57 + 0.10 = 1.57
```

Update the mean:

```
μ = (0.90×65 + 0.57×68 + 0.10×72) / 1.57
  = (58.50 + 38.76 + 7.20) / 1.57
  = 104.46 / 1.57
  ≈ 66.54
```

Update the variance:

```
σ² = [0.90(65−66.54)² + 0.57(68−66.54)² + 0.10(72−66.54)²] / 1.57
```

Update the mixing coefficient:

```
π = 1.57 / 3 = 0.523
```

**Interpretation:** About **52.3%** of the dataset currently belongs to Gaussian 1.

## One EM Iteration

```
Initialize μ, σ² and π
        ↓
Compute Gaussian probabilities
        ↓
Calculate responsibilities (E-Step)
        ↓
Update μ, σ² and π (M-Step)
        ↓
Compute log-likelihood
        ↓
Repeat until log-likelihood converges
```
---

## Step 4 — Log-Likelihood

After updating parameters, calculate

```
Log-Likelihood
```

This measures how well the Gaussian mixture explains the data.

Higher log-likelihood indicates a better fit.

The optimization should steadily improve until convergence.

---

# Stopping Condition

The mathematically correct stopping condition is:

Stop when the **log-likelihood changes negligibly** between iterations.

This implementation runs for a fixed number of iterations:

```python
iterations = 10
```

During presentations, the tracking table shows that the log-likelihood improves over iterations.

---

# Optimization Tracking Table

| Iteration | μ₁ (Mean 1) | μ₂ (Mean 2) | σ₁² (Variance 1) | σ₂² (Variance 2) | π₁ | π₂ | Log-Likelihood |
|:---------:|------------:|------------:|-----------------:|-----------------:|---:|---:|---------------:|
| 0 | 66.30 | 69.60 | 9.45 | 7.07 | 0.49 | 0.51 | -4871.94 |
| 1 | 66.27 | 69.61 | 9.96 | 6.48 | 0.49 | 0.51 | -4868.06 |
| 2 | 66.24 | 69.62 | 10.18 | 6.14 | 0.49 | 0.51 | -4866.64 |
| 3 | 66.22 | 69.64 | 10.27 | 5.92 | 0.49 | 0.51 | -4865.98 |
| 4 | 66.20 | 69.66 | 10.29 | 5.76 | 0.49 | 0.51 | -4865.62 |
| 5 | 66.19 | 69.68 | 10.28 | 5.65 | 0.49 | 0.51 | -4865.38 |
| 6 | 66.18 | 69.70 | 10.25 | 5.57 | 0.49 | 0.51 | -4865.21 |
| 7 | 66.17 | 69.72 | 10.20 | 5.51 | 0.49 | 0.51 | -4865.07 |
| 8 | 66.16 | 69.73 | 10.16 | 5.45 | 0.49 | 0.51 | -4864.96 |
| 9 | 66.15 | 69.75 | 10.11 | 5.41 | 0.49 | 0.51 | -4864.86 |

Enter test height: 67

Prediction for: 67.0
Child probability: 58.07 %
Parent probability: 41.93 %
---

# Running the Project

## Requirements

Python 3.x

No external libraries are required.

Only Python standard libraries are used:

- csv
- math

---

## Execute

Run

```bash
python FORMATIVE_TEST.py
```

Enter any height when prompted.

Example

```
Enter test height:

68
```

---




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

# Part 4 — Gradient Descent in Code

## What this is
This converts the Part 3 manual gradient descent calculations into working Python code, using SciPy and Matplotlib as required.

## Files
- `Part4_Gradient_Descent.ipynb` — Jupyter notebook (recommended). Already run once, so outputs and plots show immediately on open. Can also be re-run live during the presentation.
- `part4_gradient_descent.py` — same code as a plain Python script.

## How to run
**Option A — Jupyter / Google Colab (recommended)**
1. Go to https://colab.research.google.com
2. File → Upload notebook → select `Part4_Gradient_Descent.ipynb`
3. Run cells top to bottom (Shift + Enter on each), or just view the saved outputs.

**Option B — Local Python**
```
pip install numpy scipy matplotlib
python part4_gradient_descent.py
```

## What the code does
1. Starts from the same data, `m`, `b`, and learning rate (0.01) used in Part 3.
2. Defines the Mean Squared Error cost function.
3. Uses `scipy.optimize.approx_fprime` to compute the derivative of that cost function with respect to `m` and `b` — the SciPy requirement for this part.
4. Also computes the same gradient analytically (by hand-derived formula) and prints both side by side to show they match.
5. Updates `m` and `b` using gradient descent, once per iteration, in matrix form — every intermediate value (`y_hat`, error, MSE, gradient) is printed, nothing is hidden inside one function call.
6. Runs 4 iterations — one per group member.
7. Computes final predictions using the final `m` and `b`.
8. Plots:
   - `m` and `b` values across iterations
   - MSE (error) across iterations

## Result summary

| Iteration | m | b | MSE |
|---|---|---|---|
| 0 (init) | [-1.00, 2.00] | [1.00, 1.00] | 61.00 |
| 1 | [-1.45, 0.87] | [0.99, 0.89] | 6.50 |
| 2 | [-1.33, 1.18] | [1.02, 0.91] | 2.50 |
| 3 | [-1.37, 1.10] | [1.04, 0.90] | 2.16 |
| 4 | [-1.36, 1.12] | [1.06, 0.89] | 2.10 |

Error drops sharply after the first iteration, then keeps decreasing but flattens out. Both `m` and `b` move steadily in the direction that reduces error, confirming gradient descent is working — it slows down as it approaches a minimum, which is expected with a fixed learning rate.

## Contribution
Part 4 (this notebook/script) completed by: **Loice Teta**
## Contribution

**Gakwaya Ineza Ketia** — Responsible for Part 3:

- Manual prediction calculations
- Gradient derivation
- Four Gradient Descent iterations
- Parameter updates
- Analysis of convergence
- README documentation
