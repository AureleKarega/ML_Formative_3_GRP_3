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

These are only starting estimates.

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

The program prints the optimization progress after every iteration.

Example

| Iteration | μ1 | μ2 | Var1 | Var2 | π1 | π2 | Log-Likelihood |
|-----------|----|----|------|------|----|----|----------------|
| 0 | Initial | Initial | Initial | Initial | 0.5 | 0.5 | Initial |
| 1 | Updated | Updated | Updated | Updated | Updated | Updated | Improved |
| 2 | Updated | Updated | Updated | Updated | Updated | Updated | Improved |

This table demonstrates how EM gradually improves its estimates.

---

# Prediction

After training, the model predicts the probability that a new height belongs to each group.

Example

```
Enter test height:

69
```

Possible output

```
Prediction for: 69

Child probability:

27.54 %

Parent probability:

72.46 %
```

Rather than assigning a hard label, the model outputs posterior probabilities.

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


# Presentation Checklist

During the presentation, demonstrate:

- Explain why splitting at the global mean is not appropriate.
- Explain the Gaussian distribution.
- Explain initialization.
- Explain the Expectation (E-Step).
- Explain the Maximization (M-Step).
- Explain posterior probabilities (responsibilities).
- Explain log-likelihood.
- Show the optimization tracking table.
- Run the program live.
- Enter a random test height provided by the instructor.
- Show the exact posterior probabilities for the test height.

---
