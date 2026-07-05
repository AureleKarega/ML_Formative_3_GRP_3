"""
Bayesian Sentiment Analysis on IMDB Dataset
Uses Bayes' Theorem to compute P(Positive | keyword) for selected keywords.
No external ML libraries — only csv, re, and basic Python.
"""

import csv
import re

# ──────────────────────────────────────────────
# 1. Load the dataset
# ──────────────────────────────────────────────
DATASET_PATH = "IMDB Dataset.csv"

reviews = []
with open(DATASET_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        reviews.append({
            "text": row["review"].lower(),
            "sentiment": row["sentiment"].strip().lower()
        })

total = len(reviews)
num_positive = sum(1 for r in reviews if r["sentiment"] == "positive")
num_negative = total - num_positive

print(f"Dataset loaded: {total} reviews")
print(f"  Positive: {num_positive} ({num_positive/total:.1%})")
print(f"  Negative: {num_negative} ({num_negative/total:.1%})")
print()

# ──────────────────────────────────────────────
# 2. Keyword selection
# ──────────────────────────────────────────────
POSITIVE_KEYWORDS = ["excellent", "masterpiece", "brilliant", "heartwarming"]
NEGATIVE_KEYWORDS = ["terrible", "awful", "boring", "waste"]

ALL_KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS

# ──────────────────────────────────────────────
# 3. Helper: count keyword occurrences
# ──────────────────────────────────────────────
def contains_keyword(text: str, keyword: str) -> bool:
    """Return True if the keyword appears as a whole word in the text."""
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))

def compute_counts(keyword: str):
    kw_in_positive = sum(1 for r in reviews
                         if r["sentiment"] == "positive" and contains_keyword(r["text"], keyword))
    kw_in_negative = sum(1 for r in reviews
                         if r["sentiment"] == "negative" and contains_keyword(r["text"], keyword))
    kw_total = kw_in_positive + kw_in_negative
    return kw_in_positive, kw_in_negative, kw_total

# ──────────────────────────────────────────────
# 4. Bayes' Theorem — computing P(Positive | keyword)
# ──────────────────────────────────────────────
#
#  P(Pos | kw) = P(kw | Pos) * P(Pos)
#                ─────────────────────
#                      P(kw)
#
# Where:
#   Prior       = P(Positive)             = num_positive / total
#   Likelihood  = P(keyword | Positive)   = kw_in_positive / num_positive
#   Marginal    = P(keyword)              = kw_total / total
#   Posterior   = P(Positive | keyword)   [computed via Bayes]
# ──────────────────────────────────────────────

results = []

for keyword in ALL_KEYWORDS:
    kw_in_pos, kw_in_neg, kw_total = compute_counts(keyword)

    if kw_total == 0:
        print(f"  WARNING: keyword '{keyword}' not found in dataset — skipping.")
        continue

    prior      = num_positive / total                 # P(Positive)
    likelihood = kw_in_pos / num_positive             # P(keyword | Positive)
    marginal   = kw_total / total                     # P(keyword)
    posterior  = (likelihood * prior) / marginal      # P(Positive | keyword)

    results.append({
        "keyword":    keyword,
        "kw_total":  kw_total,
        "kw_in_pos": kw_in_pos,
        "kw_in_neg": kw_in_neg,
        "prior":     prior,
        "likelihood": likelihood,
        "marginal":  marginal,
        "posterior": posterior,
        "group":     "positive" if keyword in POSITIVE_KEYWORDS else "negative"
    })

# ──────────────────────────────────────────────
# 5. Print results table
# ──────────────────────────────────────────────
COL = 14

def fmt(val, pct=True):
    return f"{val:.4f} ({val:.1%})" if pct else f"{val}"

header = (
    f"{'Keyword':<14} {'Group':<10} "
    f"{'Prior P(+)':<18} {'Likelihood P(kw|+)':<22} "
    f"{'Marginal P(kw)':<20} {'Posterior P(+|kw)':<20} "
    f"{'Count in +':<12} {'Count in -':<12} {'Total'}"
)
sep = "─" * len(header)

print("=" * len(header))
print("  BAYESIAN SENTIMENT ANALYSIS — IMDB Dataset")
print("  Chosen direction: P(Positive | keyword)")
print("=" * len(header))
print()
print(header)
print(sep)

for r in results:
    marker = "✔" if r["group"] == "positive" else "✘"
    print(
        f"{r['keyword']:<14} {r['group']:<10} "
        f"{r['prior']:<18.4f} {r['likelihood']:<22.4f} "
        f"{r['marginal']:<20.4f} {r['posterior']:<20.4f} "
        f"{r['kw_in_pos']:<12} {r['kw_in_neg']:<12} {r['kw_total']}"
    )

print(sep)
print()

# ──────────────────────────────────────────────
# 6. Interpretation
# ──────────────────────────────────────────────
print("INTERPRETATION")
print("─" * 60)
for r in results:
    direction = "strongly positive" if r["posterior"] > 0.7 else \
                "likely positive"   if r["posterior"] > 0.55 else \
                "neutral/mixed"     if r["posterior"] > 0.45 else \
                "likely negative"   if r["posterior"] > 0.3 else \
                "strongly negative"
    print(f"  '{r['keyword']}' → P(Positive|keyword) = {r['posterior']:.1%}  [{direction}]")

print()
print("NOTE: Prior P(Positive) =", f"{num_positive/total:.1%}",
      "— keywords above this raise the probability of positivity,")
print("      keywords below lower it.")
