"""Per-example uncertainty summaries."""
import math
import random


def summarize(values, seed=0, resamples=2000):
    """Mean, standard error and percentile-bootstrap 95% interval of finite values."""
    data = [float(v) for v in values if v is not None and math.isfinite(v)]
    n = len(data)
    if n == 0:
        return {"n": 0, "mean": None, "sem": None, "ci95": [None, None], "median": None}
    mean = sum(data) / n
    sem = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1) / n) if n > 1 else 0.0
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(data) for _ in range(n)) / n for _ in range(resamples)) if n > 1 else [mean]
    low, high = boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]
    ordered = sorted(data)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"n": n, "mean": mean, "sem": sem, "ci95": [low, high], "median": median}
