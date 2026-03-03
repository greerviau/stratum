"""Generate synthetic datasets used by the calcine examples.

Run once from the repo root:

    python examples/data/generate_data.py

Produces:
    examples/data/reviews.csv          ~100k rows  (5 000 users × ~20 reviews)
    examples/data/user_profiles.csv    5 000 rows
    examples/data/sensor_readings.csv  ~500k rows  (1 000 sensors × ~500 readings)
    examples/data/documents/           1 000 plain-text files  (one per entity)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
random.seed(42)

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. User reviews
# ---------------------------------------------------------------------------

VOCAB = (
    "good bad great poor excellent terrible fine awful amazing mediocre love hate okay "
    "fantastic product service quality price fast slow broken new old expensive cheap "
    "delivery packaging return reliable durable lightweight heavy feature missing works "
    "disappointed satisfied recommend avoid buy skip value money worth upgrade downgrade"
).split()

CATEGORIES = ["electronics", "clothing", "books", "home", "sports", "food", "toys"]


def _review(rng: np.random.Generator, length: int | None = None) -> str:
    n = int(rng.integers(8, 60)) if length is None else length
    return " ".join(rng.choice(VOCAB, size=n).tolist())


print("Generating reviews.csv …")
N_USERS = 5_000
rows = []
for uid in range(N_USERS):
    n = int(RNG.integers(5, 35))
    for _ in range(n):
        rows.append(
            {
                "entity_id": f"user_{uid:05d}",
                "review": _review(RNG),
                "rating": int(RNG.integers(1, 6)),
                "category": str(RNG.choice(CATEGORIES)),
                "helpful_votes": int(RNG.integers(0, 50)),
            }
        )

reviews_df = pd.DataFrame(rows)
reviews_df.to_csv(OUT / "reviews.csv", index=False)
print(f"  {len(reviews_df):,} rows  →  reviews.csv")

# ---------------------------------------------------------------------------
# 2. User profiles
# ---------------------------------------------------------------------------

print("Generating user_profiles.csv …")
COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR"]
PLANS = ["free", "basic", "pro", "enterprise"]

profiles = pd.DataFrame(
    {
        "entity_id": [f"user_{uid:05d}" for uid in range(N_USERS)],
        "age": RNG.integers(18, 75, size=N_USERS),
        "country": RNG.choice(COUNTRIES, size=N_USERS),
        "account_plan": RNG.choice(PLANS, size=N_USERS, p=[0.5, 0.25, 0.18, 0.07]),
        "tenure_days": RNG.integers(1, 3_650, size=N_USERS),
        "email_verified": RNG.choice([True, False], size=N_USERS, p=[0.85, 0.15]),
    }
)
profiles.to_csv(OUT / "user_profiles.csv", index=False)
print(f"  {len(profiles):,} rows  →  user_profiles.csv")

# ---------------------------------------------------------------------------
# 3. Sensor time-series
# ---------------------------------------------------------------------------

print("Generating sensor_readings.csv …")
N_SENSORS = 1_000
BASE_FREQ = RNG.uniform(0.01, 0.2, size=N_SENSORS)  # cycles per reading
BASE_AMP = RNG.uniform(1.0, 50.0, size=N_SENSORS)
BASE_BIAS = RNG.uniform(-10.0, 10.0, size=N_SENSORS)
FAULT_PROB = RNG.uniform(0.001, 0.01, size=N_SENSORS)  # spike probability

sensor_rows = []
for sid in range(N_SENSORS):
    n_readings = int(RNG.integers(200, 800))
    t = np.arange(n_readings, dtype=float)
    signal = (
        BASE_AMP[sid] * np.sin(2 * np.pi * BASE_FREQ[sid] * t)
        + BASE_BIAS[sid]
        + RNG.normal(0, BASE_AMP[sid] * 0.05, size=n_readings)
    )
    # Inject random spikes
    spike_mask = RNG.random(size=n_readings) < FAULT_PROB[sid]
    signal[spike_mask] += RNG.normal(0, BASE_AMP[sid] * 5, size=spike_mask.sum())

    for _i, (ts, val) in enumerate(zip(t.tolist(), signal.tolist(), strict=True)):
        sensor_rows.append(
            {
                "entity_id": f"sensor_{sid:04d}",
                "t": int(ts),
                "value": round(val, 4),
                "sensor_type": "temperature"
                if sid % 3 == 0
                else "pressure"
                if sid % 3 == 1
                else "humidity",
            }
        )

sensor_df = pd.DataFrame(sensor_rows)
sensor_df.to_csv(OUT / "sensor_readings.csv", index=False)
print(f"  {len(sensor_df):,} rows  →  sensor_readings.csv")

# ---------------------------------------------------------------------------
# 4. Document directory  (one text file per entity, 1 000 entities)
# ---------------------------------------------------------------------------

print("Generating documents/ directory …")
N_DOCS = 1_000
doc_dir = OUT / "documents"
doc_dir.mkdir(exist_ok=True)

TOPICS = {
    "science": "research experiment hypothesis data analysis results conclusion theory",
    "finance": "revenue profit loss market investment portfolio return risk dividend",
    "sports": "team player game score win loss tournament season champion league",
    "technology": "software hardware algorithm performance latency throughput system network",
    "health": "patient treatment diagnosis symptom medication therapy recovery clinical",
}

for doc_id in range(N_DOCS):
    topic_name = RNG.choice(list(TOPICS.keys()))
    topic_words = TOPICS[topic_name].split()
    n_words = int(RNG.integers(80, 400))
    # Mix topic words with generic vocab
    pool = topic_words * 4 + VOCAB
    text = " ".join(random.choices(pool, k=n_words))
    (doc_dir / f"doc_{doc_id:04d}.txt").write_text(text)

print(f"  {N_DOCS} files  →  documents/")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nAll artifacts written to examples/data/")
print(f"  reviews.csv          {reviews_df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")
print(f"  user_profiles.csv    {profiles.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")
print(f"  sensor_readings.csv  {sensor_df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")
print(f"  documents/           {N_DOCS} .txt files")
