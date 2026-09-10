"""
STEP 1: Make some fake (but realistic) data.

Imagine we run a small delivery company. We want to predict how many
MINUTES a delivery will take, based on:
  - distance_km      : how far the delivery is
  - num_stops        : how many stops the driver makes before it
  - is_rush_hour      : 1 if during rush hour, 0 if not
  - driver_experience: years the driver has been driving

We create TWO datasets:
  - old_deliveries.csv  -> data from "last year" (what we train on)
  - new_deliveries.csv  -> data from "this month" (simulates new data
                            coming in, with a changed pattern - this
                            is what "drift" means)

This is just so we have something realistic to practice on. In a real
job, this file would instead pull from your company's database.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


def make_deliveries(n_rows, rush_hour_chance, distance_shift=0):
    """Create n_rows of pretend delivery data."""
    distance_km = np.random.uniform(1, 20, n_rows) + distance_shift
    num_stops = np.random.randint(0, 5, n_rows)
    is_rush_hour = np.random.binomial(1, rush_hour_chance, n_rows)
    driver_experience = np.random.uniform(0, 15, n_rows)

    # The "true" rule the real world follows (with some random noise added,
    # because real life is never perfectly predictable)
    delivery_minutes = (
        5
        + distance_km * 2.2
        + num_stops * 4
        + is_rush_hour * 12
        - driver_experience * 0.5
        + np.random.normal(0, 3, n_rows)
    )

    return pd.DataFrame({
        "distance_km": distance_km,
        "num_stops": num_stops,
        "is_rush_hour": is_rush_hour,
        "driver_experience": driver_experience,
        "delivery_minutes": delivery_minutes,
    })


if __name__ == "__main__":
    here = os.path.dirname(__file__)

    # "Old" data: what we train the model on (like last year's records)
    old_df = make_deliveries(2000, rush_hour_chance=0.3)
    old_df.to_csv(os.path.join(here, "old_deliveries.csv"), index=False)

    # "New" data: this month's deliveries. Notice rush_hour_chance is higher
    # and distances are a bit longer -> the world has quietly changed.
    # This is what will trigger "drift" later in the tutorial.
    new_df = make_deliveries(500, rush_hour_chance=0.6, distance_shift=4)
    new_df.to_csv(os.path.join(here, "new_deliveries.csv"), index=False)

    print("Created data/old_deliveries.csv (2000 rows) - used for training")
    print("Created data/new_deliveries.csv (500 rows)  - simulates new data")
