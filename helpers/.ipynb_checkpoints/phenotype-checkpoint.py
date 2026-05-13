import numpy as np
import pandas as pd

def attach_populations(n_rows: int, k: int, c: int) -> pd.DataFrame:
    """Construct a populations DataFrame with balanced groups.
    Pop IDs go from 1..k*k, each repeated c times.
    """
    num_pops = k * k
    expected_rows = num_pops * c
    if n_rows != expected_rows:
        raise ValueError(
            f"Row count {n_rows} != expected k*k*c = {expected_rows}. Check inputs."
        )
    pops = np.repeat(np.arange(1, num_pops + 1), c)
    return pd.DataFrame({"populations": pops})
