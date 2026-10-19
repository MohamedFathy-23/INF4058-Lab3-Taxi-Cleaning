from pathlib import Path
import pandas as pd

DATA = Path("/workspace/yellow_tripdata_2021-01.csv.gz")
OUT = Path("/workspace/reports")
OUT.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 100_000

PICKUP = "tpep_pickup_datetime"
FARE = "fare_amount"
DISTANCE = "trip_distance"
PASSENGERS = "passenger_count"


def clean(df):
    """Return a cleaned copy and counts of rows removed by each rule."""
    result = df.copy()
    removed = {}

    def apply_rule(name, mask):
        nonlocal result
        mask = mask.fillna(False)
        removed[name] = int(mask.sum())
        result = result.loc[~mask].copy()

    # Reject missing or invalid pickup timestamps.
    if PICKUP in result.columns:
        dt = pd.to_datetime(result[PICKUP], errors="coerce")
        apply_rule("invalid_or_missing_pickup", dt.isna())

        dt = pd.to_datetime(result[PICKUP], errors="coerce")
        apply_rule(
            "pickup_outside_january_2021",
            ~dt.between("2021-01-01", "2021-01-31 23:59:59.999999")
        )

    if FARE in result.columns:
        apply_rule(
            "fare_zero_or_negative",
            pd.to_numeric(result[FARE], errors="coerce").le(0)
        )

    if DISTANCE in result.columns:
        apply_rule(
            "distance_zero_or_negative",
            pd.to_numeric(result[DISTANCE], errors="coerce").le(0)
        )

    if PASSENGERS in result.columns:
        apply_rule(
            "passenger_count_zero",
            pd.to_numeric(result[PASSENGERS], errors="coerce").eq(0)
        )

    # Exact duplicates are removed after the other rules.
    apply_rule("duplicate_rows", result.duplicated(keep="first"))

    return result, removed


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    print("STEP 1: FIRST 100 ROWS")
    first = pd.read_csv(DATA, nrows=100, low_memory=False)
    print(first.head())
    print("\nSTEP 1: DATA TYPES")
    print(first.dtypes)

    print("\nSTEP 2: CHUNKED PROFILING (100,000 rows per chunk)")
    profile = None
    problems = {
        "pickup_outside_january_2021": 0,
        "fare_zero_or_negative": 0,
        "distance_zero_or_negative": 0,
        "passenger_count_zero": 0,
        "duplicate_rows_within_chunks": 0,
    }
    total_rows = 0
    rows_after = 0
    removed_total = {}

    for chunk_no, chunk in enumerate(
        pd.read_csv(DATA, chunksize=CHUNK_SIZE, low_memory=False), start=1
    ):
        total_rows += len(chunk)

        if profile is None:
            profile = pd.DataFrame(index=chunk.columns)
            profile["dtype"] = chunk.dtypes.astype(str)
            profile["nulls"] = 0
            profile["min"] = None
            profile["max"] = None

        profile["nulls"] += chunk.isna().sum()

        for col in chunk.columns:
            values = chunk[col]
            if col == PICKUP:
                values = pd.to_datetime(values, errors="coerce")
            try:
                lo, hi = values.min(), values.max()
                if pd.notna(lo):
                    if pd.isna(profile.at[col, "min"]) or lo < profile.at[col, "min"]:
                        profile.at[col, "min"] = lo
                if pd.notna(hi):
                    if pd.isna(profile.at[col, "max"]) or hi > profile.at[col, "max"]:
                        profile.at[col, "max"] = hi
            except (TypeError, ValueError):
                pass

        if PICKUP in chunk:
            dt = pd.to_datetime(chunk[PICKUP], errors="coerce")
            problems["pickup_outside_january_2021"] += int(
                (~dt.between("2021-01-01", "2021-01-31 23:59:59.999999")
                 & dt.notna()).sum()
            )
        if FARE in chunk:
            problems["fare_zero_or_negative"] += int(
                pd.to_numeric(chunk[FARE], errors="coerce").le(0).sum()
            )
        if DISTANCE in chunk:
            problems["distance_zero_or_negative"] += int(
                pd.to_numeric(chunk[DISTANCE], errors="coerce").le(0).sum()
            )
        if PASSENGERS in chunk:
            problems["passenger_count_zero"] += int(
                pd.to_numeric(chunk[PASSENGERS], errors="coerce").eq(0).sum()
            )

        cleaned, counts = clean(chunk)
        rows_after += len(cleaned)
        problems["duplicate_rows_within_chunks"] += counts["duplicate_rows"]
        for rule, count in counts.items():
            removed_total[rule] = removed_total.get(rule, 0) + count

        print(f"Processed chunk {chunk_no}: {len(chunk)} rows")

    profile["null_pct"] = (profile["nulls"] / total_rows * 100).round(2)
    profile = profile[["dtype", "nulls", "null_pct", "min", "max"]]

    print("\nPROFILE TABLE")
    print(profile.to_string())
    profile.to_csv(OUT / "profiling.csv")

    print("\nPROBLEM COUNTS (some categories may overlap)")
    problem_table = pd.DataFrame(
        [{"problem": k, "count": v} for k, v in problems.items()]
    )
    print(problem_table.to_string(index=False))
    problem_table.to_csv(OUT / "problem_counts.csv", index=False)

    print("\nCLEANING SUMMARY")
    print("Rows processed:", total_rows)
    print("Rows remaining after per-chunk cleaning:", rows_after)
    print("Rows removed:", total_rows - rows_after)
    print("Rows removed per rule:", removed_total)
    pd.DataFrame(
        [{"rule": k, "rows_removed": v} for k, v in removed_total.items()]
    ).to_csv(OUT / "cleaning_summary.csv", index=False)

    print("\nIMPORTANT LIMITATION:")
    print("Duplicate detection is within each chunk only; duplicates across chunk")
    print("boundaries are not counted. The cleaned data was NOT loaded into PostgreSQL.")
    print("Reports saved in:", OUT)


if __name__ == "__main__":
    main()
