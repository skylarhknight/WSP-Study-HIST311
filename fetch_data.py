"""
Women’s healthcare access vs female physical activity (cross-country, single-year 2019)

API-only pipeline:
- WHO GHO OData API (indicator table endpoints like /api/NCD_PAA)
- World Bank Indicators API v2

Output: women_healthcare_activity_2019.csv
"""

from __future__ import annotations
import requests
import pandas as pd
import time

WHO_BASE = "https://ghoapi.azureedge.net/api"
WB_BASE = "https://api.worldbank.org/v2"
YEAR = 2019

WHO_TOP_LIMIT = 1000  # WHO OData enforces $top <= 1000


# -----------------------------
# WHO helpers
# -----------------------------
def who_get(endpoint: str, params: dict | None = None, timeout: int = 120, retries: int = 3) -> dict:
    url = f"{WHO_BASE}/{endpoint.lstrip('/')}"
    params = params or {}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_err = e
            # Print useful context once; retry with backoff for transient issues
            if attempt == 1:
                print("WHO request failed:", r.url if "r" in locals() else url)
                if "r" in locals():
                    print("Status:", getattr(r, "status_code", None))
                    print("Response text (first 800 chars):")
                    print(getattr(r, "text", "")[:800])
            if attempt < retries:
                time.sleep(1.5 * attempt)
            else:
                raise last_err


def who_single(endpoint: str, params: dict | None = None) -> pd.DataFrame:
    js = who_get(endpoint, params=params or {}, timeout=60)
    return pd.DataFrame(js.get("value", []))


def who_paged(endpoint: str, params: dict | None = None, page_size: int = WHO_TOP_LIMIT, max_rows: int = 1_000_000) -> pd.DataFrame:
    """
    Page using $top/$skip. WHO enforces $top <= 1000, so page_size must be <= 1000.
    """
    if page_size > WHO_TOP_LIMIT:
        raise ValueError(f"page_size={page_size} exceeds WHO $top limit of {WHO_TOP_LIMIT}.")

    out: list[dict] = []
    skip = 0
    base_params = dict(params or {})
    base_params.setdefault("$orderby", "SpatialDim")

    while True:
        p = dict(base_params)
        p["$top"] = page_size
        p["$skip"] = skip

        js = who_get(endpoint, p, timeout=180)
        rows = js.get("value", [])
        if not rows:
            break

        out.extend(rows)
        skip += page_size

        # Progress every ~2k rows
        if len(out) % (page_size * 2) == 0:
            print(f"    ...fetched {len(out)} rows so far")

        if len(out) > max_rows:
            raise RuntimeError(f"WHO paging exceeded {max_rows} rows; aborting to prevent runaway download.")

    return pd.DataFrame(out)


def who_dimension_values(dim_code: str) -> pd.DataFrame:
    return pd.DataFrame(who_get(f"Dimension/{dim_code}/DimensionValues", timeout=60).get("value", []))


def infer_female_value_code() -> str:
    sex_vals = who_dimension_values("SEX")
    cols = {c.lower(): c for c in sex_vals.columns}
    code_col = cols.get("code")
    title_col = cols.get("title")
    if code_col is None or title_col is None:
        raise RuntimeError(f"Unexpected SEX dimension schema: {sex_vals.columns.tolist()}")

    female_row = sex_vals[sex_vals[title_col].str.lower().str.contains("female", na=False)]
    if female_row.empty:
        raise RuntimeError("Could not find 'female' in WHO SEX dimension values.")
    return str(female_row.iloc[0][code_col])


def detect_time_field(indicator_endpoint: str) -> str:
    sample = who_single(indicator_endpoint, params={"$top": 5, "$skip": 0, "$orderby": "SpatialDim"})
    if sample.empty:
        raise RuntimeError(f"No rows returned from WHO endpoint {indicator_endpoint}; cannot detect time field.")

    for f in ["TimeDimensionValue", "TimeDimensionBegin", "TimeDimensionYear"]:
        if f in sample.columns:
            return f

    raise RuntimeError(
        f"Could not detect a time field for {indicator_endpoint}. "
        f"Columns: {sample.columns.tolist()}"
    )


def who_pull_indicator_year(indicator_endpoint: str, year: int, sex_code: str | None = None) -> pd.DataFrame:
    time_field = detect_time_field(indicator_endpoint)
    print(f"  Detected time field for {indicator_endpoint}: {time_field}")

    if time_field == "TimeDimensionValue":
        time_filter = f"{time_field} eq '{year}'"
    else:
        time_filter = f"{time_field} eq {year}"

    if sex_code is not None:
        filt = f"({time_filter}) and (Dim1 eq '{sex_code}')"
    else:
        filt = time_filter

    print(f"  WHO fetch: {indicator_endpoint} with filter: {filt}")

    df = who_paged(indicator_endpoint, params={"$filter": filt}, page_size=WHO_TOP_LIMIT)

    if "SpatialDimType" in df.columns:
        df = df[df["SpatialDimType"] == "COUNTRY"].copy()

    # If empty with server-side sex filter, fall back to year-only then filter locally
    if df.empty and sex_code is not None:
        print("  Server-side sex filter returned 0 rows; falling back to year-only + local sex filter.")
        df_year = who_paged(indicator_endpoint, params={"$filter": time_filter}, page_size=WHO_TOP_LIMIT)
        if "SpatialDimType" in df_year.columns:
            df_year = df_year[df_year["SpatialDimType"] == "COUNTRY"].copy()

        dim_cols = [c for c in df_year.columns if c.startswith("Dim")]
        for c in dim_cols:
            if (df_year[c] == sex_code).any():
                df = df_year[df_year[c] == sex_code].copy()
                break

    return df


# -----------------------------
# World Bank helpers
# -----------------------------
def wb_get_indicator(indicator: str, year: int, countries: str = "all", per_page: int = 20000) -> pd.DataFrame:
    url = f"{WB_BASE}/country/{countries}/indicator/{indicator}"
    params = {"format": "json", "date": f"{year}:{year}", "per_page": per_page, "page": 1}

    rows: list[dict] = []
    while True:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        meta, data = r.json()
        rows.extend([d for d in data if d is not None])
        if params["page"] >= meta["pages"]:
            break
        params["page"] += 1

    df = pd.json_normalize(rows)
    keep = ["countryiso3code", "country.value", "date", "value"]
    df = df[keep].rename(columns={
        "countryiso3code": "iso3",
        "country.value": "country_wb",
        "date": "year",
        "value": indicator
    })
    df["year"] = df["year"].astype(int)
    return df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    female_code = infer_female_value_code()
    print("WHO female SEX code:", female_code)

    print("Fetching WHO NCD_PAA (physical activity)…")
    pa = who_pull_indicator_year("NCD_PAA", YEAR, sex_code=female_code)
    if pa.empty:
        raise RuntimeError("WHO NCD_PAA returned 0 rows after filtering.")

    if not {"SpatialDim", "NumericValue"}.issubset(pa.columns):
        raise RuntimeError(f"Unexpected WHO schema for NCD_PAA. Columns: {pa.columns.tolist()}")

    pa_f = pa.rename(columns={"SpatialDim": "iso3", "NumericValue": "insuff_pa_female_pct"})[
        ["iso3", "insuff_pa_female_pct"]
    ].dropna()

    print("Fetching WHO skilled birth attendance…")
    sba = who_pull_indicator_year("MDG_0000000025", YEAR, sex_code=None)
    if sba.empty:
        raise RuntimeError("WHO MDG_0000000025 returned 0 rows after filtering.")

    if not {"SpatialDim", "NumericValue"}.issubset(sba.columns):
        raise RuntimeError(f"Unexpected WHO schema for MDG_0000000025. Columns: {sba.columns.tolist()}")

    sba = sba.rename(columns={"SpatialDim": "iso3", "NumericValue": "skilled_birth_attendance_pct"})[
        ["iso3", "skilled_birth_attendance_pct"]
    ].dropna()

    print("Fetching World Bank controls…")
    gdp = wb_get_indicator("NY.GDP.PCAP.CD", YEAR)
    edu = wb_get_indicator("SE.SEC.ENRR.FE", YEAR)

    df = (
        pa_f.merge(sba, on="iso3", how="inner")
            .merge(gdp[["iso3", "NY.GDP.PCAP.CD"]], on="iso3", how="left")
            .merge(edu[["iso3", "SE.SEC.ENRR.FE"]], on="iso3", how="left")
    )

    df["female_activity_rate_pct"] = 100 - df["insuff_pa_female_pct"]

    print("\nFinal dataset shape:", df.shape)
    print("Missingness (top):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\nPreview:")
    print(df.head(10))

    out_path = f"women_healthcare_activity_{YEAR}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
