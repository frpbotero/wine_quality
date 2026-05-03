from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "winequality-red.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"


def preprocess(
    raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH
) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(
        """
        CREATE OR REPLACE TABLE raw AS
        SELECT *
        FROM read_csv_auto(?)
        """,
        [str(raw_path)],
    )

    con.execute(
        """
        COPY (
            WITH dedup AS (
                SELECT DISTINCT * FROM raw
            )
            SELECT
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                ph,
                sulphates,
                alcohol,
                -- Fusão de classes: 0=Ruim (<=5), 1=Médio (6), 2=Bom (>=7)
                CASE
                    WHEN quality <= 5 THEN 0
                    WHEN quality = 6  THEN 1
                    ELSE 2
                END AS quality
            FROM dedup
        ) TO ?
        (FORMAT PARQUET)
        """,
        [str(processed_path)],
    )
    con.close()
    print(f"[preprocessing] Dataset processado salvo em: {processed_path}")
    print(
        "[preprocessing] Classes: 0=Ruim (quality<=5)  1=Médio (quality=6)  2=Bom (quality>=7)"
    )


if __name__ == "__main__":
    preprocess()
