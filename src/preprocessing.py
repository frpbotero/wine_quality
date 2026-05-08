from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "wine_quality.csv"  # ← saída da ingestão (com 'type' e colunas normalizadas)
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"


def preprocess(
    raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH
) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(
        "CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv_auto(?)",
        [str(raw_path)],
    )

    # Detectar schema: normalizado (Supabase) ou UCI original (com espaços)
    col_names = [r[0].lower() for r in con.execute("DESCRIBE raw").fetchall()]
    is_normalized = "fixed_acidity" in col_names  # já normalizado (saída da ingestão)

    if is_normalized:
        # Schema normalizado: colunas já com underscore, pode ter 'id' extra
        sql_select = """
            SELECT
                CAST(fixed_acidity AS DOUBLE)        AS fixed_acidity,
                CAST(volatile_acidity AS DOUBLE)     AS volatile_acidity,
                CAST(citric_acid AS DOUBLE)           AS citric_acid,
                CAST(residual_sugar AS DOUBLE)        AS residual_sugar,
                CAST(chlorides AS DOUBLE)             AS chlorides,
                CAST(free_sulfur_dioxide AS DOUBLE)   AS free_sulfur_dioxide,
                CAST(total_sulfur_dioxide AS DOUBLE)  AS total_sulfur_dioxide,
                CAST(density AS DOUBLE)               AS density,
                CAST(ph AS DOUBLE)                    AS ph,
                CAST(sulphates AS DOUBLE)             AS sulphates,
                CAST(alcohol AS DOUBLE)               AS alcohol,
                CAST(type AS VARCHAR)                 AS type,
                CASE WHEN CAST(quality AS INTEGER) < 7 THEN 0 ELSE 1 END AS quality
            FROM raw
            WHERE quality IS NOT NULL
        """
    else:
        # Schema UCI original: colunas com espaços, sem 'type' ou com 'type'
        type_col = '"type"' if "type" in col_names else "NULL"
        sql_select = f"""
            SELECT
                CAST("fixed acidity"          AS DOUBLE) AS fixed_acidity,
                CAST("volatile acidity"        AS DOUBLE) AS volatile_acidity,
                CAST("citric acid"             AS DOUBLE) AS citric_acid,
                CAST("residual sugar"          AS DOUBLE) AS residual_sugar,
                CAST(chlorides                 AS DOUBLE) AS chlorides,
                CAST("free sulfur dioxide"     AS DOUBLE) AS free_sulfur_dioxide,
                CAST("total sulfur dioxide"    AS DOUBLE) AS total_sulfur_dioxide,
                CAST(density                   AS DOUBLE) AS density,
                CAST("pH"                      AS DOUBLE) AS ph,
                CAST(sulphates                 AS DOUBLE) AS sulphates,
                CAST(alcohol                   AS DOUBLE) AS alcohol,
                CAST({type_col}                AS VARCHAR) AS type,
                CASE WHEN CAST(quality AS INTEGER) < 7 THEN 0 ELSE 1 END AS quality
            FROM raw
            WHERE quality IS NOT NULL
        """

    con.execute(f"COPY ({sql_select}) TO ? (FORMAT PARQUET)", [str(processed_path)])
    con.close()
    print(f"[preprocessing] Schema detectado: {'normalizado' if is_normalized else 'UCI original'}")
    print(f"[preprocessing] Dataset processado salvo em: {processed_path}")
    print("[preprocessing] Classes:  0=Ruim (<7), 1=Bom (>=7)")
    print("[preprocessing] Features: 11 numéricas + 1 categórica (type: red/white)")


if __name__ == "__main__":
    preprocess()
