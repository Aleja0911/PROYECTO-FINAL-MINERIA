"""Carga, valida y consolida las hojas de Online Retail II."""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = ROOT_DIR / "02_DATOS" / "online_retail_II.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_FILE = OUTPUT_DIR / "online_retail_II_consolidado.csv"
REPORT_FILE = OUTPUT_DIR / "online_retail_II_validacion.json"

EXPECTED_SHEETS = ("Year 2009-2010", "Year 2010-2011")
EXPECTED_COLUMNS = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)
PERIOD_BY_SHEET = {
    "Year 2009-2010": "2009-2010",
    "Year 2010-2011": "2010-2011",
}


def main():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {SOURCE_FILE}")

    sheets = pd.read_excel(SOURCE_FILE, sheet_name=None, engine="openpyxl")
    actual_sheets = tuple(sheets)
    if actual_sheets != EXPECTED_SHEETS:
        raise ValueError(
            f"Nombres u orden de hojas inesperados: {actual_sheets}; "
            f"se esperaban {EXPECTED_SHEETS}"
        )

    frames = []
    sheet_report = {}
    for sheet_name, frame in sheets.items():
        if tuple(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Columnas inesperadas en '{sheet_name}': {tuple(frame.columns)}"
            )

        print(f"\nHoja: {sheet_name}")
        print(f"Dimensiones (filas de datos, columnas): {frame.shape}")
        print("Tipos de datos:")
        print(frame.dtypes.to_string())
        print("Registros de ejemplo:")
        print(frame.head(3).to_string(index=False))

        sheet_report[sheet_name] = {
            "dataframe_dimensions": [int(frame.shape[0]), int(frame.shape[1])],
            "worksheet_dimensions_including_header": [
                int(frame.shape[0] + 1),
                int(frame.shape[1]),
            ],
            "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "example_records": json.loads(
                frame.head(3).to_json(
                    orient="records",
                    date_format="iso",
                    force_ascii=False,
                )
            ),
        }

        frame = frame.copy()
        frame["Period"] = PERIOD_BY_SHEET[sheet_name]
        frame["SourceSheet"] = sheet_name
        frames.append(frame)

    consolidated = pd.concat(frames, ignore_index=True)
    expected_rows = sum(len(frame) for frame in frames)
    if len(consolidated) != expected_rows:
        raise RuntimeError(
            f"El consolidado tiene {len(consolidated)} filas; "
            f"se esperaban {expected_rows}"
        )
    if consolidated[["Period", "SourceSheet"]].isna().any().any():
        raise RuntimeError("Hay valores faltantes en las columnas de procedencia.")

    source_counts = consolidated["SourceSheet"].value_counts().to_dict()
    expected_source_counts = {
        sheet_name: len(sheets[sheet_name]) for sheet_name in EXPECTED_SHEETS
    }
    if source_counts != expected_source_counts:
        raise RuntimeError(
            f"Los conteos por hoja no coinciden: {source_counts}; "
            f"se esperaban {expected_source_counts}"
        )

    consolidated.to_csv(
        CONSOLIDATED_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    report = {
        "source_file": str(SOURCE_FILE.relative_to(ROOT_DIR)),
        "sheet_names": list(actual_sheets),
        "sheets": sheet_report,
        "consolidated_dimensions": [
            int(consolidated.shape[0]),
            int(consolidated.shape[1]),
        ],
        "consolidated_columns": list(consolidated.columns),
        "records_by_source_sheet": {
            sheet_name: int(source_counts[sheet_name])
            for sheet_name in EXPECTED_SHEETS
        },
        "period_by_source_sheet": PERIOD_BY_SHEET,
        "output_file": CONSOLIDATED_FILE.name,
    }
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDimensiones consolidadas (filas, columnas): {consolidated.shape}")
    print("Registros por hoja de origen:")
    print(consolidated["SourceSheet"].value_counts().to_string())
    print(f"\nConsolidado guardado en: {CONSOLIDATED_FILE}")
    print(f"Reporte de validación guardado en: {REPORT_FILE}")


if __name__ == "__main__":
    main()
