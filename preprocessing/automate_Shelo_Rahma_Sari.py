from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(
    input_path: str | Path,
    output_path: str | Path,
    drop_duplicates: bool = True,
    handle_outlier_age: bool = True,
    bin_age: bool = True,
) -> Path:
    """
    Melakukan preprocessing dataset Bank Direct Marketing Campaigns.

    Tahapan preprocessing (sesuai template MSML & rubric):
    1) Pemeriksaan dan penanganan data kosong (missing values)
    2) Menghapus data duplikat
    3) Deteksi dan penanganan outlier (IQR pada fitur age) - opsional
    4) Binning usia (age_group) - opsional, fitur age asli dihapus
    5) Encoding target y (yes/no → 1/0)
    6) Encoding fitur kategorikal (one-hot encoding)
    7) Standarisasi fitur numerik
    8) Menyimpan dataset hasil preprocessing ke file CSV
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    # Membaca dataset mentah
    df = pd.read_csv(input_path)

    # 1) Pemeriksaan missing values
    # Dataset ini umumnya tidak memiliki missing values,
    # sehingga tidak dilakukan imputasi secara eksplisit
    _ = df.isnull().sum()

    # 2) Menghapus data duplikat
    if drop_duplicates:
        df = df.drop_duplicates()

    # 3) Encoding target terlebih dahulu agar tidak ikut ter-encode
    if "y" not in df.columns:
        raise KeyError("Kolom target 'y' tidak ditemukan dalam dataset.")
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    if df["y"].isnull().any():
        raise ValueError("Kolom target 'y' mengandung nilai selain 'yes' atau 'no'.")

    # 4) Penanganan outlier pada fitur age menggunakan metode IQR
    if handle_outlier_age and "age" in df.columns:
        q1 = df["age"].quantile(0.25)
        q3 = df["age"].quantile(0.75)
        iqr = q3 - q1
        batas_bawah = q1 - 1.5 * iqr
        batas_atas = q3 + 1.5 * iqr
        df = df[(df["age"] >= batas_bawah) & (df["age"] <= batas_atas)]

    # 5) Binning usia
    # Jika binning aktif, fitur age numerik dihapus
    if bin_age and "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 100],
            labels=["Young", "Adult", "Senior", "Elder"],
            include_lowest=True,
        )
        df = df.drop(columns=["age"])

    # Memisahkan fitur dan target
    X = df.drop(columns=["y"])
    y = df["y"]

    # 6) One-hot encoding untuk fitur kategorikal
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 7) Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # 8) Menyimpan dataset hasil preprocessing
    processed_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    processed_df["y"] = y.values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    return output_path


def parse_args() -> argparse.Namespace:
    """
    Mengatur argumen command line untuk menjalankan preprocessing otomatis.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing otomatis dataset Bank Marketing"
    )
    parser.add_argument(
        "--input",
        default="Bank Marketing Data Set_raw/bank-direct-marketing-campaigns.csv",
        help="Path ke dataset mentah (default: dataset_raw/bank-direct-marketing-campaigns.csv)",
    )
    parser.add_argument(
        "--output",
        default="preprocessing/bank_marketing_preprocessed.csv",
        help="Path untuk menyimpan dataset hasil preprocessing",
    )
    parser.add_argument(
        "--no-drop-duplicates",
        action="store_true",
        help="Nonaktifkan penghapusan data duplikat",
    )
    parser.add_argument(
        "--no-outlier-age",
        action="store_true",
        help="Nonaktifkan penanganan outlier pada fitur age",
    )
    parser.add_argument(
        "--no-bin-age",
        action="store_true",
        help="Nonaktifkan binning usia (age_group)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = preprocess(
        input_path=args.input,
        output_path=args.output,
        drop_duplicates=not args.no_drop_duplicates,
        handle_outlier_age=not args.no_outlier_age,
        bin_age=not args.no_bin_age,
    )
    print(f"✅ Preprocessing selesai. Dataset tersimpan di: {output_file}")
