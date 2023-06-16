import random
from pathlib import Path
from typing import Tuple

import librosa
import pandas as pd


def train_val_split(
    audio_directory: Path,
    df: pd.DataFrame,
    val_size: float = 0.2,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert "fname" in df.columns and "label" in df.columns
    df = df[["fname", "label"]]
    df["duration"] = df["fname"].apply(
        lambda wav_path: librosa.get_duration(filename=audio_directory / wav_path)
    )

    train_data = list()
    val_data = list()

    for _, label_df in df.groupby(by="label"):
        step = 100

        label_df = label_df.sort_values(by="duration").copy()
        for start_idx in range(0, len(label_df), step):
            data_part = label_df[start_idx : start_idx + step].values.tolist()

            if shuffle:
                data_part = random.sample(data_part, k=len(data_part))
                items_count = len(data_part)
                val_data.extend(data_part[0 : int(items_count * val_size)])
                train_data.extend(data_part[int(items_count * val_size) :])
            else:
                NotImplementedError()

    val_data = random.sample(val_data, k=len(val_data))
    train_data = random.sample(train_data, k=len(train_data))

    val_df = pd.DataFrame(val_data, columns=["fname", "label", "duration"])
    train_df = pd.DataFrame(train_data, columns=["fname", "label", "duration"])
    return train_df, val_df
