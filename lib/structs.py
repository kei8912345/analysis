# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SensorData:
    """
    1つのセンサ（または解析結果）の時系列データを保持するコンテナ。
    Pandas DataFrameの列ではなく、このオブジェクト単位でデータを回します。
    """
    name: str             # データ名 (例: "P_Ch1", "HSC_Brightness")
    data: np.ndarray      # 信号データ本体 (1次元配列)
    fs: float             # サンプリングレート [Hz]
    unit: str = ""        # 単位 (例: "MPa", "K")
    start_time: float = 0.0 # 計測開始時間 [s]
    source: str = ""      # データソース情報 (デバッグ用)

    @property
    def time(self) -> np.ndarray:
        """
        時間軸配列をオンデマンドで生成して返す。
        メモリを節約するため、dataとしては持たない。
        """
        n = len(self.data)
        # t = index / fs + start_time
        if self.fs <= 0:
            return np.zeros(n)
        return (np.arange(n) / self.fs) + self.start_time

    def __repr__(self):
        return f"<SensorData '{self.name}': {len(self.data)} samples, {self.fs:.1f}Hz, Unit='{self.unit}'>"