# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from structs import SensorData

class DataProcessor:
    """
    データ前処理エンジン。
    Dict[str, SensorData] に対して処理を行う。
    """
    def __init__(self):
        pass

    def apply_preprocessing(self, data_store, processing_list):
        """
        Args:
            data_store (dict): {name: SensorData}
            processing_list (list): 処理内容の辞書リスト
        """
        if not processing_list: return data_store
        
        print("  [Processor] 前処理パイプラインを実行中...")
        
        for proc in processing_list:
            method = proc.get('method')
            targets = proc.get('targets', [])
            
            # 存在するターゲットのみ抽出
            valid_targets = [t for t in targets if t in data_store]
            if not valid_targets: continue

            if method == 'moving_average':
                window = proc.get('window', 10)
                center = proc.get('center', True)
                self._apply_moving_average(data_store, valid_targets, window, center)
            
            # 他のメソッドが必要ならここに追加
            
        return data_store

    def _apply_moving_average(self, data_store, targets, window, center):
        for name in targets:
            sensor = data_store[name]
            raw_data = sensor.data
            
            # NaN対策: pandasのrollingが便利なので一時的に借用 (速度的にも十分)
            # dataはnumpy配列なので、Series化して計算し、valuesで戻す
            smoothed = pd.Series(raw_data).rolling(window=window, center=center, min_periods=1).mean().values
            
            # 上書き更新
            sensor.data = smoothed
        
        c_str = "Center" if center else "Backward"
        print(f"    🔄 移動平均: win={window} ({c_str}), targets={targets}")