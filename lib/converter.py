# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pickle
from structs import SensorData

class DataConverter:
    def __init__(self):
        pass

    # ★修正: default_start_time を受け取るように変更
    def process(self, csv_path, output_dir, sensor_configs, processing_config, default_sampling_rate=None, default_start_time=0.0):
        """
        CSVを読み込み、SensorDataの辞書に変換して保存する。
        """
        file_name = os.path.basename(csv_path)
        base_name = os.path.splitext(file_name)[0]
        
        print(f"  [Converter] CSV読み込み中: {file_name}")
        
        try:
            raw_df = self._smart_load_csv(csv_path)
            if raw_df is None: return None
        except Exception as e:
            print(f"  [エラー] CSV読み込み失敗: {e}")
            return None

        converted_dict = {}

        # --- 物理量変換ループ ---
        for sensor in sensor_configs:
            col_id = sensor.get('id')
            name = sensor.get('name')
            
            # カラム特定
            target_col = None
            if col_id in raw_df.columns:
                target_col = col_id
            else:
                clean_cols = {c.strip(): c for c in raw_df.columns}
                if col_id.strip() in clean_cols:
                    target_col = clean_cols[col_id.strip()]
            
            if target_col is None:
                continue

            # 係数計算
            max_phys = sensor.get('max_pressure') or sensor.get('max_phys', 1.0)
            volt_range = sensor.get('range') or sensor.get('max_volt', 10.0)
            slope = max_phys / volt_range if volt_range != 0 else 0.0
            offset = sensor.get('offset', 0.0)
            
            # 数値変換
            raw_val = pd.to_numeric(raw_df[target_col], errors='coerce').fillna(0.0).values
            
            # 物理量計算
            phys_data = raw_val * slope + offset

            # メタデータ取得
            fs = float(sensor.get('sampling_rate', default_sampling_rate))
            unit = sensor.get('unit', '')
            
            # ★修正: ここで start_time を正しくセット
            s_data = SensorData(
                name=name,
                data=phys_data,
                fs=fs,
                unit=unit,
                start_time=default_start_time, # ← yamlの値を反映
                source=file_name
            )
            converted_dict[name] = s_data

        if not converted_dict:
            print(f"  [警告] 有効な列が見つかりませんでした: {file_name}")
            return None

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(converted_dict, f)
            
        print(f"  [Converter] 変換保存完了: {save_path} ({len(converted_dict)} channels, t0={default_start_time})")
        return save_path

    def _smart_load_csv(self, path):
        """ヘッダー位置固定で読み込み"""
        try:
            # 3行目(index 2)ヘッダー
            df = pd.read_csv(path, header=2, low_memory=False)
            # 11行目(index 7)からデータ開始
            if len(df) > 7:
                df = df.iloc[7:].reset_index(drop=True)
                return df
            return None
        except Exception:
            return None