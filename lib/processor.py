# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import signal
from structs import SensorData
import copy

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
            
            if method != 'copy_channel':
                valid_targets = [t for t in targets if t in data_store]
                if not valid_targets: continue
            else:
                valid_targets = targets

            if method == 'moving_average':
                window = proc.get('window', 10)
                center = proc.get('center', True)
                self._apply_moving_average(data_store, valid_targets, window, center)

            elif method == 'moving_median':
                # ★追加: 移動中央値フィルタ
                window = proc.get('window', 11)
                center = proc.get('center', True)
                self._apply_moving_median(data_store, valid_targets, window, center)

            elif method == 'lowpass_filter':
                cutoff = proc.get('cutoff_freq', 10.0)
                order = proc.get('order', 2)
                self._apply_lowpass_filter(data_store, valid_targets, cutoff, order)

            elif method == 'offset_correction':
                time_range = proc.get('range', [-1.0, 0.0])
                target_val = proc.get('target_value', 0.0)
                self._apply_offset_correction(data_store, valid_targets, time_range, target_val)

            elif method == 'polynomial_fit':
                degree = proc.get('degree', 5)
                self._apply_polynomial_fit(data_store, valid_targets, degree)
            
            elif method == 'copy_channel':
                suffix = proc.get('suffix', '_Raw')
                self._apply_copy_channel(data_store, valid_targets, suffix)

            elif method == 'savgol_filter':
                window_length = proc.get('window_length', 51)
                polyorder = proc.get('polyorder', 3)
                self._apply_savgol_filter(data_store, valid_targets, window_length, polyorder)
            
        return data_store

    def _apply_moving_average(self, data_store, targets, window, center):
        for name in targets:
            sensor = data_store[name]
            raw_data = sensor.data
            smoothed = pd.Series(raw_data).rolling(window=window, center=center, min_periods=1).mean().values
            sensor.data = smoothed
        c_str = "Center" if center else "Backward"
        print(f"    🔄 移動平均: win={window} ({c_str}), targets={targets}")

    def _apply_moving_median(self, data_store, targets, window, center):
        """
        移動中央値フィルタ: スパイクノイズや量子化ノイズに強く、エッジを保存する
        """
        for name in targets:
            sensor = data_store[name]
            raw_data = sensor.data
            # Pandasのrolling().median()を使用
            smoothed = pd.Series(raw_data).rolling(window=window, center=center, min_periods=1).median().values
            sensor.data = smoothed
        c_str = "Center" if center else "Backward"
        print(f"    🧱 移動中央値: win={window} ({c_str}), targets={targets}")

    def _apply_lowpass_filter(self, data_store, targets, cutoff_freq, order):
        for name in targets:
            sensor = data_store[name]
            data = sensor.data
            fs = sensor.fs
            nyq = 0.5 * fs
            normal_cutoff = cutoff_freq / nyq
            if normal_cutoff >= 1.0:
                print(f"    ⚠️ LPFスキップ: カットオフ周波数がナイキスト周波数を超えています ({name})")
                continue
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            if len(data) > 3 * max(len(a), len(b)):
                try:
                    y = signal.filtfilt(b, a, data)
                    sensor.data = y
                except Exception as e:
                    print(f"    ⚠️ LPFエラー ({name}): {e}")
            else:
                print(f"    ⚠️ LPFスキップ: データ点数が不足しています ({name})")
        print(f"    📉 LPF適用: fc={cutoff_freq}Hz, order={order}, targets={targets}")

    def _apply_offset_correction(self, data_store, targets, time_range, target_val):
        print(f"    ⚖️ オフセット補正 (基準範囲: {time_range} s, Target: {target_val})")
        for name in targets:
            sensor = data_store[name]
            t = sensor.time
            data = sensor.data
            mask = (t >= time_range[0]) & (t <= time_range[1])
            if np.any(mask):
                current_mean = np.mean(data[mask])
                correction = target_val - current_mean
                sensor.data = data + correction
                print(f"      -> {name}: {current_mean:.4f} -> {target_val:.4f} (Correction: {correction:+.4f})")
            else:
                print(f"      ⚠️ {name}: 指定範囲にデータがないため補正スキップ")

    def _apply_polynomial_fit(self, data_store, targets, degree):
        print(f"    📐 多項式近似 (Degree: {degree}, targets={targets})")
        for name in targets:
            sensor = data_store[name]
            t = sensor.time
            y = sensor.data
            coeffs = np.polyfit(t, y, degree)
            poly_func = np.poly1d(coeffs)
            sensor.data = poly_func(t)

    def _apply_copy_channel(self, data_store, targets, suffix):
        print(f"    ©️ チャンネル複製 (Suffix: '{suffix}')")
        for name in targets:
            if name not in data_store: continue
            original = data_store[name]
            new_name = f"{name}{suffix}"
            new_sensor = copy.deepcopy(original)
            new_sensor.name = new_name
            new_sensor.source = f"Copied from {name}"
            data_store[new_name] = new_sensor
            print(f"      -> {name} copied to {new_name}")

    def _apply_savgol_filter(self, data_store, targets, window_length, polyorder):
        print(f"    ✨ Savitzky-Golayフィルタ (Window: {window_length}, PolyOrder: {polyorder}, targets={targets})")
        for name in targets:
            sensor = data_store[name]
            if window_length % 2 == 0:
                window_length += 1
            try:
                sensor.data = signal.savgol_filter(sensor.data, window_length, polyorder)
            except Exception as e:
                print(f"    ⚠️ S-G Filterエラー ({name}): {e}")