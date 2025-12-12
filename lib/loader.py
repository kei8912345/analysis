# -*- coding: utf-8 -*-
import os
import glob
import re
import pickle
import numpy as np

try:
    from .converter import DataConverter
    from .physics import PhysicsEngine
    from .processor import DataProcessor
    from .structs import SensorData
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from converter import DataConverter
    from physics import PhysicsEngine
    from processor import DataProcessor
    from structs import SensorData

class DataLoader:
    TARGET_SOURCES = ['pressure', 'vibration', 'hsc']

    def __init__(self, series_config, output_dir=None):
        self.series_config = series_config
        settings = series_config.get('settings', {})
        self.base_dir = settings.get('base_dir') or series_config.get('base_dir', '.')
        self.sources = series_config.get('sources', {})
        
        self.converter = DataConverter()
        self.physics = PhysicsEngine()
        self.processor = DataProcessor()
        
        self.results_root = output_dir if output_dir else os.path.join(self.base_dir, "033_解析結果")

    def load_shot_data(self, spec_config, force_reload=False):
        shot_number = spec_config['shot_number']
        measurements = spec_config.get('measurements', [])
        processing_config = spec_config.get('processing', {})
        acquisition_config = spec_config.get('acquisition', {})
        
        default_sr = float(acquisition_config.get('sampling_rate', 1000.0))
        start_time_offset = float(acquisition_config.get('start_time', 0.0))
        
        data_store = {}

        for source_name, source_info in self.sources.items():
            if source_name not in self.TARGET_SOURCES: continue

            cache_root = self._get_cache_directory(source_name)
            os.makedirs(cache_root, exist_ok=True)
            
            print(f"[Loader] ソース: '{source_name}'")

            # === HSC ===
            if source_name == 'hsc':
                hsc_pkl_path = os.path.join(cache_root, f"shot{shot_number:03d}_hsc.pkl")
                if os.path.exists(hsc_pkl_path):
                    try:
                        with open(hsc_pkl_path, 'rb') as f:
                            hsc_data = pickle.load(f)
                            if isinstance(hsc_data, dict):
                                data_store.update(hsc_data)
                                print(f"  -> HSCデータ結合: {len(hsc_data)} items")
                    except Exception as e:
                        print(f"  ⚠️ HSCロード失敗: {e}")
                continue

            # === CSV系センサ ===
            folder_name = source_info.get('folder')
            hint = source_info.get('hint', None)
            target_dir = os.path.join(self.base_dir, folder_name)
            
            csv_path = self._smart_find_file(target_dir, shot_number, hint=hint)
            if not csv_path:
                print(f"  ⚠️ ファイルなし: Shot {shot_number} in {folder_name}")
                continue
            
            file_base = os.path.splitext(os.path.basename(csv_path))[0]
            cache_path = os.path.join(cache_root, file_base + ".pkl")

            loaded_dict = None
            if not force_reload and self._is_cache_valid(csv_path, cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        loaded_dict = pickle.load(f)
                except: pass
            
            if loaded_dict is None:
                save_path = self.converter.process(
                    csv_path=csv_path, 
                    output_dir=cache_root, 
                    sensor_configs=measurements, 
                    processing_config=processing_config,
                    default_sampling_rate=default_sr,
                    default_start_time=start_time_offset
                )
                if save_path:
                    with open(save_path, 'rb') as f:
                        loaded_dict = pickle.load(f)

            if loaded_dict and isinstance(loaded_dict, dict):
                data_store.update(loaded_dict)
                # ★修正: 何がロードされたかをログに出す
                loaded_keys = list(loaded_dict.keys())
                print(f"  -> 結合: {len(loaded_dict)} items from {source_name}")
                print(f"     👀 Loaded Keys: {loaded_keys}")

        # --- STFTの解析結果があればロードして時系列データとして統合 ---
        stft_dir = os.path.join(self.results_root, ".cache", "stft")
        stft_pkl = os.path.join(stft_dir, f"shot{shot_number:03d}_stft.pkl")
        if os.path.exists(stft_pkl):
            try:
                with open(stft_pkl, 'rb') as f:
                    stft_res = pickle.load(f)
                    count = 0
                    for key, val in stft_res.items():
                        if 'peak_freq' in val and 't' in val:
                            t_arr = val['t']
                            fs_est = 1.0 / (t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1.0
                            t0 = t_arr[0]
                            
                            new_name = f"{key}_PeakFreq"
                            data_store[new_name] = SensorData(
                                name=new_name,
                                data=val['peak_freq'],
                                fs=fs_est,
                                unit="Hz",
                                start_time=t0,
                                source="STFT_Analysis"
                            )
                            
                            new_name_p = f"{key}_PeakPower"
                            data_store[new_name_p] = SensorData(
                                name=new_name_p,
                                data=val['peak_power'],
                                fs=fs_est,
                                unit="dB",
                                start_time=t0,
                                source="STFT_Analysis"
                            )
                            count += 2
                    if count > 0:
                        print(f"  -> STFT抽出データ結合: {count} items (PeakFreq, PeakPower)")
            except Exception as e:
                print(f"  ⚠️ STFTロード警告: {e}")

        if not data_store:
            print("❌ 有効なデータがロードできませんでした。")
            return {}

        # 後処理
        pre_pipeline = spec_config.get('pre_processing', [])
        if pre_pipeline: self.processor.apply_preprocessing(data_store, pre_pipeline)

        derived = spec_config.get('derived_channels', {})
        self.physics.add_derived_channels(data_store, derived)

        post_pipeline = spec_config.get('post_processing', [])
        if post_pipeline: self.processor.apply_preprocessing(data_store, post_pipeline)

        return data_store

    def _get_cache_directory(self, source_name):
        base_cache_dir = os.path.join(self.results_root, ".cache")
        if source_name == 'vibration': dir_name = "vibration"
        elif source_name == 'hsc': dir_name = "hsc_brightness"
        else: dir_name = source_name
        return os.path.join(base_cache_dir, dir_name)

    def _smart_find_file(self, search_dir, shot_num, hint=None):
        if not os.path.exists(search_dir): return None
        files = glob.glob(os.path.join(search_dir, "*.csv"))
        candidates = [f for f in files if int(shot_num) in [int(n) for n in re.findall(r'\d+', os.path.basename(f))]]
        if not candidates: return None
        if len(candidates) > 1 and hint:
            filtered = [c for c in candidates if hint.lower() in os.path.basename(c).lower()]
            if filtered: return filtered[0]
        return candidates[0]

    def _is_cache_valid(self, source, cache):
        if not os.path.exists(cache): return False
        return os.path.getmtime(source) < os.path.getmtime(cache)