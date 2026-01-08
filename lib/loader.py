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

        # 1. 各ソースからのデータロード (CSV, HSC等)
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
                loaded_keys = list(loaded_dict.keys())
                print(f"  -> 結合: {len(loaded_dict)} items from {source_name}")

        # 2. STFT解析結果のロード
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

        # 3. 前処理 (Pre-processing)
        # 物理量計算の前に、圧力のオフセット補正などが必要なためここで実行
        pre_pipeline = spec_config.get('pre_processing', [])
        if pre_pipeline: 
            self.processor.apply_preprocessing(data_store, pre_pipeline)

        # 4. 物理量計算 (Derived Channels) ★ここを修正してキャッシュ対応
        derived = spec_config.get('derived_channels', {})
        if derived:
            # キャッシュ保存先の設定
            derived_cache_dir = os.path.join(self.results_root, ".cache", "derived")
            os.makedirs(derived_cache_dir, exist_ok=True)
            derived_pkl_path = os.path.join(derived_cache_dir, f"shot{shot_number:03d}_derived.pkl")
            
            loaded_derived = False
            
            # キャッシュ読み込みトライ
            if not force_reload and os.path.exists(derived_pkl_path):
                try:
                    with open(derived_pkl_path, 'rb') as f:
                        derived_data = pickle.load(f)
                        data_store.update(derived_data)
                        print(f"  -> 派生物理量(キャッシュ)結合: {len(derived_data)} items")
                        loaded_derived = True
                except Exception as e:
                    print(f"  ⚠️ 派生量キャッシュロード失敗: {e}")

            # キャッシュがない、またはロード失敗なら計算して保存
            if not loaded_derived:
                keys_before = set(data_store.keys())
                
                # 計算実行 (data_storeに直接追加される)
                self.physics.add_derived_channels(data_store, derived)
                
                keys_after = set(data_store.keys())
                new_keys = keys_after - keys_before
                
                # 新しく増えたデータだけを保存
                if new_keys:
                    derived_data_to_save = {k: data_store[k] for k in new_keys}
                    try:
                        with open(derived_pkl_path, 'wb') as f:
                            pickle.dump(derived_data_to_save, f)
                        print(f"  💾 派生物理量を保存: {derived_pkl_path} ({len(new_keys)} items)")
                    except Exception as e:
                        print(f"  ⚠️ 派生量保存失敗: {e}")

        # 5. 後処理 (Post-processing)
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