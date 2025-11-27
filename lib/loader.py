# -*- coding: utf-8 -*-
import os
import glob
import re
import pickle

try:
    from .converter import DataConverter
    from .physics import PhysicsEngine
    from .processor import DataProcessor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from converter import DataConverter
    from physics import PhysicsEngine
    from processor import DataProcessor

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
        
        # ★修正: spec.yaml から acquisition 設定を読み込む
        acquisition_config = spec_config.get('acquisition', {})
        # デフォルトサンプリングレート
        default_sr = float(acquisition_config.get('sampling_rate', 1000.0))
        # ★重要: トリガー前時間 (例: -0.1) を取得
        start_time_offset = float(acquisition_config.get('start_time', 0.0))
        
        data_store = {}

        for source_name, source_info in self.sources.items():
            if source_name not in self.TARGET_SOURCES: continue

            cache_root = self._get_cache_directory(source_name)
            os.makedirs(cache_root, exist_ok=True)
            
            print(f"[Loader] ソース: '{source_name}'")

            # === HSC ===
            if source_name == 'hsc':
                # HSCは hsc_analyzer 側で pre_trigger_frames から start_time を計算済み
                hsc_pkl_path = os.path.join(cache_root, f"shot{shot_number:03d}_hsc.pkl")
                
                if os.path.exists(hsc_pkl_path):
                    try:
                        with open(hsc_pkl_path, 'rb') as f:
                            hsc_data = pickle.load(f)
                            if isinstance(hsc_data, dict):
                                data_store.update(hsc_data)
                                print(f"  -> HSCデータ結合: {len(hsc_data)} items")
                            else:
                                print(f"  ⚠️ HSCキャッシュ形式不一致 (スキップ)")
                    except Exception as e:
                        print(f"  ⚠️ HSCロード失敗: {e}")
                else:
                    print(f"  ℹ️  HSCキャッシュなし (未解析): {os.path.basename(hsc_pkl_path)}")
                continue

            # === CSV系センサ (Pressure, Vibration) ===
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
                        temp_data = pickle.load(f)
                        if isinstance(temp_data, dict):
                            # ★キャッシュの start_time が spec と合っているか確認するのは難しいので
                            # specの値で上書きする処理を入れるとより安全だが、今回はConverter再実行で対応
                            loaded_dict = temp_data
                        else:
                            print(f"  🔄 古い形式のキャッシュを検出 -> 再生成します")
                except: pass
            
            if loaded_dict is None:
                # ★修正: start_time と sampling_rate を渡す
                save_path = self.converter.process(
                    csv_path=csv_path, 
                    output_dir=cache_root, 
                    sensor_configs=measurements, 
                    processing_config=processing_config,
                    default_sampling_rate=default_sr,
                    default_start_time=start_time_offset # ← これが重要
                )
                if save_path:
                    with open(save_path, 'rb') as f:
                        loaded_dict = pickle.load(f)

            if loaded_dict and isinstance(loaded_dict, dict):
                data_store.update(loaded_dict)
                print(f"  -> 結合: {len(loaded_dict)} items from {source_name}")

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