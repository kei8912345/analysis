# -*- coding: utf-8 -*-
import os
import glob
import re
import pandas as pd
import numpy as np
import yaml

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
        
        if output_dir:
            self.results_root = output_dir
        else:
            self.results_root = os.path.join(self.base_dir, "033_解析結果")

    def load_shot_data(self, spec_config, force_reload=False):
        shot_number = spec_config['shot_number']
        measurements = spec_config.get('measurements', [])
        
        processing_config = spec_config.get('processing', {})
        derived_channels = spec_config.get('derived_channels', {})
        pre_calc_pipeline = spec_config.get('pre_processing', [])
        post_calc_pipeline = spec_config.get('post_processing', spec_config.get('preprocessing', []))
        
        acquisition_config = spec_config.get('acquisition', {})
        default_sr = acquisition_config.get('sampling_rate', 1000.0)
        start_time = acquisition_config.get('start_time', 0.0)

        loaded_dfs = []

        # --- Phase 1: 各ソースのロード ---
        for source_name, source_info in self.sources.items():
            if source_name not in self.TARGET_SOURCES:
                continue

            cache_root = self._get_cache_directory(source_name)
            os.makedirs(cache_root, exist_ok=True)

            print(f"[Loader] ソース読み込み処理: '{source_name}'")

            # === HSCデータの読み込み ===
            if source_name == 'hsc':
                hsc_pkl_dir = os.path.join(self.results_root, "hsc_timeseries")
                hsc_pkl_name = f"shot{shot_number:03d}_hsc.pkl"
                hsc_pkl_path = os.path.join(hsc_pkl_dir, hsc_pkl_name)

                if os.path.exists(hsc_pkl_path):
                    print(f"  -> HSC解析済みデータを発見: {hsc_pkl_name}")
                    try:
                        df_hsc = pd.read_pickle(hsc_pkl_path)
                        # HSCは既にTime列を持っているはずなので、インデックスに設定して保持
                        if 'Time' in df_hsc.columns:
                            df_hsc = df_hsc.set_index('Time')
                            # 重複Timeの排除（念のため）
                            df_hsc = df_hsc[~df_hsc.index.duplicated(keep='first')]
                            loaded_dfs.append(df_hsc)
                            print(f"     (Time軸同期: OK, Shape: {df_hsc.shape})")
                        else:
                            print("  ⚠️ HSCデータにTime列がありません。スキップします。")
                    except Exception as e:
                        print(f"  ⚠️ HSCデータの読み込み失敗: {e}")
                else:
                    print(f"  ℹ️  HSCデータが見つかりません (未解析): {hsc_pkl_name}")
                continue

            # === 通常のセンサデータ (Pressure, Vibration) ===
            folder_name = source_info.get('folder')
            hint = source_info.get('hint', None)
            target_dir = os.path.join(self.base_dir, folder_name)

            csv_path = self._smart_find_file(target_dir, shot_number, hint=hint)
            if not csv_path:
                print(f"  -> [警告] Shot {shot_number} のファイルが {source_name} フォルダに見つかりません。")
                continue
            
            file_base = os.path.splitext(os.path.basename(csv_path))[0]
            fmt = processing_config.get('save_format', 'pkl').lower()
            ext = ".csv" if fmt == 'csv' else ".pkl"
            cache_path = os.path.join(cache_root, file_base + ext)

            df_source = None
            if not force_reload and self._is_cache_valid(csv_path, cache_path):
                # print(f"  -> キャッシュを発見しました: {os.path.basename(cache_path)}")
                if fmt == 'csv': df_source = pd.read_csv(cache_path)
                else: df_source = pd.read_pickle(cache_path)
            else:
                print(f"  -> 変換処理を実行します...")
                saved_path = self.converter.process(
                    csv_path=csv_path,
                    output_dir=cache_root,
                    sensor_configs=measurements,
                    processing_config=processing_config,
                    default_sampling_rate=default_sr
                )
                if saved_path:
                    if fmt == 'csv': df_source = pd.read_csv(saved_path)
                    else: df_source = pd.read_pickle(saved_path)

            if df_source is not None and not df_source.empty:
                # --- 時間軸の生成 (Time Index) ---
                # Converterが保存したメタデータ(attrs)からサンプリングレートを取得
                # 複数の列でレートが混在することは稀と仮定し、最初の有効なレートを採用
                current_sr = default_sr
                if hasattr(df_source, 'attrs') and df_source.attrs:
                    for col_meta in df_source.attrs.values():
                        if isinstance(col_meta, dict) and 'sampling_rate' in col_meta:
                            sr_val = col_meta['sampling_rate']
                            if sr_val:
                                current_sr = float(sr_val)
                                break
                
                # 時間配列作成
                n_samples = len(df_source)
                times = (np.arange(n_samples) / current_sr) + start_time
                
                # インデックスに設定
                df_source.index = times
                df_source.index.name = 'Time'
                
                # メタデータを維持しつつリストに追加
                loaded_dfs.append(df_source)
                print(f"  -> ロード完了 ({source_name}): SR={current_sr}Hz, Samples={n_samples}")

        # --- Phase 2: データ統合 (Time軸基準のOuter Join) ---
        if not loaded_dfs:
            raise FileNotFoundError(f"Shot {shot_number} のデータロードに失敗しました (有効なデータソースがありません)。")

        print("  [Loader] データ統合中 (Time基準)...")
        # axis=1 で結合すると、インデックス(Time)に基づいて自動的に整列・結合されます。
        # 存在しない時刻の値は NaN になります。
        integrated_df = pd.concat(loaded_dfs, axis=1)
        
        # Timeをインデックスから列に戻す (以降の処理のため)
        integrated_df = integrated_df.sort_index().reset_index()

        # --- Phase 3以降 ---
        if pre_calc_pipeline:
            print("  [Loader] Pre-Processing...")
            integrated_df = self.processor.apply_preprocessing(integrated_df, pre_calc_pipeline)
        
        # Physics計算 (行ごとに計算されるため、NaNが含まれていてもその行の結果がNaNになるだけで問題なし)
        integrated_df = self.physics.add_derived_channels(integrated_df, derived_channels, sampling_rate=default_sr)
        
        if post_calc_pipeline:
            print("  [Loader] Post-Processing...")
            integrated_df = self.processor.apply_preprocessing(integrated_df, post_calc_pipeline)

        return integrated_df

    def _get_cache_directory(self, source_name):
        base_cache_dir = os.path.join(self.results_root, ".cache")
        if source_name == 'vibration': dir_name = "vibration"
        elif source_name == 'hsc': dir_name = "hsc_brightness"
        else: dir_name = source_name
        return os.path.join(base_cache_dir, dir_name)

    def _smart_find_file(self, search_dir, shot_num, hint=None):
        if not os.path.exists(search_dir):
            return None
        files = glob.glob(os.path.join(search_dir, "*.csv"))
        candidates = []
        for f_path in files:
            fname = os.path.basename(f_path)
            nums_in_name = [int(n) for n in re.findall(r'\d+', fname)]
            if shot_num in nums_in_name:
                candidates.append(f_path)
        if not candidates: return None
        if len(candidates) > 1 and hint:
            filtered = [c for c in candidates if hint.lower() in os.path.basename(c).lower()]
            if filtered: return filtered[0]
        return candidates[0]

    def _is_cache_valid(self, source_path, cache_path):
        if not os.path.exists(cache_path): return False
        return os.path.getmtime(source_path) < os.path.getmtime(cache_path)