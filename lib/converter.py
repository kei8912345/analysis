# -*- coding: utf-8 -*-
import pandas as pd
import os

class DataConverter:
    def __init__(self):
        pass

    def process(self, csv_path, output_dir, sensor_configs, processing_config, default_sampling_rate=None):
        """
        CSVを読み込み、変換して保存する。
        """
        file_name = os.path.basename(csv_path)
        base_name = os.path.splitext(file_name)[0]
        
        print(f"  [Converter] CSV読み込み中: {file_name}")
        
        try:
            # スマートロード実行 (ヘッダー検出 + 中間メタデータの除去)
            raw_df = self._smart_load_csv(csv_path)
            if raw_df is None:
                return None
                
        except Exception as e:
            print(f"  [エラー] CSVの読み込みに失敗しました: {e}")
            return None

        converted_data = {}
        metadata_dict = {}

        # --- 物理量変換ループ ---
        for sensor in sensor_configs:
            col_id = sensor.get('id')
            name = sensor.get('name')
            
            # カラム存在チェック
            target_col = None
            if col_id in raw_df.columns:
                target_col = col_id
            else:
                clean_cols = {c.strip(): c for c in raw_df.columns}
                if col_id.strip() in clean_cols:
                    target_col = clean_cols[col_id.strip()]
            
            if target_col is None:
                continue

            max_phys = sensor.get('max_pressure') or sensor.get('max_phys', 1.0)
            volt_range = sensor.get('range') or sensor.get('max_volt', 10.0)
            slope = max_phys / volt_range if volt_range != 0 else 0.0
            
            # 文字列として読み込まれている場合の対策 (pd.to_numeric)
            # ここでエラーになる値（メタデータの残りなど）は NaN (0.0) になる
            raw_val = pd.to_numeric(raw_df[target_col], errors='coerce').fillna(0.0)
            
            val = raw_val * slope
            offset = sensor.get('offset', 0.0)
            converted_data[name] = val + offset

            s_rate = sensor.get('sampling_rate', default_sampling_rate)
            metadata_dict[name] = {
                'unit': sensor.get('unit', ''),
                'sampling_rate': s_rate,
                'original_col': col_id
            }

        if not converted_data:
            print(f"  [警告] 指定された列が見つかりませんでした: {file_name}")
            return None

        out_df = pd.DataFrame(converted_data)
        out_df.attrs = metadata_dict

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        save_fmt = processing_config.get('save_format', 'pkl').lower()

        if save_fmt == 'csv':
            save_path = os.path.join(output_dir, f"{base_name}_converted.csv")
            out_df.to_csv(save_path, index=False)
            print(f"  [Converter] CSV保存完了: {save_path}")
        else:
            save_path = os.path.join(output_dir, f"{base_name}.pkl")
            out_df.to_pickle(save_path)
            print(f"  [Converter] PKL保存完了: {save_path}")

        return save_path

    def _smart_load_csv(self, path):
        """
        ユーザー指定の固定フォーマットで読み込む
        ヘッダー: 3行目 (index 2)
        データ開始: 11行目 (間の7行はメタデータとして無視)
        """
        try:
            # header=2 で3行目をヘッダーとして読み込む
            # これにより 1~2行目は無視され、4行目以降がデータ候補としてdfに入る
            df = pd.read_csv(path, header=2, low_memory=False)
            
            # 4行目(index 0) ～ 10行目(index 6) はメタデータ（BlockSizeなど）なので削除
            # データは11行目(index 7)から始まる
            if len(df) > 7:
                df = df.iloc[7:]
                df.reset_index(drop=True, inplace=True)
                print(f"  [Converter] フォーマット固定読み込み: データ開始11行目 (メタデータ7行削除)")
                return df
            else:
                print(f"  [警告] データ行が足りません (11行未満)")
                return None

        except Exception as e:
            print(f"  [警告] 読み込み失敗 ({e})")
            return None