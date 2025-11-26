# -*- coding: utf-8 -*-

class DataProcessor:
    """
    データ前処理エンジン。
    メモリ上のDataFrameに対して、平滑化、リサンプリング、フィルタリングなどの
    信号処理を動的に適用する。
    """
    def __init__(self):
        pass

    def apply_preprocessing(self, df, processing_list):
        """
        ロード済みDataFrameに対して、指定された前処理リストを順次適用する。
        
        Args:
            df (pd.DataFrame): 処理対象のDataFrame
            processing_list (list): 処理内容の辞書リスト (plans.yaml または spec.yaml から供給)
            
        Returns:
            pd.DataFrame: 処理後のDataFrame
        """
        if not processing_list:
            return df
        
        print("  [Processor] 前処理パイプラインを実行中...")
        
        for proc in processing_list:
            method = proc.get('method')
            targets = proc.get('targets', [])
            
            # データフレームに存在する列だけを処理対象にする
            valid_targets = [t for t in targets if t in df.columns]
            
            if not valid_targets:
                # 指定された列が一つもない場合はスキップ（ログは冗長になるので出さないか、デバッグレベルで出す）
                continue

            if method == 'moving_average':
                window = proc.get('window', 10)
                center = proc.get('center', True)
                self._apply_moving_average(df, valid_targets, window, center)
            
            elif method == 'resample':
                # 将来実装: ダウンサンプリング処理
                print(f"    ⚠️ 未実装のメソッドです: {method}")
                pass
            
            elif method == 'lowpass_filter':
                # 将来実装: バターワースフィルタなど
                print(f"    ⚠️ 未実装のメソッドです: {method}")
                pass
            
            else:
                print(f"    ⚠️ 不明な前処理メソッドです: {method}")

        return df

    def _apply_moving_average(self, df, targets, window, center):
        """
        移動平均フィルタを適用し、元の列を上書きする。
        
        Args:
            df (pd.DataFrame): データフレーム
            targets (list): 適用する列名のリスト
            window (int): 窓幅（データ点数）
            center (bool): 窓を中央に配置するかどうか
        """
        for col in targets:
            # pandasのrollingメソッドを使用
            # min_periods=1 により、データの端でもNaNにならずに計算値を返す
            df[col] = df[col].rolling(window=window, center=center, min_periods=1).mean()
        
        center_str = "中央" if center else "後方"
        print(f"    🔄 移動平均適用: 窓幅={window} ({center_str}), 対象={targets}")

