# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.signal as signal
import pickle

class STFTAnalyzer:
    """
    時系列データに対して短時間フーリエ変換(STFT)を行い、
    時間-周波数解析結果を出力するクラス。
    
    【修正点】
    データフレームのインデックス（時間）を無視せず、
    指定されたサンプリングレートで厳密にリサンプリングを行ってから
    STFTを計算するように変更しました。これにより時間軸の伸長を防ぎます。
    """
    def __init__(self):
        pass

    def process(self, df, spec_config, output_root_dir, default_sampling_rate=None):
        """
        メイン処理関数
        
        Args:
            df (pd.DataFrame): 解析対象の時系列データ (IndexはTimeであること)
            spec_config (dict): spec.yamlの内容
            output_root_dir (str): 保存先ルート
            default_sampling_rate (float): 全体のデフォルトサンプリング周波数
        """
        
        # 設定の読み込み
        stft_conf = spec_config.get('analysis', {}).get('stft', {})
        if not stft_conf:
            print("❌ [STFT] Specファイルに 'analysis.stft' 設定がありません。")
            return None

        settings = stft_conf.get('settings', {})
        targets = stft_conf.get('targets', [])
        
        if not targets:
            print("⚠️ [STFT] 解析対象(targets)が指定されていません。")
            return None

        # パラメータ展開
        window_type = settings.get('window', 'hann')
        nperseg = settings.get('nperseg', 512)
        noverlap = settings.get('noverlap', 256)
        
        # 保存先の準備
        save_dir = os.path.join(output_root_dir, "stft_results")
        os.makedirs(save_dir, exist_ok=True)

        results = {}
        
        # 時間軸情報の取得
        if 'Time' in df.columns and df.index.name != 'Time':
            # もしTimeが列にあってIndexでない場合の保険
            df = df.set_index('Time')
        
        # DataFrameのインデックスが時間軸であることを前提とします
        # インデックスが数値型でない場合はエラーになる可能性があります
        t_start = df.index.min()
        t_end = df.index.max()
        
        print(f"\n🌊 [STFT] 周波数解析を開始します...")
        print(f"⚙️  [設定] Window: {window_type}, Length: {nperseg}, Overlap: {noverlap}")
        
        measurements = spec_config.get('measurements', [])
        meas_dict = {m.get('name'): m for m in measurements}

        for col in targets:
            if col not in df.columns:
                print(f"⚠️ [STFT] カラム '{col}' がデータに存在しません。スキップします。")
                continue
            
            # --- サンプリングレートの特定 ---
            fs_target = default_sampling_rate
            
            if col in meas_dict:
                m_info = meas_dict[col]
                if 'fps' in m_info:
                    fs_target = float(m_info['fps'])
                elif 'sampling_rate' in m_info:
                    fs_target = float(m_info['sampling_rate'])
            
            if fs_target is None:
                print(f"⚠️ [STFT] '{col}' のサンプリングレートが特定できません。スキップします。")
                continue

            source = "Default"
            if col in meas_dict:
                if 'fps' in meas_dict[col]: source = "Spec(fps)"
                elif 'sampling_rate' in meas_dict[col]: source = "Spec(sampling_rate)"
            
            print(f"    🎯 解析対象: {col} (Target fs={fs_target}Hz)")

            # === 重要な修正箇所 ===
            # DataFrameの生データをそのまま使うのではなく、
            # 指定された fs_target に基づいて新しい時間軸を作成し、データをリサンプリングします。
            
            # 1. 理想的な時間軸を作成
            #    t_start から t_end まで、dt = 1/fs_target で刻む
            expected_times = np.arange(t_start, t_end, 1.0 / fs_target)
            
            # 2. 補間 (Resampling)
            #    df[col]には欠損(NaN)が含まれる可能性があるため、まずは元データ内で補間
            #    その後、新しい時間軸(expected_times)に合わせて値を拾う
            valid_series = df[col].interpolate(limit_direction='both').fillna(0)
            
            # numpy.interp を使用して、dfの実際のIndex(時間)に基づいて値を再サンプリング
            # これにより、データの疎密に関わらず、等間隔かつ正しい長さの配列(sig)が得られます
            sig = np.interp(expected_times, df.index.values, valid_series.values)
            
            print(f"       -> リサンプリング実行: {len(df)} rows -> {len(sig)} samples (Source: {source})")
            
            # STFT計算
            f, t, Zxx = signal.stft(
                sig, 
                fs=fs_target, 
                window=window_type, 
                nperseg=nperseg, 
                noverlap=noverlap,
                detrend='constant' 
            )
            
            # 時間軸の補正（絶対時間に直す）
            t_abs = t + t_start
            
            # 振幅スペクトル
            amp = np.abs(Zxx)

            results[col] = {
                'f': f,
                't': t_abs,
                'Zxx': Zxx,
                'Amp': amp,
                'params': settings,
                'fs': fs_target
            }
            
            print(f"       ✅ 計算完了 (Time steps: {len(t)}, Freq steps: {len(f)})")

        # 保存
        shot_num = spec_config.get('shot_number', 0)
        save_name = f"shot{shot_num:03d}_stft.pkl"
        save_path = os.path.join(save_dir, save_name)
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 [STFT] 保存完了: {save_path}")
            return save_path
        except Exception as e:
            print(f"❌ [STFT] 保存エラー: {e}")
            return None