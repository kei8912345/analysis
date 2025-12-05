# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.signal as signal
import pickle

class STFTAnalyzer:
    """
    SensorDataに対してSTFTを行う。
    """
    def __init__(self):
        pass

    def process(self, data_store, spec_config, output_root_dir):
        """
        Args:
            data_store (dict): {name: SensorData}
        """
        stft_conf = spec_config.get('analysis', {}).get('stft', {})
        if not stft_conf: return None

        settings = stft_conf.get('settings', {})
        targets = stft_conf.get('targets', [])
        
        if not targets: return None

        window_type = settings.get('window', 'hann')
        nperseg = settings.get('nperseg', 512)
        noverlap = settings.get('noverlap', 256)
        
        save_dir = os.path.join(output_root_dir, ".cache", "stft")
        os.makedirs(save_dir, exist_ok=True)
        results = {}

        print(f"\n🌊 [STFT] 解析開始 (Win:{nperseg}, Overlap:{noverlap})")

        for name in targets:
            if name not in data_store:
                print(f"⚠️ [STFT] データなし: {name}")
                continue

            sensor = data_store[name]
            sig = sensor.data
            fs = sensor.fs
            
            # STFT計算
            f, t, Zxx = signal.stft(
                sig, fs=fs, window=window_type, 
                nperseg=nperseg, noverlap=noverlap, detrend='constant'
            )
            
            t_abs = t + sensor.start_time
            amp = np.abs(Zxx)

            # --- ★追加: ピーク周波数と強度の抽出 ---
            # 各時間ステップ(列)ごとに、最大値を持つインデックスを探す
            max_indices = np.argmax(amp, axis=0)
            
            # インデックスを周波数に変換
            peak_freqs = f[max_indices]
            
            # その周波数の強度を取得 (対数dB変換しておく)
            # fancy indexing: [行インデックス配列, 列インデックス配列]
            peak_powers = 20 * np.log10(amp[max_indices, np.arange(amp.shape[1])] + 1e-9)

            results[name] = {
                'f': f,
                't': t_abs,
                'Zxx': Zxx,
                'Amp': amp,
                'fs': fs,
                'unit': sensor.unit,
                # 解析結果として保存
                'peak_freq': peak_freqs,   # 時系列: 周波数 [Hz]
                'peak_power': peak_powers, # 時系列: 強度 [dB]
                'dt_stft': t[1] - t[0] if len(t) > 1 else 0 # 時間刻み
            }
            print(f"    ✅ {name}: {len(t)} steps (fs={fs:.0f}Hz) -> Peak Trace Extracted")

        shot_num = spec_config.get('shot_number', 0)
        save_path = os.path.join(save_dir, f"shot{shot_num:03d}_stft.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 [STFT] 保存: {save_path}")
        return save_path