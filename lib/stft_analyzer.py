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
        
        # ★変更点: 保存先を .cache/stft に隠す
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
            
            results[name] = {
                'f': f,
                't': t_abs,
                'Zxx': Zxx,
                'Amp': np.abs(Zxx),
                'fs': fs,
                'unit': sensor.unit
            }
            print(f"    ✅ {name}: {len(t)} steps (fs={fs:.0f}Hz)")

        shot_num = spec_config.get('shot_number', 0)
        save_path = os.path.join(save_dir, f"shot{shot_num:03d}_stft.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 [STFT] 保存: {save_path}")
        return save_path