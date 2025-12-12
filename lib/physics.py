# -*- coding: utf-8 -*-
import numpy as np
from structs import SensorData

class PhysicsEngine:
    """
    物理量の計算を行うクラス。
    """
    GAS_CONSTANTS = {"Air": 287.058, "H2": 4124.0, "N2": 296.8, "O2": 259.8, "Ar": 208.1}
    GAMMAS = {"Air": 1.4, "H2": 1.405, "N2": 1.4, "O2": 1.395, "Ar": 1.667}

    def __init__(self):
        pass

    def add_derived_channels(self, data_store, derived_configs):
        """
        Args:
            data_store (dict): {name: SensorData}
            derived_configs (dict): derived_channels config
        """
        if not data_store or not derived_configs:
            return data_store

        print("  [Physics] 派生物理量の計算...")

        for name, config in derived_configs.items():
            calc_type = config.get('type')
            if calc_type in ['choked_flow', 'nozzle_flow']:
                self._calc_compressible_flow(data_store, name, config)
            
        return data_store

    def _get_data_array(self, data_store, source_key):
        """ソースがキーなら配列を、数値ならその値を返す"""
        if isinstance(source_key, str) and source_key in data_store:
            return data_store[source_key].data
        if isinstance(source_key, (int, float)):
            return float(source_key)
        return None

    def _calc_compressible_flow(self, data_store, target_name, config):
        src_p = config.get('source_p')
        src_t = config.get('source_t')
        
        # 配列(または定数)の取得
        P_raw = self._get_data_array(data_store, src_p)
        T_raw = self._get_data_array(data_store, src_t)

        if P_raw is None or T_raw is None:
            print(f"    ⚠️ 計算スキップ: ソース不足 ({target_name})")
            return

        # 基準となるSensorDataを取得
        ref_sensor = data_store.get(src_p)
        if not ref_sensor and isinstance(src_p, str): return 
        
        # --- パラメータ ---
        gas_type = config.get('gas_type', 'Air')
        Cd = config.get('Cd', 1.0)
        A_mm2 = config.get('A_mm2', 1.0)
        cutoff_ratio = config.get('cutoff_ratio', 1.0) # チョーク判定用ではなく、計算打ち切り用の比率
        
        R = self.GAS_CONSTANTS.get(gas_type, 287.0)
        gamma = self.GAMMAS.get(gas_type, 1.4)
        
        # 背圧の設定 (デフォルトは標準大気圧)
        P_back_MPa = config.get('back_pressure', 0.1013)
        Pb_Pa = P_back_MPa * 1.0e6
        
        # --- ★重要: 自動ゼロ点補正 ---
        # データの先頭部分(初期値)を「背圧と平衡状態にある圧力」とみなして補正する
        # これにより、センサのオフセット誤差や大気圧変動をキャンセルし、
        # P_raw が初期値に戻ったときに確実に流量が0になるようにする。
        
        P0_Pa = None
        
        if isinstance(P_raw, np.ndarray):
            # 初期値の取得 (先頭100点または全データの1%程度)
            n_init = min(100, len(P_raw))
            if n_init > 0:
                p_initial_mean = np.mean(P_raw[:n_init])
                
                # 「計測値 - 初期値」で差圧を出し、それに背圧を足して絶対圧とする
                # P0_abs = (P_measure - P_init) + P_back
                # これなら P_measure == P_init のとき P0_abs == P_back となり、流量は0になる。
                P0_Pa = (P_raw - p_initial_mean) * 1.0e6 + Pb_Pa
                
                # 念のため負圧防止 (背圧より極端に低い値は背圧にクリップ)
                P0_Pa = np.maximum(P0_Pa, Pb_Pa * 0.999) 
            else:
                P0_Pa = P_raw * 1.0e6 # データが短すぎる場合はそのまま
        else:
            P0_Pa = float(P_raw) * 1.0e6

        # 温度: K (絶対値)
        T0_safe = np.abs(T_raw) + 1.0e-9
        A_m2 = A_mm2 * 1.0e-6

        # --- 計算 (NumPy配列演算) ---
        # 圧力比 (Pb / P0)
        # P0が補正されているため、初期状態では Ratio ≒ 1.0 となる
        current_ratio = np.divide(Pb_Pa, P0_Pa)
        
        # 1.0を超える(逆流条件)は1.0にクリップして流量0にする
        current_ratio = np.minimum(current_ratio, 1.0)
        
        # 臨界圧力比
        critical_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

        # 1. Choked Flow (チョーク流れ)
        # P0が高いとき (ratio < critical)
        term_choked = np.sqrt(gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
        m_dot_choked = (Cd * A_m2 * P0_Pa / np.sqrt(R * T0_safe)) * term_choked

        # 2. Unchoked Flow (亜音速流れ)
        # P0が低いとき (ratio > critical)
        term_inner = (current_ratio ** (2/gamma)) - (current_ratio ** ((gamma+1)/gamma))
        term_inner = np.maximum(term_inner, 0.0) # ルート内負防止
        m_dot_unchoked = Cd * A_m2 * P0_Pa * np.sqrt(
            (2*gamma / (R*T0_safe*(gamma-1))) * term_inner
        )

        # 統合
        m_dot = m_dot_choked.copy() if isinstance(m_dot_choked, np.ndarray) else np.full_like(P0_Pa, m_dot_choked)
        
        # アンチョーク領域の適用
        mask_unchoked = (current_ratio > critical_ratio)
        if isinstance(m_dot, np.ndarray):
            val_unchoked = m_dot_unchoked[mask_unchoked] if isinstance(m_dot_unchoked, np.ndarray) else m_dot_unchoked
            m_dot[mask_unchoked] = val_unchoked
            
            # カットオフ (比率が1.0に極めて近い、つまり差圧がない場合は強制0)
            # cutoff_ratio (例: 0.99) 以上なら0にする
            mask_cutoff = (current_ratio >= cutoff_ratio)
            m_dot[mask_cutoff] = 0.0

        # 単位変換 kg/s -> g/s
        result_data = m_dot * 1000.0

        # 結果をSensorDataとして登録
        fs_new = ref_sensor.fs if ref_sensor else 1.0
        t0_new = ref_sensor.start_time if ref_sensor else 0.0

        new_sensor = SensorData(
            name=target_name,
            data=result_data,
            fs=fs_new,
            unit="g/s",
            start_time=t0_new,
            source=f"Derived(from {src_p})"
        )
        
        data_store[target_name] = new_sensor
        print(f"    🔍 計算完了: {target_name} (Mean: {np.mean(result_data):.2f} g/s, Auto-Zero: ON)")