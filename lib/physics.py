# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class PhysicsEngine:
    """
    物理量の計算を行うクラス。
    負圧入力によるNaN発生を防ぐ安全装置付き。
    """
    
    GAS_CONSTANTS = {
        "Air": 287.058, "H2": 4124.0, "N2": 296.8, "O2": 259.8, "Ar": 208.1
    }
    GAMMAS = {
        "Air": 1.4, "H2": 1.405, "N2": 1.4, "O2": 1.395, "Ar": 1.667
    }

    def __init__(self):
        pass

    def add_derived_channels(self, df, derived_configs, sampling_rate=None):
        if df is None or df.empty or not derived_configs:
            return df

        print("  [Physics] 派生物理量の計算を開始します...")

        for name, config in derived_configs.items():
            calc_type = config.get('type')
            if calc_type == 'choked_flow' or calc_type == 'nozzle_flow':
                self._calc_compressible_flow(df, name, config, sampling_rate)
            
        return df

    def _get_values(self, df, source):
        if isinstance(source, (int, float)):
            return source
        if isinstance(source, str):
            if source in df.columns:
                return df[source]
        return None

    def _calc_compressible_flow(self, df, target_name, config, sampling_rate):
        """
        圧縮性流体の流量計算
        """
        src_p_key = config.get('source_p')
        src_t_key = config.get('source_t')
        
        P_raw = self._get_values(df, src_p_key)
        T_raw = self._get_values(df, src_t_key)

        if P_raw is None or T_raw is None:
            print(f"    ⚠️ 計算スキップ: データ不足 (P={src_p_key}, T={src_t_key})")
            return

        # --- パラメータ取得 ---
        gas_type = config.get('gas_type', 'Air')
        Cd = config.get('Cd', 1.0)
        A_mm2 = config.get('A_mm2', 1.0)
        cutoff_ratio = config.get('cutoff_ratio', 1.0)
        
        R = self.GAS_CONSTANTS.get(gas_type, 287.058)
        gamma = self.GAMMAS.get(gas_type, 1.4)

        # --- 背圧(P_back)の自動計算 ---
        back_pressure_duration = config.get('back_pressure_duration', 0.5)
        P_back_MPa = 0.1013

        if isinstance(P_raw, (pd.Series, np.ndarray)) and sampling_rate:
            n_samples = int(back_pressure_duration * sampling_rate)
            n_samples = min(n_samples, len(P_raw))
            if n_samples > 0:
                P_back_MPa = P_raw.iloc[:n_samples].mean()
                print(f"    🔍 背圧自動取得 ({back_pressure_duration}s): {P_back_MPa:.4f} MPa")
            else:
                print("    ⚠️ 背圧計算エラー: データ点数が0です。")
        else:
            print("    ℹ️ 背圧固定値を使用")

        # --- 計算準備 ---
        # ★修正: P0_Pa が 負や0 になると計算が爆発(NaN)するので、極小値(1e-9)でクリップする
        # これにより、オフセットズレで -0.001 MPa とかになってもエラーにならない
        P0_Pa = P_raw * 1.0e6
        if isinstance(P0_Pa, (pd.Series, np.ndarray)):
            P0_Pa = np.maximum(P0_Pa, 1.0e-9)
        else:
            P0_Pa = max(P0_Pa, 1.0e-9)

        Pb_Pa = P_back_MPa * 1.0e6
        
        T0_K = T_raw
        if isinstance(T0_K, (pd.Series, np.ndarray)):
            T0_safe = T0_K.abs() + 1e-9
        else:
            T0_safe = abs(T0_K) + 1e-9
            
        A_m2 = A_mm2 * 1.0e-6

        # --- 圧力比計算 ---
        # Pb / P0 (背圧 / 上流圧)
        current_ratio = np.divide(Pb_Pa, P0_Pa)

        # 臨界圧力比
        critical_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        
        # --- 流量計算 ---
        # 1. チョーク (理論最大流量)
        term_choked = np.sqrt(gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)))
        m_dot_choked = (Cd * A_m2 * P0_Pa / np.sqrt(R * T0_safe)) * term_choked

        # 2. 亜音速 (Saint-Venant)
        # ★修正: マイナス乗などによるNaNを防ぐため、計算順序に注意
        term_unchoked_inner = (current_ratio ** (2 / gamma)) - (current_ratio ** ((gamma + 1) / gamma))
        term_unchoked_inner = np.maximum(term_unchoked_inner, 0)
        
        m_dot_unchoked = Cd * A_m2 * P0_Pa * np.sqrt(
            (2 * gamma / (R * T0_safe * (gamma - 1))) * term_unchoked_inner
        )

        # --- 統合 & カットオフ ---
        m_dot_kg_s = m_dot_choked
        
        # 亜音速領域の上書き
        mask_unchoked = (current_ratio > critical_ratio) & (current_ratio < 1.0)
        
        if isinstance(m_dot_kg_s, (pd.Series, np.ndarray)):
            m_dot_kg_s = np.where(mask_unchoked, m_dot_unchoked, m_dot_kg_s)
            
            # カットオフ判定
            mask_no_flow = (current_ratio >= cutoff_ratio)
            m_dot_kg_s = np.where(mask_no_flow, 0.0, m_dot_kg_s)
        else:
            if current_ratio >= cutoff_ratio: m_dot_kg_s = 0.0
            elif current_ratio > critical_ratio: m_dot_kg_s = m_dot_unchoked

        m_dot_g_s = m_dot_kg_s * 1000.0
        df[target_name] = m_dot_g_s
        
        # NaN除去して平均を表示
        res_mean = np.nanmean(m_dot_g_s) if hasattr(m_dot_g_s, 'mean') else m_dot_g_s
        print(f"    🔍 流量計算完了 [{target_name}]:")
        print(f"       - 臨界圧力比 : {critical_ratio:.4f}")
        print(f"       - 平均背圧   : {P_back_MPa:.4f} MPa")
        print(f"       - カットオフ : 比率 {cutoff_ratio} 以上は流量0とみなします")
        print(f"       - 平均流量   : {res_mean:.4f} g/s")