# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 13:35:41 2025
Modified for Shot09 Analysis based on PDF theory.
Optimized for speed using scipy.optimize.
Fixed torque zero-point issue at rest.
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from structs import SensorData
import time

class TurbineSimulator:
    """
    タービンのコールドフロー試験における理想挙動シミュレーションと
    実測データに基づくトルク解析を行うクラス。
    
    PDF記載の「理論フィッティング」手順に基づき、
    速度三角形を厳密に計算してトルクを算出します。
    """
    def __init__(self, config):
        self.config = config.copy() 
        self._load_params(self.config)

    def _load_params(self, config):
        self.I = float(config['I'])
        self.r = float(config['r'])
        self.A = float(config.get('A_eff', config.get('A')))
        
        # --- 角度設定 ---
        # alpha: ステータ流出角 (PDF図より周方向基準29度)
        self.alpha_deg = float(config['alpha'])
        
        # beta2: ロータ流出角 (PDF記載は70度)
        # 【重要】PDF記載の「70度」は軸方向基準（周方向からは20度）と解釈します。
        # ユーザー確認済み: 流入(90-29) + 流出(90-70) = 81度の転向角となる設定です。
        raw_beta2 = float(config.get('beta2', 70.0))
        
        # 補正: 45度以上の値が入ってきたら軸基準とみなして周基準に変換
        if raw_beta2 > 45.0:
            self.beta2_deg = 90.0 - raw_beta2
        else:
            self.beta2_deg = raw_beta2
        
        self.eta_ad = float(config.get('eta_ad', 1.0))
        self.time_delay = float(config.get('time_delay', 0.0))
        
        # 損失係数
        self.loss_A = float(config['loss_coeffs']['A'])
        self.loss_B = float(config['loss_coeffs']['B'])
        self.rho = float(config.get('rho', 1.165))
        
        # 角度をラジアンに変換 (これらは全て周方向基準の角度になります)
        self.alpha_rad = np.radians(self.alpha_deg)
        self.beta2_rad = np.radians(self.beta2_deg)
        
        # 異常値除去用のリミット
        opt_conf = config.get('optimization', {})
        self.value_limit_rpm = opt_conf.get('value_limit_rpm', None)

        # フィルタ設定
        proc_conf = config.get('processing', {})
        self.med_win = int(proc_conf.get('median_filter_window', 11))
        self.sigma_val = float(proc_conf.get('gaussian_filter_sigma', 5.0))
        
        if self.med_win % 2 == 0: self.med_win += 1

    def update_params(self, A_eff=None, eta_ad=None, time_delay=None):
        """最適化ループから呼ばれるパラメータ更新"""
        if A_eff is not None: self.A = float(A_eff)
        if eta_ad is not None: self.eta_ad = float(eta_ad)
        if time_delay is not None: self.time_delay = float(time_delay)

    def get_target_rpm(self, sensor_rpm):
        """最適化のために、前処理（フィルタリング済み）の実測RPMのみを返す"""
        t = sensor_rpm.time
        N_raw = sensor_rpm.data.copy()
        
        # 単位変換
        unit_str = str(sensor_rpm.unit).lower()
        if "hz" in unit_str:
            N_rpm = N_raw * 60.0
        elif "rpm" in unit_str:
            N_rpm = N_raw
        else:
            N_rpm = N_raw * 60.0

        # 異常値処理
        if self.value_limit_rpm is not None:
            outlier_mask = N_rpm > self.value_limit_rpm
            if np.sum(outlier_mask) > 0:
                valid_mask = ~outlier_mask
                if np.sum(valid_mask) > 0:
                    x = np.arange(len(N_rpm))
                    N_rpm[outlier_mask] = np.interp(x[outlier_mask], x[valid_mask], N_rpm[valid_mask])
                else:
                    N_rpm[:] = 0.0
        
        # フィルタリング
        omega_exp = N_rpm * (2 * np.pi / 60.0)
        safe_med_win = self.med_win
        if len(t) < safe_med_win: 
            safe_med_win = max(1, len(t) if len(t)%2!=0 else len(t)-1)
        
        omega_med = medfilt(omega_exp, kernel_size=safe_med_win)
        
        # 表示・比較用スムージング
        sg_win = min(15, len(t) if len(t)%2!=0 else len(t)-1)
        if sg_win < 3: sg_win = 3
        omega_smooth_disp = savgol_filter(omega_med, window_length=sg_win, polyorder=3)
        
        to_rpm = 60.0 / (2 * np.pi)
        return omega_smooth_disp * to_rpm

    def _run_simulation_fast(self, t, m_dot_kg_s, A_val, eta_val):
        """
        最適化ループ用の高速シミュレーションメソッド。
        """
        # 事前計算
        k_v = 1.0 / (self.rho * A_val)
        
        # 三角関数定数
        cos_alpha = np.cos(self.alpha_rad)
        sin_alpha = np.sin(self.alpha_rad)
        cos_beta2 = np.cos(self.beta2_rad)
        
        # 結果配列
        omega_sim = np.zeros_like(t)
        current_omega = 0.0
        
        # ループ定数
        I_inv = 1.0 / self.I
        loss_A = self.loss_A
        loss_B = self.loss_B
        r = self.r
        
        # オイラー積分
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            if dt <= 0: continue
            
            m_curr = m_dot_kg_s[i]
            
            # --- トルク計算 (展開形) ---
            V2 = m_curr * k_v
            V_theta2 = V2 * cos_alpha
            U = r * current_omega
            
            # W_theta2 = V_theta2 - U
            # W_axial2 = V2 * sin_alpha
            W_theta2 = V_theta2 - U
            W2_mag = np.sqrt(W_theta2*W_theta2 + (V2*sin_alpha)**2)
            V_theta3 = U - (W2_mag * cos_beta2)
            
            # T_fluid
            T_fluid = eta_val * m_curr * r * (V_theta2 - V_theta3)
            
            # --- 損失 ---
            # 静止時(omega=0)の摩擦による逆回転防止 & 0点処理
            if current_omega < 1e-4: # ほぼ0なら摩擦も0とみなす
                 T_loss = 0.0
            else:
                 T_loss = loss_A + loss_B * current_omega

            net_torque = T_fluid - T_loss
            
            current_omega += (net_torque * I_inv) * dt
            if current_omega < 0: current_omega = 0.0
            omega_sim[i+1] = current_omega

        to_rpm = 60.0 / (2 * np.pi)
        return omega_sim * to_rpm

    def process(self, sensor_m_dot, sensor_rpm):
        """最終結果出力用のメインプロセス"""
        t = sensor_rpm.time
        
        # 流量補間
        t_delayed = t - self.time_delay
        m_dot_g_s = np.interp(t_delayed, sensor_m_dot.time, sensor_m_dot.data, left=0.0, right=0.0)
        m_dot_kg_s = m_dot_g_s / 1000.0

        # シミュレーション実行
        N_sim = self._run_simulation_fast(t, m_dot_kg_s, self.A, self.eta_ad)
        
        N_filtered = self.get_target_rpm(sensor_rpm)
        
        # 生データのRPM換算（表示用）
        N_raw = sensor_rpm.data.copy()
        unit_str = str(sensor_rpm.unit).lower()
        if "hz" in unit_str:
            N_exp_raw_disp = N_raw * 60.0
        else:
             N_exp_raw_disp = N_raw
             
        if self.value_limit_rpm is not None:
             outlier_mask = N_exp_raw_disp > self.value_limit_rpm
             if np.sum(outlier_mask) > 0:
                 N_exp_raw_disp[outlier_mask] = np.nan

        # トルク解析 (実測ベース)
        omega_exp_rpm = N_filtered
        omega_exp_rad = omega_exp_rpm * (2 * np.pi / 60.0)
        
        omega_for_deriv = gaussian_filter1d(omega_exp_rad, sigma=self.sigma_val)
        d_omega_dt = np.gradient(omega_for_deriv, t)
        
        # --- 損失トルク計算 (0点補正付き) ---
        # 回転数が非常に低い(例: 10rpm未満)場合は、摩擦項Aを無視して0にする
        # これにより、開始前・終了後のトルクオフセット(A=0.0036など)を除去できる
        is_rotating = omega_exp_rpm > 10.0 # 10rpm threshold
        
        T_loss_exp = np.zeros_like(omega_exp_rad)
        T_loss_exp[is_rotating] = self.loss_A + self.loss_B * omega_for_deriv[is_rotating]
        
        T_actual = self.I * d_omega_dt + T_loss_exp
        
        # 停止中は T_actual も強制的に0にする（微分ノイズ対策）
        T_actual[~is_rotating] = 0.0
        
        T_ideal_eff_arr = self._calc_fluid_torque(omega_for_deriv, m_dot_kg_s)
        # 理想トルクも停止中かつ流量なしなら0
        is_flowing = m_dot_kg_s > 1e-4
        T_ideal_eff_arr[~(is_rotating | is_flowing)] = 0.0
        
        T_ideal_filtered = gaussian_filter1d(T_ideal_eff_arr, sigma=self.sigma_val)

        fs = sensor_rpm.fs
        t0 = sensor_rpm.start_time
        def make_sd(name, data, unit):
            return SensorData(name=name, data=data, fs=fs, unit=unit, start_time=t0, source="TurbineSim")

        results = {}
        results['N_sim'] = make_sd("Simulated RPM", N_sim, "rpm")
        results['N_exp_raw'] = make_sd("Exp RPM (Raw)", N_exp_raw_disp, "rpm")
        results['N_exp_filtered'] = make_sd("Exp RPM (Filtered)", N_filtered, "rpm")
        results['T_actual'] = make_sd("Actual Torque", T_actual, "Nm")
        results['T_ideal_point'] = make_sd(f"Ideal Torque (eta={self.eta_ad:.2f})", T_ideal_filtered, "Nm")
        results['T_ideal_raw'] = make_sd("Ideal Torque (Raw)", T_ideal_eff_arr, "Nm")
        results['m_dot_kg'] = make_sd(f"Mass Flow (Delayed {self.time_delay}s)", m_dot_kg_s, "kg/s")

        return results

    def _calc_fluid_torque(self, omega, m_dot):
        """配列計算用: 実測データ解析に使用"""
        V2 = m_dot / (self.rho * self.A)
        V_theta2 = V2 * np.cos(self.alpha_rad)
        U = self.r * omega
        
        W_theta2 = V_theta2 - U
        W_axial2 = V2 * np.sin(self.alpha_rad)
        W2_mag = np.sqrt(W_theta2**2 + W_axial2**2)
        
        W_theta3_mag = W2_mag * np.cos(self.beta2_rad)
        V_theta3 = U - W_theta3_mag
        
        T_ideal = m_dot * self.r * (V_theta2 - V_theta3)
        return self.eta_ad * T_ideal


class TurbineOptimizer:
    """パラメータ自動同定クラス (高速化版・スケーリングなし)"""
    def __init__(self, simulator_config):
        self.base_config = simulator_config.copy()
        
    def fit(self, sensor_m_dot, sensor_rpm, 
            A_range_mm2=(100, 2000, 10),
            eta_range=(0.0, 1.0, 0.01),
            delay_range=(0.0, 0.0, 1.0),
            value_limit=None):
        
        print("🚀 [Optimizer] パラメータ同定 (Scipy Minimize) を開始します...")
        
        if value_limit is not None:
             if 'optimization' not in self.base_config:
                 self.base_config['optimization'] = {}
             self.base_config['optimization']['value_limit_rpm'] = value_limit
             print(f"   Value Limit: <= {value_limit}")

        simulator = TurbineSimulator(self.base_config)
        
        # ターゲットデータ準備
        print("   Preparing target data...")
        N_target = simulator.get_target_rpm(sensor_rpm)
        t_sim = sensor_rpm.time
        
        # マスク作成 (有効区間のみで評価)
        mask = N_target > (np.max(N_target) * 0.05)
        valid_count = np.sum(mask)
        print(f"   Valid points for RMSE: {valid_count} / {len(N_target)}")
        
        # 最適化変数: [A_eff, eta_ad] のみ
        bounds = [
            (A_range_mm2[0] * 1e-6, A_range_mm2[1] * 1e-6), # x[0]: A_eff
            (eta_range[0], eta_range[1])                    # x[1]: eta
        ]
        
        # 初期推定値
        x0 = [
            (bounds[0][0] + bounds[0][1]) / 2.0,
            (bounds[1][0] + bounds[1][1]) / 2.0
        ]
        
        delay_vals = np.arange(delay_range[0], delay_range[1] + delay_range[2]/2, delay_range[2])
        
        best_global_score = float('inf')
        best_global_params = {}
        
        start_time = time.time()
        
        # 目的関数
        def objective_func(x, current_m_dot):
            A_val, eta_val = x
            
            # シミュレーション実行 (スケーリング引数なし)
            N_sim = simulator._run_simulation_fast(t_sim, current_m_dot, A_val, eta_val)
            
            diff = N_sim[mask] - N_target[mask]
            rmse = np.sqrt(np.mean(diff**2))
            return rmse

        total_delays = len(delay_vals)
        print(f"   Searching across {total_delays} delay steps...")
        
        for idx, delay in enumerate(delay_vals):
            t_query = t_sim - delay
            m_dot_g_s = np.interp(t_query, sensor_m_dot.time, sensor_m_dot.data, left=0.0, right=0.0)
            m_dot_kg_s = m_dot_g_s / 1000.0
            
            res = minimize(
                objective_func, 
                x0, 
                args=(m_dot_kg_s,),
                method='L-BFGS-B',
                bounds=bounds,
                tol=1e-4
            )
            
            if res.fun < best_global_score:
                best_global_score = res.fun
                best_global_params = {
                    'A_eff_mm2': res.x[0] * 1e6,
                    'eta_ad': res.x[1],
                    'delay': delay,
                    'rmse': res.fun
                }
                x0 = res.x
            
            if (idx+1) % 5 == 0 or (idx+1) == total_delays:
                print(f"   Step {idx+1}/{total_delays}: Delay={delay:.3f}s -> RMSE={res.fun:.2f}")

        elapsed = time.time() - start_time
        print(f"\n✅ 高速最適化完了 ({elapsed:.1f}s)")
        print(f"   🏆 Best Params:")
        if best_global_params:
            print(f"      A_eff   = {best_global_params['A_eff_mm2']:.1f} mm^2")
            print(f"      eta_ad  = {best_global_params['eta_ad']:.3f}")
            print(f"      delay   = {best_global_params['delay']:.3f} s")
            print(f"      RMSE    = {best_global_params['rmse']:.2f} rpm")
            
            # Simulatorにベストパラメータをセットして終了
            simulator.update_params(
                A_eff=best_global_params['A_eff_mm2']*1e-6,
                eta_ad=best_global_params['eta_ad'],
                time_delay=best_global_params['delay']
            )
        else:
            print("      No optimal parameters found.")
        
        return best_global_params