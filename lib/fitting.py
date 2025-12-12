# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit

class CoastingFitter:
    """
    コースティングダウン（減速）解析のための物理モデルとフィッティング機能を提供します。
    
    モデル: I * dw/dt = -(A + Bw)
    解: w(t) = (w0 + alpha/beta) * exp(-beta * t) - alpha/beta
    ただし alpha = A/I, beta = B/I
    """
    def __init__(self):
        pass

    def rpm_to_rads(self, rpm):
        return rpm * (2 * np.pi / 60.0)

    def rads_to_rpm(self, rads):
        return rads * (60.0 / (2 * np.pi))

    def model_func(self, t, w0, alpha, beta):
        """
        フィッティング用モデル関数 (時間は相対時間 t - t_start を想定)
        t: 時間 [s]
        w0: 初期角速度 [rad/s]
        alpha: 乾性抵抗項 A/I [rad/s^2]
        beta: 粘性抵抗項 B/I [1/s]
        """
        # オーバーフロー防止のためのクリップなどは適宜考慮
        term1 = (w0 + alpha / beta) * np.exp(-beta * t)
        term2 = alpha / beta
        return term1 - term2

    def fit(self, t_data, y_data_rpm, fit_range=None):
        """
        データに対してフィッティングを行います。
        計算はすべて [rad/s] 系で行います。
        
        Args:
            t_data (array): 時間配列 [s]
            y_data_rpm (array): 回転数配列 [rpm]
            fit_range (list): [start, end] フィッティングに使用する時間範囲
            
        Returns:
            dict: 推定されたパラメータとフィッティング結果
        """
        # 1. データの抽出
        if fit_range:
            mask = (t_data >= fit_range[0]) & (t_data <= fit_range[1])
            t_use = t_data[mask]
            y_use_rpm = y_data_rpm[mask]
        else:
            t_use = t_data
            y_use_rpm = y_data_rpm

        if len(t_use) < 10:
            return {"success": False, "message": "データ点数が不足しています"}

        # 2. 単位変換 (rpm -> rad/s)
        y_use_rads = self.rpm_to_rads(y_use_rpm)
        
        # 時間のオフセット（計算安定化のため、開始時間を0とする）
        t_start = t_use[0]
        t_relative = t_use - t_start

        # 3. 初期値の推定
        w0_init = y_use_rads[0]
        # 簡易的に beta(時定数の逆数) を推定: データ半分で減衰すると仮定
        beta_init = 0.1 
        alpha_init = w0_init * beta_init * 0.1 # 適当な初期値
        
        p0 = [w0_init, alpha_init, beta_init]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]) # 物理的に正の値のみ

        try:
            popt, pcov = curve_fit(
                self.model_func, 
                t_relative, 
                y_use_rads, 
                p0=p0, 
                bounds=bounds,
                maxfev=10000
            )
            
            w0_opt, alpha_opt, beta_opt = popt
            
            # 4. 結果の整理
            # フィッティング曲線生成用の関数（絶対時間 t を受け取るラッパー）
            def fitted_curve_func(t_abs):
                t_rel = t_abs - t_start
                w_rads = self.model_func(t_rel, w0_opt, alpha_opt, beta_opt)
                return self.rads_to_rpm(w_rads) # rpmで返す

            return {
                "success": True,
                "alpha": alpha_opt, # [rad/s^2]
                "beta": beta_opt,   # [1/s]
                "w0": self.rads_to_rpm(w0_opt), # [rpm]
                "t_use": t_use,
                "y_use": y_use_rpm,
                "fit_func": fitted_curve_func,
                "t_start": t_start
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    def calculate_physics_params(self, alpha, beta, moment_of_inertia):
        """
        慣性モーメント I [kg m^2] が既知の場合、A, B を計算します。
        alpha = A/I  -> A = alpha * I
        beta = B/I   -> B = beta * I
        """
        I = moment_of_inertia
        A = alpha * I
        B = beta * I
        return A, B