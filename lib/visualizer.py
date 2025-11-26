# -*- coding: utf-8 -*-
import os
import platform
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import pickle

class Visualizer:
    """
    論文・レポート品質のグラフ描画を行うクラス
    Planファイルに基づいて、時系列グラフとスペクトログラムを描画する。
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        self.stft_data_cache = {} # STFTデータのロードキャッシュ
        
        # --- Matplotlib Global Settings ---
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 12
        
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.serif'] = ['MS Mincho', 'Times New Roman', 'Yu Mincho']
        elif system == 'Darwin':
            plt.rcParams['font.serif'] = ['Hiragino Mincho ProN', 'Times New Roman']
        else:
            plt.rcParams['font.serif'] = ['DejaVu Serif', 'TakaoMincho']

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['axes.grid'] = False
        plt.rcParams['figure.figsize'] = (6, 4.5)
        plt.rcParams['axes.prop_cycle'] = cycler(color=['black', 'red', 'blue', 'green', 'purple', 'brown'])

    def visualize(self, plan_config, df=None, stft_pkl_path=None, shot_name=None):
        """
        Planに基づいて可視化を実行する統合メソッド
        Args:
            plan_config (dict): plan.yamlの内容
            df (pd.DataFrame): 時系列データ (timeseriesタスク用)
            stft_pkl_path (str): STFT結果ファイルのパス (stft_spectrogramタスク用)
            shot_name (str): ショット名 (ファイル名付与用, optional)
        """
        tasks = plan_config.get('tasks', [])
        if not tasks:
            return

        print(f"  [Visualizer] 描画処理を開始します... (保存先: {self.figures_dir})")
        os.makedirs(self.figures_dir, exist_ok=True)

        # STFTデータのロード（必要なら）
        stft_data = None
        if stft_pkl_path and os.path.exists(stft_pkl_path):
            # タスクにSTFTが含まれているか確認
            if any(t.get('type') == 'stft_spectrogram' for t in tasks):
                try:
                    with open(stft_pkl_path, 'rb') as f:
                        stft_data = pickle.load(f)
                except Exception as e:
                    print(f"  ⚠️ STFTデータのロードに失敗: {e}")

        for task in tasks:
            kind = task.get('type')
            
            if kind == 'timeseries':
                if df is not None:
                    self._plot_timeseries(df, task)
                else:
                    print(f"  ⚠️ データフレームがないため、時系列グラフ '{task.get('title')}' をスキップします。")
            
            elif kind == 'stft_spectrogram':
                if stft_data:
                    self._plot_spectrogram_from_plan(stft_data, task, shot_name)
                else:
                    print(f"  ⚠️ STFTデータがないため、スペクトログラム '{task.get('title')}' をスキップします。")

    def _plot_spectrogram_from_plan(self, stft_all_data, task, shot_name=None):
        """Plan指定に基づいてスペクトログラムを描画"""
        target = task.get('target')
        title = task.get('title', f"Spectrogram: {target}")
        opts = task.get('plot_options', {})

        if target not in stft_all_data:
            print(f"    ⚠️ STFTデータ内にターゲット '{target}' が見つかりません。")
            return

        data = stft_all_data[target]
        f = data['f']
        t = data['t']
        Amp = data['Amp']

        # パワースペクトル密度(dB)に変換
        spec_db = 20 * np.log10(Amp + 1e-9)

        # プロット作成
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # カラーレンジの決定
        clim = opts.get('c_lim', None)
        if clim:
            vmin, vmax = clim
        else:
            # 自動設定
            vmin = np.percentile(spec_db, 5)
            vmax = np.percentile(spec_db, 99)

        cmap = opts.get('cmap', 'jet')

        # 描画
        mesh = ax.pcolormesh(t, f, spec_db, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # カラーバー
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Power [dB]")

        # 軸ラベル
        ax.set_title(title)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        
        # 範囲設定
        if opts.get('x_lim'): ax.set_xlim(opts['x_lim'])
        if opts.get('y_lim'): ax.set_ylim(opts['y_lim'])

        ax.minorticks_on()
        plt.tight_layout()

        # 保存
        stft_fig_dir = os.path.join(self.figures_dir, "stft")
        os.makedirs(stft_fig_dir, exist_ok=True)
        
        safe_title = title.replace(" ", "_").replace("/", "-").replace(":", "")
        
        if shot_name:
            filename = f"{shot_name}_{safe_title}.png"
        else:
            filename = f"{safe_title}.png"
            
        save_path = os.path.join(stft_fig_dir, filename)
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"    🌈 保存完了: {os.path.basename(save_path)}")

    def _plot_timeseries(self, df, task):
        # ... (既存コードと同じ)
        title = task.get('title', 'Untitled')
        x_col = task.get('x', 'Time')
        opts = task.get('plot_options', {})
        
        aspect = opts.get('aspect_ratio', 'golden')
        figsize = plt.rcParams['figure.figsize']
        if aspect == 'square': figsize = (5.0, 5.0)
        elif isinstance(aspect, (list, tuple)) and len(aspect) == 2: figsize = aspect

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.minorticks_on()

        if x_col in df.columns:
            x_data = df[x_col]
            x_label = opts.get('x_label', x_col)
        else:
            x_data = df.index
            x_label = opts.get('x_label', "Index")

        y1_cols = task.get('y', [])
        if isinstance(y1_cols, str): y1_cols = [y1_cols]
        y2_cols = task.get('secondary_y', [])
        if y2_cols and isinstance(y2_cols, str): y2_cols = [y2_cols]
        elif not y2_cols: y2_cols = []

        series_styles = opts.get('series_styles', {})
        base_style = {
            'linestyle': opts.get('linestyle', '-'),
            'marker': opts.get('marker', None),
            'markersize': opts.get('markersize', 4),
            'linewidth': opts.get('linewidth', 1.5),
            'alpha': opts.get('alpha', 1.0)
        }

        lines = []
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        def _safe_plot(ax, x, y_col, style, color_idx):
            if y_col not in df.columns:
                print(f"    ⚠️ カラムが見つかりません: {y_col}")
                return None
            valid_data = df[y_col].dropna()
            if valid_data.empty:
                print(f"    ⚠️ データが全てNaN(空)です: {y_col}")
                return None

            current_style = style.copy()
            if 'color' not in current_style:
                current_style['color'] = color_cycle[color_idx % len(color_cycle)]
            if y_col in series_styles:
                current_style.update(series_styles[y_col])

            ls = current_style.get('linestyle')
            mk = current_style.get('marker')
            if ls in [None, 'None', 'none', ''] and mk in [None, 'None', 'none', '']:
                current_style['marker'] = 'o'
                current_style['linestyle'] = 'None'

            label = current_style.pop('label', y_col)
            ln, = ax.plot(x, df[y_col], label=label, **current_style)
            return ln

        for i, col in enumerate(y1_cols):
            ln = _safe_plot(ax1, x_data, col, base_style, i)
            if ln: lines.append(ln)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(opts.get('y_label', "Primary Axis"))
        if opts.get('grid', False): ax1.grid(True)

        if y2_cols:
            ax2 = ax1.twinx()
            ax2.minorticks_on()
            sec_base_style = base_style.copy()
            if opts.get('linestyle') is None: sec_base_style['linestyle'] = '--'
            for j, col in enumerate(y2_cols):
                ln = _safe_plot(ax2, x_data, col, sec_base_style, len(y1_cols) + j)
                if ln: lines.append(ln)
            ax2.set_ylabel(opts.get('y2_label', "Secondary Axis"))

        ax1.set_title(title)
        if lines:
            labs = [l.get_label() for l in lines]
            ax1.legend(lines, labs, loc=opts.get('legend_loc', 'best'), frameon=False)
        else:
            print(f"    ⚠️ 表示可能なデータ系列がありません ({title})")

        if 'x_lim' in opts and opts['x_lim']: ax1.set_xlim(opts['x_lim'])
        if 'y_lim' in opts and opts['y_lim']: ax1.set_ylim(opts['y_lim'])
        if opts.get('y_log', False): ax1.set_yscale('log')

        plt.tight_layout()
        safe_title = title.replace(" ", "_").replace("/", "-").replace(":", "")
        save_path = os.path.join(self.figures_dir, f"{safe_title}.png")
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    📈 保存完了: {os.path.basename(save_path)}")
        except Exception as e:
            print(f"    ❌ 保存失敗 ({title}): {e}")
        finally:
            plt.close()