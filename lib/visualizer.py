# -*- coding: utf-8 -*-
import os
import platform
import matplotlib.pyplot as plt
from cycler import cycler
import pickle
import numpy as np

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        
        # --- Matplotlib設定 ---
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 12
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.prop_cycle'] = cycler(color=['black', 'red', 'blue', 'green', 'purple'])
        
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.serif'] = ['MS Mincho', 'Times New Roman']
        elif system == 'Darwin':
            plt.rcParams['font.serif'] = ['Hiragino Mincho ProN', 'Times New Roman']

    def visualize(self, plan_config, data_store=None, stft_pkl_path=None, shot_name=None):
        """
        Args:
            data_store (dict): {name: SensorData}
        """
        tasks = plan_config.get('tasks', [])
        if not tasks: return

        print(f"  [Visualizer] 描画開始...")
        os.makedirs(self.figures_dir, exist_ok=True)

        stft_data = None
        if stft_pkl_path and os.path.exists(stft_pkl_path):
            try:
                with open(stft_pkl_path, 'rb') as f: stft_data = pickle.load(f)
            except: pass

        for task in tasks:
            kind = task.get('type')
            if kind == 'timeseries':
                if data_store: self._plot_timeseries(data_store, task)
            elif kind == 'stft_spectrogram':
                if stft_data: self._plot_spectrogram(stft_data, task, shot_name)

    def _plot_timeseries(self, data_store, task):
        title = task.get('title', 'Untitled')
        opts = task.get('plot_options', {})
        
        # --- レイアウト設定 (固定) ---
        margin_left = 0.15
        margin_right = 0.82
        margin_bottom = 0.15
        margin_top = 0.90
        
        figsize = (7, 5)
        if opts.get('aspect_ratio') == 'square': figsize = (6, 6)

        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 描画領域を固定
        fig.subplots_adjust(
            left=margin_left, right=margin_right, 
            bottom=margin_bottom, top=margin_top
        )
        
        y1_cols = task.get('y', [])
        if isinstance(y1_cols, str): y1_cols = [y1_cols]
        y2_cols = task.get('secondary_y', [])
        if isinstance(y2_cols, str): y2_cols = [y2_cols]

        # 描画ヘルパー
        def _plot_on_ax(ax, target_names, linestyle_default='-'):
            lines = []
            for i, name in enumerate(target_names):
                if name not in data_store:
                    print(f"    ⚠️ データなし: {name}")
                    continue
                
                sensor = data_store[name]
                x = sensor.time
                y = sensor.data.copy() # 元データを書き換えないようにコピー
                
                # ★追加: RPM変換ロジック (時系列用)
                freq_unit = opts.get('frequency_unit', 'Hz')
                if freq_unit.lower() == 'rpm':
                    y = y * 60.0
                
                # --- スタイル設定の優先順位処理 ---
                # 1. series_styles (個別) > 2. plot_options (全体) > 3. default (引数)
                
                series_conf = opts.get('series_styles', {}).get(name, {})
                ls = series_conf.get('linestyle', opts.get('linestyle', linestyle_default))
                marker = series_conf.get('marker', opts.get('marker', None))
                ms = series_conf.get('markersize', opts.get('markersize', None))
                color = series_conf.get('color', opts.get('color', None))
                label = series_conf.get('label', name)
                
                ln, = ax.plot(x, y, 
                              label=label, 
                              linestyle=ls, 
                              marker=marker,
                              markersize=ms,
                              color=color, 
                              linewidth=1.5)
                lines.append(ln)
            return lines

        lines1 = _plot_on_ax(ax1, y1_cols, '-')
        ax1.set_ylabel(opts.get('y_label', 'Primary'))
        ax1.set_xlabel(opts.get('x_label', 'Time [s]'))

        lines2 = []
        if y2_cols:
            ax2 = ax1.twinx()
            lines2 = _plot_on_ax(ax2, y2_cols, '--')
            ax2.set_ylabel(opts.get('y2_label', 'Secondary'))

        all_lines = lines1 + lines2
        if all_lines:
            labs = [l.get_label() for l in all_lines]
            ax1.legend(all_lines, labs, loc=opts.get('legend_loc', 'best'), frameon=False)

        ax1.set_title(title)
        if opts.get('x_lim'): ax1.set_xlim(opts['x_lim'])
        if opts.get('y_lim'): ax1.set_ylim(opts['y_lim'])
        if opts.get('grid'): ax1.grid(True, linestyle=':')

        safe_title = title.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(self.figures_dir, f"{safe_title}.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"    📈 保存: {os.path.basename(save_path)}")

    def _plot_spectrogram(self, stft_all_data, task, shot_name):
        target = task.get('target')
        if target not in stft_all_data: return
        
        data = stft_all_data[target]
        f, t, Amp = data['f'], data['t'], data['Amp']
        
        spec_db = 20 * np.log10(Amp + 1e-9)
        opts = task.get('plot_options', {})

        # ★追加: 周波数単位の変換 (rpm対応) - スペクトログラム用
        freq_unit = opts.get('frequency_unit', 'Hz')
        if freq_unit.lower() == 'rpm':
            f = f * 60.0
            default_y_label = "Frequency [rpm]"
        else:
            default_y_label = "Freq [Hz]"

        # --- レイアウト設定 (時系列と統一) ---
        margin_left = 0.15
        margin_right = 0.82
        margin_bottom = 0.15
        margin_top = 0.90
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        fig.subplots_adjust(
            left=margin_left, right=margin_right, 
            bottom=margin_bottom, top=margin_top
        )

        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 99)
        if opts.get('c_lim'): vmin, vmax = opts['c_lim']

        # 変換後の f を使用して描画
        mesh = ax.pcolormesh(t, f, spec_db, cmap=opts.get('cmap', 'jet'), shading='gouraud', vmin=vmin, vmax=vmax)
        
        cax_width = 0.02
        cax_left = margin_right + 0.02
        cax_bottom = margin_bottom
        cax_height = margin_top - margin_bottom
        
        cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
        plt.colorbar(mesh, cax=cax, label=opts.get('c_label', "Power [dB]"))
        
        ax.set_title(task.get('title', target))
        ax.set_xlabel(opts.get('x_label', "Time [s]"))
        
        # 指定がなければ自動設定したラベルを使う
        ax.set_ylabel(opts.get('y_label', default_y_label))
        
        if opts.get('y_lim'): ax.set_ylim(opts['y_lim'])
        if opts.get('x_lim'): ax.set_xlim(opts['x_lim'])

        save_name = f"{shot_name}_STFT_{target}.png" if shot_name else f"STFT_{target}.png"
        
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300)
        plt.close()
        print(f"    🌈 STFT描画: {save_name} (Unit: {freq_unit})")