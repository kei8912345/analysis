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
        
        # アスペクト比など
        figsize = (6, 4.5)
        if opts.get('aspect_ratio') == 'square': figsize = (5, 5)

        fig, ax1 = plt.subplots(figsize=figsize)
        
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
                y = sensor.data
                
                # スタイル
                style = opts.get('series_styles', {}).get(name, {})
                ls = style.get('linestyle', linestyle_default)
                label = style.get('label', name)
                color = style.get('color', None) # Noneなら自動サイクル
                
                ln, = ax.plot(x, y, label=label, linestyle=ls, color=color, linewidth=1.5)
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

        # 凡例統合
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
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"    📈 保存: {os.path.basename(save_path)}")

    def _plot_spectrogram(self, stft_all_data, task, shot_name):
        # (ロジックはほぼ同じなので要点のみ)
        target = task.get('target')
        if target not in stft_all_data: return
        
        data = stft_all_data[target]
        f, t, Amp = data['f'], data['t'], data['Amp']
        spec_db = 20 * np.log10(Amp + 1e-9)

        fig, ax = plt.subplots(figsize=(7, 5))
        opts = task.get('plot_options', {})
        
        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 99)
        if opts.get('c_lim'): vmin, vmax = opts['c_lim']

        mesh = ax.pcolormesh(t, f, spec_db, cmap=opts.get('cmap', 'jet'), shading='gouraud', vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, ax=ax, label="Power [dB]")
        
        ax.set_title(task.get('title', target))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Freq [Hz]")
        if opts.get('y_lim'): ax.set_ylim(opts['y_lim'])

        save_name = f"{shot_name}_STFT_{target}.png" if shot_name else f"STFT_{target}.png"
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300)
        plt.close()
        print(f"    🌈 STFT描画: {save_name}")