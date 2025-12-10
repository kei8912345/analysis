# -*- coding: utf-8 -*-
import os
import platform
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
import pickle
import numpy as np

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        
        self.shared_time_ranges = {}

        # --- Matplotlib設定 ---
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 12
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.prop_cycle'] = cycler(color=['black', 'red', 'blue', 'green', 'purple'])
        
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.minor.visible'] = True
        
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.serif'] = ['MS Mincho', 'Times New Roman']
        elif system == 'Darwin':
            plt.rcParams['font.serif'] = ['Hiragino Mincho ProN', 'Times New Roman']

    def visualize(self, plan_config, data_store=None, stft_pkl_path=None, shot_name=None):
        raw_tasks = plan_config.get('tasks', [])
        if not raw_tasks: return

        print(f"  [Visualizer] 描画処理を開始します...")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.shared_time_ranges = {} 

        stft_data = None
        if stft_pkl_path and os.path.exists(stft_pkl_path):
            try:
                with open(stft_pkl_path, 'rb') as f: stft_data = pickle.load(f)
            except: pass

        # 実行順序の制御
        providers = []
        consumers = []
        others = []

        for task in raw_tasks:
            stats = task.get('plot_options', {}).get('stats', {})
            if stats.get('enable'):
                if 'define_range' in stats:
                    providers.append(task)
                elif 'use_range' in stats:
                    consumers.append(task)
                else:
                    others.append(task)
            else:
                others.append(task)
        
        sorted_tasks = providers + others + consumers

        for task in sorted_tasks:
            kind = task.get('type')
            if kind == 'timeseries':
                if data_store: self._plot_timeseries(data_store, task)
            elif kind == 'stft_spectrogram':
                if stft_data: self._plot_spectrogram(stft_data, task, shot_name)

    def _plot_timeseries(self, data_store, task):
        title = task.get('title', 'Untitled')
        opts = task.get('plot_options', {})
        
        # --- デバッグ出力（シンプル版） ---
        if 'legend_labels' in opts:
            targets = list(opts['legend_labels'].keys())
            print(f"    ℹ️  [設定] '{title}': 凡例ラベル置換あり -> {targets}")
        
        margin_left = 0.15
        margin_right = 0.82
        margin_bottom = 0.15
        margin_top = 0.90
        
        figsize = (7, 5)
        if opts.get('aspect_ratio') == 'square': figsize = (6, 6)

        fig, ax1 = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=margin_left, right=margin_right, bottom=margin_bottom, top=margin_top)
        
        y1_cols = task.get('y', [])
        if isinstance(y1_cols, str): y1_cols = [y1_cols]
        y2_cols = task.get('secondary_y', [])
        if isinstance(y2_cols, str): y2_cols = [y2_cols]

        def _plot_on_ax(ax, target_names, linestyle_default='-'):
            lines = []
            for i, name in enumerate(target_names):
                if name not in data_store:
                    print(f"    ⚠️ データなし: {name}")
                    continue
                
                sensor = data_store[name]
                x = sensor.time
                y = sensor.data.copy()
                
                freq_unit = opts.get('frequency_unit', 'Hz')
                if freq_unit.lower() == 'rpm': y = y * 60.0
                
                series_conf = opts.get('series_styles', {}).get(name, {})
                
                # --- ラベル名の決定 ---
                legend_map = opts.get('legend_labels', {})
                mapped_name = legend_map.get(name, name)
                label = series_conf.get('label', mapped_name)
                
                ls = series_conf.get('linestyle', opts.get('linestyle', linestyle_default))
                marker = series_conf.get('marker', opts.get('marker', None))
                ms = series_conf.get('markersize', opts.get('markersize', None))
                color = series_conf.get('color', opts.get('color', None))
                
                ln, = ax.plot(x, y, label=label, linestyle=ls, marker=marker, markersize=ms, color=color, linewidth=1.5)
                lines.append(ln)
            return lines

        lines1 = _plot_on_ax(ax1, y1_cols, '-')
        ax1.set_ylabel(opts.get('y_label', 'Primary'))
        ax1.set_xlabel(opts.get('x_label', 'Time [s]'))
        
        ax1.minorticks_on()
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which='both', top=True, right=True, direction='in')

        lines2 = []
        if y2_cols:
            ax2 = ax1.twinx()
            lines2 = _plot_on_ax(ax2, y2_cols, '--')
            ax2.set_ylabel(opts.get('y2_label', 'Secondary'))
            ax2.minorticks_on()
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.tick_params(which='both', direction='in')

        # --- 凡例表示 ---
        all_lines = lines1 + lines2
        if all_lines:
            labs = [l.get_label() for l in all_lines]
            
            legend_opts = opts.get('legend', {})
            legend_loc = legend_opts.get('loc', 'upper right')
            legend_fontsize = legend_opts.get('fontsize', 12)
            
            ax1.legend(all_lines, labs, loc=legend_loc, frameon=False, fontsize=legend_fontsize)

        ax1.set_title(title)
        if opts.get('x_lim'): ax1.set_xlim(opts['x_lim'])
        if opts.get('y_lim'): ax1.set_ylim(opts['y_lim'])
        if opts.get('grid'): ax1.grid(True, linestyle=':')

        # --- 統計解析 & 表示 ---
        stats_conf = opts.get('stats', {})
        if stats_conf.get('enable', False) and y1_cols:
            target_name = y1_cols[0]
            if target_name in data_store:
                sensor = data_store[target_name]
                t = sensor.time
                d = sensor.data.copy()

                freq_unit = opts.get('frequency_unit', 'Hz')
                if freq_unit.lower() == 'rpm': d = d * 60.0

                t_start, t_end = None, None
                
                if 'use_range' in stats_conf:
                    ref_name = stats_conf['use_range']
                    if ref_name in self.shared_time_ranges:
                        t_start, t_end = self.shared_time_ranges[ref_name]
                
                if t_start is None:
                    max_val = np.nanmax(d)
                    thresh_ratio = float(stats_conf.get('threshold', 0.9))
                    thresh_val = max_val * thresh_ratio
                    valid_indices = np.where(d >= thresh_val)[0]
                    if len(valid_indices) > 0:
                        t_start = t[valid_indices[0]]
                        t_end = t[valid_indices[-1]]
                        if 'define_range' in stats_conf:
                            def_name = stats_conf['define_range']
                            self.shared_time_ranges[def_name] = (t_start, t_end)
                            print(f"    💾 範囲定義 '{def_name}': {t_start:.3f} - {t_end:.3f} s")

                if t_start is not None and t_end is not None:
                    mask = (t >= t_start) & (t <= t_end)
                    segment = d[mask]
                    
                    if len(segment) > 0:
                        mean_val = np.nanmean(segment)
                        max_val = np.nanmax(segment)
                        
                        lc = stats_conf.get('color', 'red')
                        ax1.axvline(t_start, color=lc, linestyle=':', linewidth=1.5, alpha=0.8)
                        ax1.axvline(t_end, color=lc, linestyle=':', linewidth=1.5, alpha=0.8)
                        
                        calc_modes = stats_conf.get('calc_mode', 'mean')
                        if isinstance(calc_modes, str): calc_modes = [calc_modes]
                        
                        lines = []
                        if "mean" in calc_modes: lines.append(f"Mean: {mean_val:.2f}")
                        if "max" in calc_modes:  lines.append(f"Max:  {max_val:.2f}")
                        
                        stats_text = "\n".join(lines)
                        
                        stats_pos = stats_conf.get('position', [0.95, 0.80])
                        stats_fontsize = stats_conf.get('fontsize', 12)
                        
                        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=lc)
                        ax1.text(stats_pos[0], stats_pos[1], stats_text, transform=ax1.transAxes, fontsize=stats_fontsize,
                                verticalalignment='top', horizontalalignment='right', bbox=props)

        # ファイル名は元のシンプルな処理に戻しました（LaTeX記号が含まれるとエラーになるので注意）
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

        freq_unit = opts.get('frequency_unit', 'Hz')
        if freq_unit.lower() == 'rpm':
            f = f * 60.0
            default_y_label = "Frequency [rpm]"
        else:
            default_y_label = "Freq [Hz]"

        margin_left = 0.15
        margin_right = 0.82
        margin_bottom = 0.15
        margin_top = 0.90
        
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.subplots_adjust(left=margin_left, right=margin_right, bottom=margin_bottom, top=margin_top)

        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 99)
        if opts.get('c_lim'): vmin, vmax = opts['c_lim']

        mesh = ax.pcolormesh(t, f, spec_db, cmap=opts.get('cmap', 'jet'), shading='gouraud', vmin=vmin, vmax=vmax)
        
        cax_width = 0.02
        cax_left = margin_right + 0.02
        cax_bottom = margin_bottom
        cax_height = margin_top - margin_bottom
        
        cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
        plt.colorbar(mesh, cax=cax, label=opts.get('c_label', "Power [dB]"))
        
        ax.set_title(task.get('title', target))
        ax.set_xlabel(opts.get('x_label', "Time [s]"))
        ax.set_ylabel(opts.get('y_label', default_y_label))
        
        if opts.get('y_lim'): ax.set_ylim(opts['y_lim'])
        if opts.get('x_lim'): ax.set_xlim(opts['x_lim'])
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', top=True, right=True, direction='in')

        save_name = f"{shot_name}_STFT_{target}.png" if shot_name else f"STFT_{target}.png"
        plt.savefig(os.path.join(self.figures_dir, save_name), dpi=300)
        plt.close()
        print(f"    🌈 STFT描画: {save_name} (Unit: {freq_unit})")