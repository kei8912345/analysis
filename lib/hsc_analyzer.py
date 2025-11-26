# -*- coding: utf-8 -*-
import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

# --- バッチ処理用ワーカー関数 ---
def _worker_process_batch(args):
    """
    画像のリスト(バッチ)を受け取り、まとめて解析して返す関数
    起動オーバーヘッドを削減するためにループ処理を内部で行う
    
    Args:
        args: (file_paths_list, roi_list) のタプル
    Returns:
        list: [ {roi_name: val, ...}, ... ] (画像枚数分のリスト)
    """
    file_paths, roi_list = args
    batch_results = []

    for img_path in file_paths:
        frame_result = {}
        try:
            # 日本語パス対応読み込み
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

            if img is None:
                frame_result = {item['name']: np.nan for item in roi_list}
            else:
                h_img, w_img = img.shape
                
                for item in roi_list:
                    name = item['name']
                    roi = item['roi']

                    val = np.nan
                    if roi:
                        try:
                            if isinstance(roi, str): roi = eval(roi)
                            x, y, w, h = map(int, roi)
                            
                            # クリップ
                            x_s = max(0, min(x, w_img))
                            y_s = max(0, min(y, h_img))
                            x_e = max(0, min(x + w, w_img))
                            y_e = max(0, min(y + h, h_img))

                            if x_e > x_s and y_e > y_s:
                                crop = img[y_s:y_e, x_s:x_e]
                                val = np.mean(crop)
                            else:
                                val = 0.0
                        except:
                            val = np.nan
                    else:
                        val = np.mean(img)
                    
                    frame_result[name] = val

        except Exception:
            frame_result = {item['name']: np.nan for item in roi_list}
        
        batch_results.append(frame_result)

    return batch_results

class HSCAnalyzer:
    """
    ハイスピードカメラ(HSC)の連番画像を並列バッチ処理で高速解析するクラス。
    """
    def __init__(self):
        pass

    def process_shot(self, base_search_dir, shot_number, output_root_dir, spec_config):
        start_time = time.time()
        print(f"\n🎬 [HSC解析] Shot {shot_number} の処理を開始します...")

        # 1. 保存先
        save_dir = os.path.join(output_root_dir, "hsc_timeseries")
        os.makedirs(save_dir, exist_ok=True)
        
        # 2. フォルダ特定
        target_img_dir = self._find_shot_folder(base_search_dir, shot_number)
        if not target_img_dir:
            print(f"❌ [HSC解析] 画像フォルダが見つかりません: {base_search_dir}")
            return None

        # 3. 画像リスト
        print("    📂 ファイルリストを取得中...")
        image_files = sorted(glob.glob(os.path.join(target_img_dir, "*.jpg")))
        total_frames = len(image_files)
        if total_frames == 0:
            print(f"❌ [HSC解析] 画像ファイル(.jpg)がありません。")
            return None
        
        print(f"    📊 対象: {os.path.basename(target_img_dir)} ({total_frames} frames)")
        
        # 4. 設定抽出
        hsc_settings = self._extract_hsc_settings(spec_config)
        if not hsc_settings:
            print(f"❌ [HSC解析] HSC設定(ROI)がありません。")
            return None

        fps = hsc_settings['fps']
        pre_trig = hsc_settings['pre_trigger_frames']
        roi_list = hsc_settings['rois']
        
        print(f"    ⚙️  FPS: {fps}, PreTrig: {pre_trig}, ROI数: {len(roi_list)}")

        # 5. バッチ作成 (ここが高速化の肝)
        # 画像を chunk_size 枚ずつの束にする
        # IOボトルネックを考慮し、大きすぎず小さすぎないサイズ (例: 100~500枚)
        chunk_size = 500 
        chunks = [image_files[i:i + chunk_size] for i in range(0, total_frames, chunk_size)]
        
        # ワーカーへの引数リスト
        task_args = [(chunk, roi_list) for chunk in chunks]

        # ワーカー数 (論理コア数 - 1 程度が安全)
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        # ディスクIO負荷が高い場合は少なめの方が速い場合もあるので上限キャップ
        max_workers = min(max_workers, 8) 
        
        print(f"🚀 [並列処理] ワーカー数:{max_workers}, バッチサイズ:{chunk_size} で解析開始...")

        results_accum = {item['name']: [] for item in roi_list}

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # バッチ単位でtqdmを回す (updateを手動で行うことで枚数ベースのバーにする)
                with tqdm(total=total_frames, unit="fr", desc="    Processing") as pbar:
                    # mapで順序を保ったまま実行
                    for batch_res in executor.map(_worker_process_batch, task_args):
                        # バッチ分の結果を統合
                        for frame_res in batch_res:
                            for name, val in frame_res.items():
                                results_accum[name].append(val)
                        
                        # プログレスバーを進める
                        pbar.update(len(batch_res))

        except Exception as e:
            print(f"\n❌ [HSC解析] エラー発生: {e}")
            return None

        # 6. DataFrame化
        df = pd.DataFrame(results_accum)
        
        times = (np.arange(total_frames) - pre_trig) / fps
        df.insert(0, 'Time', times)

        # 7. 保存
        save_name = f"shot{shot_number:03d}_hsc.pkl"
        save_path = os.path.join(save_dir, save_name)
        
        try:
            df.to_pickle(save_path)
            total_time = time.time() - start_time
            print(f"✅ [HSC解析] 完了: {save_path}")
            print(f"   (所要時間: {total_time:.1f}s, 平均速度: {total_frames/total_time:.1f} fps)")
            return save_path
            
        except Exception as e:
            print(f"❌ [HSC解析] 保存エラー: {e}")
            return None

    def _extract_hsc_settings(self, spec_config):
        measurements = spec_config.get('measurements', [])
        hsc_items = [
            m for m in measurements 
            if str(m.get('id', '')).lower() == 'hsc' or str(m.get('type', '')).upper().startswith('HSC')
        ]
        
        if not hsc_items and 'hsc_analysis' in spec_config:
            legacy = spec_config['hsc_analysis']
            if 'rois' in legacy:
                hsc_items = legacy['rois']
                for item in hsc_items:
                    if 'fps' not in item: item['fps'] = legacy.get('fps', 1000.0)
                    if 'pre_trigger_frames' not in item: item['pre_trigger_frames'] = legacy.get('pre_trigger_frames', 0)
            else:
                hsc_items = [legacy]

        if not hsc_items: return None

        first_item = hsc_items[0]
        base_fps = float(first_item.get('fps', 1000.0))
        base_pre_trig = int(first_item.get('pre_trigger_frames', 0))

        roi_list = []
        for item in hsc_items:
            name = item.get('name', item.get('id', 'Unknown'))
            roi_raw = item.get('roi', None)
            roi_list.append({'name': name, 'roi': roi_raw})

        return {'fps': base_fps, 'pre_trigger_frames': base_pre_trig, 'rois': roi_list}

    def _find_shot_folder(self, search_root, shot_num):
        if not os.path.exists(search_root): return None
        subdirs = [d for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        candidates = [d for d in subdirs if str(shot_num) in re.findall(r'\d+', d)]
        if not candidates: return None
        candidates.sort(key=len)
        return os.path.join(search_root, candidates[0])