# -*- coding: utf-8 -*-
import os
import glob
import re
import cv2
import numpy as np
import time
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from structs import SensorData

# --- ワーカー関数 (変更なし) ---
def _worker_process_batch(args):
    file_paths, roi_list = args
    batch_results = []
    for img_path in file_paths:
        frame_result = {}
        try:
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
                            x_s, y_s = max(0, min(x, w_img)), max(0, min(y, h_img))
                            x_e, y_e = max(0, min(x + w, w_img)), max(0, min(y + h, h_img))
                            if x_e > x_s and y_e > y_s:
                                val = np.mean(img[y_s:y_e, x_s:x_e])
                            else: val = 0.0
                        except: val = np.nan
                    else: val = np.mean(img)
                    frame_result[name] = val
        except:
            frame_result = {item['name']: np.nan for item in roi_list}
        batch_results.append(frame_result)
    return batch_results

class HSCAnalyzer:
    def __init__(self):
        pass

    def process_shot(self, base_search_dir, shot_number, output_root_dir, spec_config):
        start_time = time.time()
        print(f"\n🎬 [HSC解析] Shot {shot_number} 処理開始")

        # ★変更点: 保存先を .cache/hsc_brightness に変更
        save_dir = os.path.join(output_root_dir, ".cache", "hsc_brightness")
        os.makedirs(save_dir, exist_ok=True)
        
        target_img_dir = self._find_shot_folder(base_search_dir, shot_number)
        if not target_img_dir:
            print(f"❌ 画像フォルダなし: {base_search_dir}")
            return None

        image_files = sorted(glob.glob(os.path.join(target_img_dir, "*.jpg")))
        total_frames = len(image_files)
        if total_frames == 0:
            print("❌ jpgファイルなし")
            return None
        
        hsc_settings = self._extract_hsc_settings(spec_config)
        if not hsc_settings: return None

        fps = hsc_settings['fps']
        pre_trig = hsc_settings['pre_trigger_frames']
        roi_list = hsc_settings['rois']
        
        # --- 並列処理 ---
        chunk_size = 500 
        chunks = [image_files[i:i + chunk_size] for i in range(0, total_frames, chunk_size)]
        task_args = [(chunk, roi_list) for chunk in chunks]
        max_workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
        
        results_accum = {item['name']: [] for item in roi_list}

        print(f"🚀 解析中 ({total_frames} frames, {len(roi_list)} ROIs)...")
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                with tqdm(total=total_frames, unit="fr") as pbar:
                    for batch_res in executor.map(_worker_process_batch, task_args):
                        for frame_res in batch_res:
                            for name, val in frame_res.items():
                                results_accum[name].append(val)
                        pbar.update(len(batch_res))
        except Exception as e:
            print(f"❌ 解析エラー: {e}")
            return None

        # --- SensorData化 ---
        hsc_data_store = {}
        start_t = -(pre_trig / fps)

        for name, val_list in results_accum.items():
            data_arr = np.array(val_list, dtype=float)
            
            s_data = SensorData(
                name=name,
                data=data_arr,
                fs=fps,
                unit="brightness",
                start_time=start_t,
                source="HSC_Image_Analysis"
            )
            hsc_data_store[name] = s_data

        # 保存
        save_name = f"shot{shot_number:03d}_hsc.pkl"
        save_path = os.path.join(save_dir, save_name)
        
        with open(save_path, 'wb') as f:
            pickle.dump(hsc_data_store, f)
            
        print(f"✅ 保存完了: {save_path}")
        return save_path

    def _extract_hsc_settings(self, spec_config):
        measurements = spec_config.get('measurements', [])
        hsc_items = [m for m in measurements if str(m.get('id', '')).lower() == 'hsc' or str(m.get('type', '')).upper().startswith('HSC')]
        
        if not hsc_items and 'hsc_analysis' in spec_config:
            legacy = spec_config['hsc_analysis']
            hsc_items = legacy.get('rois', [legacy]) if 'rois' in legacy else [legacy]
            if isinstance(hsc_items, dict): hsc_items = [hsc_items]
            for item in hsc_items:
                if 'fps' not in item: item['fps'] = legacy.get('fps', 1000.0)
                if 'pre_trigger_frames' not in item: item['pre_trigger_frames'] = legacy.get('pre_trigger_frames', 0)

        if not hsc_items: return None
        
        first = hsc_items[0]
        return {
            'fps': float(first.get('fps', 1000.0)),
            'pre_trigger_frames': int(first.get('pre_trigger_frames', 0)),
            'rois': [{'name': i.get('name', i.get('id')), 'roi': i.get('roi')} for i in hsc_items]
        }

    def _find_shot_folder(self, search_root, shot_num):
        if not os.path.exists(search_root): return None
        subdirs = [d for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        candidates = [d for d in subdirs if str(shot_num) in re.findall(r'\d+', d)]
        if not candidates: return None
        candidates.sort(key=len)
        return os.path.join(search_root, candidates[0])