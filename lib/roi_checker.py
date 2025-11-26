# -*- coding: utf-8 -*-
import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ROIVisualizer:
    """
    HSC画像のROI（関心領域）を確認するための軽量クラス。
    Matplotlibを使用して、座標軸付きの確認画像を生成する。
    """
    def __init__(self):
        # 日本語フォント設定 (環境に合わせて適宜フォールバック)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Maru Gothic Pro', 'DejaVu Sans']

    def generate_preview(self, base_search_dir, shot_number, output_root_dir, spec_config):
        """
        ROI確認画像を生成するメイン処理
        """
        # 1. 保存先ディレクトリの準備
        output_dir = os.path.join(output_root_dir, "ROI確認")
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. 画像フォルダの特定
        target_img_dir = self._find_shot_folder(base_search_dir, shot_number)
        if not target_img_dir:
            print(f"❌ [ROI] Shot {shot_number} の画像フォルダが見つかりません: {base_search_dir}")
            return

        # 3. 最初の1枚だけを探す
        image_files = sorted(glob.glob(os.path.join(target_img_dir, "*.jpg")))
        if not image_files:
            print(f"❌ [ROI] 画像ファイル(.jpg)が見つかりません。")
            return
        
        target_img_path = image_files[0]
        print(f"🔍 [ROI] 参照画像: {os.path.basename(target_img_path)}")

        # 4. 画像読み込み (日本語パス対応版 & Matplotlib用RGB変換)
        try:
            img_array = np.fromfile(target_img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"❌ [ROI] 画像の読み込みに失敗しました。")
                return

            # MatplotlibはRGB配列を期待するので変換
            if len(img.shape) == 2:
                # グレースケール -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # BGRA -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img_rgb = img.copy()

            print(f"    ℹ️  Image Info: {img.shape[1]}x{img.shape[0]} px")

        except Exception as e:
            print(f"❌ [ROI] 画像デコードエラー: {e}")
            return

        # 5. ROI情報の抽出
        measurements = spec_config.get('measurements', [])
        hsc_items = [
            m for m in measurements 
            if str(m.get('id', '')).lower() == 'hsc' or str(m.get('type', '')).upper().startswith('HSC')
        ]

        if not hsc_items and 'hsc_analysis' in spec_config:
            print("⚠️ [ROI] measurements内にHSC設定がありません。hsc_analysis(旧設定)を使用します。")
            legacy_conf = spec_config['hsc_analysis']
            legacy_item = legacy_conf.copy()
            legacy_item['name'] = legacy_item.get('name', 'HSC_Legacy')
            if 'rois' in legacy_conf:
                hsc_items = legacy_conf['rois']
            else:
                hsc_items = [legacy_item]

        if not hsc_items:
            print(f"⚠️ [ROI] SpecファイルにHSC(ROI)の設定が見つかりません。保存せずに終了します。")
            return

        # 6. Matplotlibによる描画処理
        try:
            # 図の作成 (サイズは画像のアスペクト比に合わせるが、最大幅を制限)
            h, w = img_rgb.shape[:2]
            fig_width = 10
            fig_height = fig_width * (h / w)
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # 画像を表示
            ax.imshow(img_rgb)
            
            count = 0
            print(f"ℹ️  [ROI] 検出されたHSC設定数: {len(hsc_items)}")
            
            for item in hsc_items:
                name = item.get('name', 'Unknown')
                roi = item.get('roi', None) # [x, y, w, h]
                
                if roi:
                    try:
                        if isinstance(roi, str): roi = eval(roi)
                        
                        if len(roi) == 4:
                            x, y, rect_w, rect_h = map(int, roi)
                            
                            # 範囲チェックログ
                            if x >= w or y >= h:
                                print(f"    ⚠️ Warning: ROI ({x},{y}) が画像サイズ ({w}x{h}) の外にあります！")
                            
                            print(f"    ✏️  Drawing: {name} -> Rect({x}, {y}, {rect_w}, {rect_h})")

                            # --- 赤枠 (Rectangle Patch) ---
                            # xyは左下ではなく「左上」が基準 (Matplotlibの画像座標系はy軸が下向き)
                            rect = patches.Rectangle((x, y), rect_w, rect_h, 
                                                     linewidth=2, edgecolor='red', facecolor='none')
                            ax.add_patch(rect)
                            
                            # --- テキストラベル ---
                            # 枠の左上に表示。視認性を上げるため背景色をつける
                            ax.text(x, y - 5, name, color='yellow', fontsize=10, fontweight='bold',
                                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
                            
                            count += 1
                        else:
                            print(f"⚠️ [ROI] '{name}' のROI要素数が不正です: {roi}")
                    except Exception as e:
                         print(f"⚠️ [ROI] '{name}' のROI描画中にエラー: {e}")
                else:
                    print(f"⚠️ [ROI] '{name}' のROI設定が空(None)です。")

            # 7. 軸・グリッドの設定
            ax.set_title(f"Shot {shot_number} ROI Check", fontsize=14)
            ax.set_xlabel("X [pixel]")
            ax.set_ylabel("Y [pixel]")
            
            # グリッドを表示 (水色の点線)
            ax.grid(True, which='both', color='cyan', linestyle=':', linewidth=0.5, alpha=0.5)
            # 副目盛りを表示
            ax.minorticks_on()
            
            # 8. 保存
            # MatplotlibならPNG形式がきれい
            save_name = f"Shot{shot_number:02d}_ROI_Check_Grid.png"
            save_path = os.path.join(output_dir, save_name)
            
            # 余白を調整して保存
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close(fig) # メモリ解放
            
            if count > 0:
                print(f"✅ [ROI] 確認画像を保存しました (Grid付): {save_path}")
            else:
                print(f"⚠️ [ROI] 画像は保存されましたが、ROIは描画されませんでした。")

        except Exception as e:
            print(f"❌ [ROI] 描画/保存エラー: {e}")
            import traceback
            traceback.print_exc()

    def _find_shot_folder(self, search_root, shot_num):
        """Shot番号を含むフォルダを検索"""
        if not os.path.exists(search_root): return None
        subdirs = [d for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))]
        
        candidates = [d for d in subdirs if str(shot_num) in re.findall(r'\d+', d)]
        
        if not candidates: return None
        candidates.sort(key=len)
        return os.path.join(search_root, candidates[0])