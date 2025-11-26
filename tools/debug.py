import sys
import os
import pandas as pd
import yaml

# --- 設定セクション ---
TARGET_SHOT = 3

# 1. ライブラリ(lib)がある場所
LIB_DIR = r"C:\Users\kei89\analysis\lib"

# 2. ★重要★ プロジェクトのルートディレクトリ (yamlsフォルダがある場所の親)
# ここをあなたの実際の作業フォルダに固定します
PROJECT_ROOT = r"C:\Users\kei89\Desktop\03_実験データ\032_解析プログラム\RDTE\202511_単体燃焼冷走試験\20251102_冷走"

# 3. YAMLフォルダ名
YAML_DIR_NAME = "yamls"
# ----------------------

# ライブラリパスを通す
if LIB_DIR not in sys.path:
    sys.path.append(LIB_DIR)

try:
    # 相対インポートではなく、LIB_DIRから直接インポート
    from loader import DataLoader
except ImportError:
    print("❌ ライブラリ(loader.py)の読み込みに失敗しました。LIB_DIRを確認してください。")
    sys.exit(1)

def inspect_csv_header(shot_num):
    print(f"\n🔍 Shot {shot_num} CSV構造診断モード")
    print(f"📂 プロジェクトルート: {PROJECT_ROOT}")

    # パス解決
    yaml_root = os.path.join(PROJECT_ROOT, YAML_DIR_NAME)
    series_path = os.path.join(yaml_root, "series", "series_20251102.yaml")
    
    if not os.path.exists(series_path):
        print(f"❌ Seriesファイルが見つかりません: {series_path}")
        print("   -> PROJECT_ROOT のパスが間違っていないか確認してください。")
        return

    print(f"📄 Series定義ロード: {os.path.basename(series_path)}")
    with open(series_path, 'r', encoding='utf-8') as f: 
        series_conf = yaml.safe_load(f)
    
    # loaderに渡す base_dir も PROJECT_ROOT 起点や series.yaml 内の定義に従う必要がありますが
    # ここでは series_conf['base_dir'] が絶対パスで書かれているか、
    # あるいは series.yaml からの相対パス解決を期待しているかによります。
    # 安全のため、DataLoaderには明示的にこのconfigを渡します。
    
    loader = DataLoader(series_config=series_conf)
    
    # DataLoader内で base_dir が '.' (カレント) になっていると失敗するので、
    # もし series.yaml の base_dir が相対パスなら、PROJECT_ROOT を基準に解決するように補正します。
    if loader.base_dir == '.' or not os.path.isabs(loader.base_dir):
        # series_20251102.yaml の中身によりますが、通常はここを書き換えてあげると親切
        # 今回は loader.base_dir を PROJECT_ROOT の親（生データ置き場）に向ける必要があるかもしれません
        # ただし、DataLoaderの仕様上、series.yamlの `base_dir` が正しく設定されている前提で動きます。
        pass

    # 特に問題になりやすい「圧力データ(pressure)」を確認
    source_name = 'pressure'
    if source_name not in loader.sources:
        print(f"❌ Source '{source_name}' が設定にありません。")
        return

    print(f"  -> '{source_name}' の元ファイルを特定します...")
    
    # ファイル特定ロジック
    source_info = loader.sources[source_name]
    folder_name = source_info.get('folder')
    
    # 検索パスの構築: loader.base_dir が絶対パスならそれを使う、そうでなければ PROJECT_ROOT 基準と仮定してみる
    if os.path.isabs(loader.base_dir):
        target_dir = os.path.join(loader.base_dir, folder_name)
    else:
        # base_dir が "." の場合など、series.yaml の場所基準で考える必要があるが
        # ここでは「絶対パス」で書かれていることを期待して、見つからなければ警告を出す
        target_dir = os.path.join(loader.base_dir, folder_name)

    print(f"     (検索フォルダ: {target_dir})")

    hint = source_info.get('hint')
    csv_path = loader._smart_find_file(target_dir, shot_num, hint=hint)
    
    if not csv_path:
        print("❌ 対象のCSVファイルが見つかりませんでした。")
        print("   -> series.yaml の base_dir 設定が正しいか確認してください。")
        print("   -> フォルダパスが実際に存在するか確認してください。")
        return

    print(f"📄 発見: {os.path.basename(csv_path)}")
    print(f"📂 フルパス: {csv_path}")

    # 1. 生テキストの表示
    print("\n" + "="*60)
    print("[ファイルの先頭 20行 (Raw Text)]")
    print("="*60)
    
    raw_lines = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            for _ in range(20):
                line = f.readline()
                if not line: break
                raw_lines.append(line.rstrip())
        
        for i, line in enumerate(raw_lines):
            print(f"{i+1:02d}: {line}")
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    # 2. ヘッダー検出ロジックのシミュレーション
    print("\n" + "="*60)
    print("[プログラムによる解釈]")
    print("="*60)
    
    valid_lines = [(i, line) for i, line in enumerate(raw_lines) if line.strip()]
    sep_counts = [line.count(',') for i, line in valid_lines]
    
    if not sep_counts:
        print("  ⚠️ 有効なデータ行が見つかりません。")
        return

    max_sep = max(sep_counts)
    
    detected_header_idx = -1
    for i, line in enumerate(raw_lines):
        if line.strip() and line.count(',') == max_sep:
            detected_header_idx = i
            break
    
    if detected_header_idx != -1:
        print(f"  ✅ ヘッダーと判定された行: {detected_header_idx + 1} 行目")
        print(f"     内容: {raw_lines[detected_header_idx]}")
        
        data_start_idx = detected_header_idx + 1
        print(f"  📊 データ開始行 (推定): {data_start_idx + 1} 行目から")
        
        if len(raw_lines) > data_start_idx:
            first_data_row = raw_lines[data_start_idx]
            print(f"     実際のデータ1行目: {first_data_row}")
            
            vals = first_data_row.split(',')
            print(f"     -> 値の例: {vals[0:3]} ...")
            
            # 単位行判定チェック
            try:
                float(vals[0])
                print("     ✅ 先頭カラムは数値として変換可能です。")
            except ValueError:
                print(f"     ⚠️ 警告: 先頭カラム '{vals[0]}' は数値変換できません！")
                print("     -> これが原因で縦線が出ている可能性があります。")
                print("     -> (対策) 'start_index' を増やして、この行をスキップする必要があります。")
    else:
        print("  ⚠️ ヘッダー位置を特定できませんでした。")

    print("\n" + "-"*60)

if __name__ == "__main__":
    inspect_csv_header(TARGET_SHOT)