#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TR_TSalarm.py

說明:
    此程式用於自動偵測大雷雨警報，並依警報內容判斷受影響區域與臺鐵沿線站點，
    進而產生對應的圖檔與文字訊息，最後透過 LINE Notify 發送訊息（附圖）通知相關人員。
    
主要功能包含：
    1. 從 CWA 取得雷雨警報與降雨預報（QPF）資料
    2. 檢查是否已有相同警報記錄，若無則存檔（避免重複通知）
    3. 讀取站點與網格資料，根據警報範圍判斷受影響的車站
    4. 依受影響狀況組合文字訊息（包含影響範圍、降雨量預報等）
    5. 讀取雷達回波圖、色階條與鐵路地圖，繪製受影響區域圖
    6. 發送 LINE Notify 訊息（附圖），區分對內與對外通知
    
檔案路徑採用相對路徑，請將所有相關檔案置於同一資料夾或其子目錄中：
    exfile/          ← 放置 station_info_UTF8.txt、grid_station_info_UTF8.txt、railway_region.txt、test-Photoroom.png、msjhbd.ttc 等
    TS_alarm/        ← 警報記錄與輸出圖檔存放目錄

注意:
    若未來需改用 LINE Messenger，可將 send_line_notification() 函式內傳送部分註解取消，並調整相關參數。

@author: user
@date: 2024-07-04
"""

import os
import sys
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from PIL import Image
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import pandas as pd


# =============================================================================
# 全域參數設定 (使用相對路徑)
# =============================================================================
EXFILE_DIR = "./exfile"
# 警報記錄與輸出圖檔存放目錄
ALARMFILE_DIR = "./TS_alarm"

# 字型檔案與此 Python 檔案位於同一個目錄下
font_path = os.path.join(os.path.dirname(__file__), "STHeiti Medium.ttc")
title_font = fm.FontProperties(fname=font_path, size=20)
# =============================================================================
# 輔助函式
# =============================================================================

def load_rail_map_image():
    """
    從 Taiwan_rail_map.svg 載入鐵路地圖影像，並轉換為 PIL Image 物件以便進行繪圖。
    這樣在放大時可保有更高解析度的清晰度。
    """
    import io
    try:
        import cairosvg
    except ImportError:
        raise ImportError("需要安裝 cairosvg 模組以轉換 SVG 檔案，請執行：pip install cairosvg")
    
    # 定義 SVG 檔案的路徑（請確認 Taiwan_rail_map.svg 放置於 exfile 目錄中）
    svg_path = os.path.join(EXFILE_DIR, "Taiwan_rail_map.svg")
    
    # 使用 cairosvg 將 SVG 轉換為 PNG 格式的二進位資料
    png_data = cairosvg.svg2png(url=svg_path)
    
    # 透過 BytesIO 將二進位資料轉換為 PIL Image 物件
    rail_map_image = Image.open(io.BytesIO(png_data))
    
    return rail_map_image


def swap_columns(matrix, col1, col2):
    """
    將二維串列中指定的兩個欄位交換順序。
    :param matrix: 二維串列，每個元素為一個 list
    :param col1: 欲交換的第一欄索引
    :param col2: 欲交換的第二欄索引
    """
    for row in matrix:
        row[col1], row[col2] = row[col2], row[col1]

def loadCWAQPF(poly, R1, R2):
    """
    取得中央氣象局 QPF 資料，並依據警報區域計算降雨百分位數。
    
    :param poly: 警報影響區域的多邊形座標 (list of [lat, lon])
    :param R1: 目標百分位數 (例如 50)
    :param R2: 目標百分位數 (例如 95)
    :return: (time_str, QPF1, QPF2) -> 預報時間、QPF 50%、QPF 95%
    """

    # 1. 下載 CWA QPF 資料
    url = 'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-B0046-001?Authorization=rdec-key-123-45678-011121314&format=JSON'
    response = requests.get(url)
    data = response.json()

    # 2. 解析 JSON 並轉換時間 (修正時區)
    content = data['cwaopendata']['dataset']['contents']['content']
    time_full = data['cwaopendata']['dataset']['datasetInfo']['parameterSet']['DateTime']
    time_str = (pd.to_datetime(time_full, utc=True) + pd.Timedelta(hours=8)).strftime('%H:%M')

    # 3. 轉換 QPF 資料陣列
    c = np.fromstring(content, sep=',', dtype=float)
    c[c < 0] = np.nan  # 將負值轉為 NaN
    cc = c.reshape((561, 441))  # 確保形狀一致

    # 4. 建立緯度、經度網格 (修正步長一致性)
    x = np.arange(118, 123.5 + 0.0125, 0.0125)
    y = np.arange(20, 27 + 0.0125, 0.0125)
    xx, yy = np.meshgrid(x, y)

    # 5. 建立多邊形物件
    poly_obj = plt.Polygon(poly, closed=True)
    points = np.vstack((xx.ravel(), yy.ravel())).T
    path = Path(poly_obj.get_xy())

    # 6. 確認點是否在多邊形內
    mask = path.contains_points(points).reshape(xx.shape)

    # 7. 讀取鐵路區域資料並匹配 `mask` (修正形狀問題)
    all_railway = np.loadtxt("railway_region.txt", dtype=int)

    # 確保 `mask` 與 `all_railway` 維度一致
    if mask.shape != all_railway.shape:
        min_shape = (min(mask.shape[0], all_railway.shape[0]), min(mask.shape[1], all_railway.shape[1]))
        mask = mask[:min_shape[0], :min_shape[1]]
        all_railway = all_railway[:min_shape[0], :min_shape[1]]

    # 8. 計算 QPF 值
    affected_railway = all_railway * mask.astype(int)
    affected_railway_bool = affected_railway.astype(bool)

    if np.all(np.isnan(cc)):  # 如果所有值皆為 NaN，回傳 -999
        QPF1, QPF2 = -999., -999.
    elif not np.any(affected_railway_bool):  # 如果沒有影響臺鐵線路，則用多邊形內的降雨量
        QPF1 = np.nanpercentile(cc[mask], R1)
        QPF2 = np.nanpercentile(cc[mask], R2)
    else:  # 影響臺鐵線路的情況
        QPF1 = np.nanpercentile(cc[affected_railway_bool], R1)
        QPF2 = np.nanpercentile(cc[affected_railway_bool], R2)

    # 9. 四捨五入至 5 的倍數
    QPF1 = np.round(QPF1 / 5) * 5
    QPF2 = np.round(QPF2 / 5) * 5

    return time_str, QPF1, QPF2

def check_new_alarm(wr, alarmfile_dir):
    """
    檢查是否已有相同警報記錄，若有則退出程式；若無則將警報資料存檔以避免重複通知。
    :param wr: 從 CWA 取得的警報資料 (list)
    :param alarmfile_dir: 警報記錄存放目錄
    :return: (t_lst_str, alarm_id)
    """
    # 將 UTC 時間轉為本地時間 (加 8 小時)
    t_str = datetime.strptime(str(wr[0]['effective']), "%Y-%m-%dT%H:%M:%SZ")
    t_local = t_str + timedelta(hours=8)
    t_lst_str = t_local.strftime("%Y%m%d")
    alarm_id = wr[0]['id']
    
    # 建立存放警報記錄的目錄 (list/)
    alarmfile_list = os.path.join(alarmfile_dir, "list")
    if not os.path.exists(alarmfile_list):
        os.makedirs(alarmfile_list)
    checkfile_path = os.path.join(alarmfile_list, f"{t_lst_str}_{alarm_id}.json")
    
    if os.path.isfile(checkfile_path):
        sys.exit("No new alarm!!!")  # 已有記錄則退出
    else:
        with open(checkfile_path, 'w', encoding='utf-8') as json_file:
            json.dump(wr[0], json_file, ensure_ascii=False, indent=4)
    
    return t_lst_str, alarm_id

def prepare_output_directories(alarmfile_dir, t_lst_str):
    """
    根據警報日期建立存放結果圖檔的目錄 (格式：YYYY/MM/DD/)
    :param alarmfile_dir: 警報存放主目錄
    :param t_lst_str: 警報日期字串 (YYYYMMDD)
    :return: figdir (結果圖檔存放目錄)
    """
    year = t_lst_str[:4]
    month = t_lst_str[4:6]
    day = t_lst_str[6:8]
    figdir = os.path.join(alarmfile_dir, year, month, day)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    return figdir



def parse_polygon(polygon_str):
    """
    將警報資料中的 polygon 字串轉換為座標列表，並調整經緯度順序 (lat, lon)。
    :param polygon_str: 字串，格式如 "lon,lat lon,lat ..."
    :return: 座標列表，格式 [[lat, lon], ...]
    """
    # 以空白分隔各點，再以逗號拆分數值
    wpoly = [list(map(float, point.split(','))) for point in polygon_str.split()]
    # 原資料為 (lon, lat)，交換順序使其變成 (lat, lon)
    swap_columns(wpoly, 0, 1)
    return wpoly

def generate_time_strings(wr):
    """
    根據警報資料中的 effective 與 expires 欄位產生文字用時間字串。
    :param wr: 警報資料 (list)
    :return: (tt0, tt1) 字串，分別代表警報生效時間與過期時間 (格式依需求調整)
    """
    # effective 轉為本地時間 (加 8 小時)
    tt0 = (np.datetime64(wr[0]['effective']) + np.timedelta64(8, 'h')).astype(str)
    tt0 = tt0[:16].replace('T', ' ')
    # expires 取時分
    tt1 = (np.datetime64(wr[0]['expires']) + np.timedelta64(8, 'h')).astype(str)
    tt1 = tt1[11:16]
    return tt0, tt1


def determine_affected_stations(poly, grid_data, station_data):
    """
    根據警報區域多邊形，從網格站點資料判定哪些站點位於警報區內，
    再根據網格判定結果，從 station_data 中挑出受影響站點索引，
    並加入鄰近車站以補足斷層。
    :param poly: 多邊形物件 (matplotlib.patches.Polygon)
    :param grid_data: 字典，包含 x_grid, y_grid, stationName_grid
    :param station_data: 字典，包含 stationName (原始順序)
    :return: 受影響站點索引列表 (all_station_indices)
    """
    x_grid = grid_data["x_grid"]
    y_grid = grid_data["y_grid"]
    stationName_grid = grid_data["stationName_grid"]
    all_station_indices = []
    
    # 先建立網格中是否位於多邊形內的布林陣列
    all_grid_contain = np.zeros(len(x_grid), dtype=bool)
    for i in range(len(x_grid)):
        if poly.contains_point((x_grid[i], y_grid[i])):
            all_grid_contain[i] = True
    
    # 若某站點的名稱存在於網格資料中受影響的部分，視為該站受影響
    for i, name in enumerate(station_data["stationName"]):
        if name in stationName_grid[all_grid_contain]:
            all_station_indices.append(i)
    
    # 檢查是否需要補加鄰近站 (加入相鄰站若尚未納入)
    extra_indices = []
    for idx in all_station_indices:
        # 若前一站存在且同一線且未加入則補加
        if idx != 0 and station_data["lineName"][idx] == station_data["lineName"][idx-1] and (idx-1) not in all_station_indices:
            extra_indices.append(idx-1)
        # 若後一站存在且同一線且未加入則補加
        if idx != len(station_data["stationName"]) - 1 and station_data["lineName"][idx] == station_data["lineName"][idx+1] and (idx+1) not in all_station_indices:
            extra_indices.append(idx+1)
    merged_list = list(set(all_station_indices + extra_indices))
    merged_list.sort()
    return merged_list

def load_radar_data():
    """
    從中央氣象局取得雷達回波圖與色階條，並處理色階條透明度調整。
    :return: (radar_image, ttR, radar_colorbar)
    """
    # 取得雷達回波資料
    radar_url_api = 'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/O-A0058-006?Authorization=rdec-key-123-45678-011121314&format=JSON'
    response = requests.get(radar_url_api)
    SR = response.json()
    radar_img_url = SR['cwaopendata']['dataset']['resource']['ProductURL']
    radar_image = Image.open(requests.get(radar_img_url, stream=True).raw)
    ttR = SR['cwaopendata']['dataset']['DateTime']
    ttR = ttR[:10] + ' ' + ttR[11:16]
    
    # 取得雷達色階條
    radar_colorbar_url = 'https://www.cwa.gov.tw/V8/assets/img/radar/colorbar_n.png'
    radar_colorbar = Image.open(requests.get(radar_colorbar_url, stream=True).raw).convert("RGBA")
    radar_colorbar_array = np.array(radar_colorbar)
    # 定義需要調整透明度的區域 (根據圖檔大小調整)
    x_start, y_start, x_end, y_end = 1, 1, 1124, 68
    alpha_mask = np.ones((radar_colorbar.height, radar_colorbar.width), dtype=np.float32)
    alpha_mask[y_start:y_end, x_start:x_end] = 0.55  # 設定透明度 0.55
    alpha_layer = (alpha_mask * 255).astype(np.uint8)
    rgba_image_array = np.dstack((radar_colorbar_array[:, :, :3], alpha_layer))
    radar_colorbar = Image.fromarray(rgba_image_array)
    
    return radar_image, ttR, radar_colorbar
def plot_alarm_map(wpoly, radar_image, rail_map_image, radar_colorbar, figdir, tt0, ttR):
    """
    繪製警報範圍地圖，將由 Taiwan_rail_map.svg 轉換後的底圖置於最上層，
    雷達回波圖疊在下方，並在圖片上方加入標題，格式類似於：
    2024/12/07 21:02 大雷雨影響範圍
    """
    print("📌 正在繪製警報範圍地圖...")

    # 建立圖表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 處理警報範圍座標
    wpoly_mod = np.array(wpoly)
    wpoly_mod[:, 0] = (wpoly_mod[:, 0] - 118) * 600
    wpoly_mod[:, 1] = 3600 - (wpoly_mod[:, 1] - 20.5) * 600

    # 先繪製雷達回波圖 (放在較底層)
    ax.imshow(radar_image, alpha=0.55, zorder=1)

    # 再繪製鐵路地圖 (放在最上層)
    ax.imshow(rail_map_image,
              extent=[1800-480*1.69, 1800+480*1.69, 1800+640*1.6, 1800-640*1.785],
              alpha=0.8,
              zorder=2)

    # 繪製警報範圍多邊形
    poly = Polygon(wpoly_mod, closed=True, facecolor="red", alpha=0.3, edgecolor="darkred", linewidth=2)
    ax.add_patch(poly)

    # 隱藏橫軸與縱軸的數字
    ax.set_xticks([])
    ax.set_yticks([])

    # 設定地圖顯示範圍（以避免過度放大）
    min_x, max_x = np.min(wpoly_mod[:, 0]), np.max(wpoly_mod[:, 0])
    min_y, max_y = np.min(wpoly_mod[:, 1]), np.max(wpoly_mod[:, 1])
    mid_x, mid_y = (max_x + min_x) / 2, (max_y + min_y) / 2
    radius = max((max_x - min_x), (max_y - min_y)) * 13
    ax.set_xlim(mid_x - radius, mid_x + radius)
    ax.set_ylim(mid_y + radius, mid_y - radius)

    # 新增一個位於圖片下方的座標軸，顯示雷達色階圖例
    cb_ax = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    cb_ax.imshow(radar_colorbar)
    cb_ax.axis('off')

    # 使用 STHeiti Medium.ttc 字型設定標題
    font_path = os.path.join(EXFILE_DIR, "STHeiti Medium.ttc")  # 確保字型檔案在 exfile 目錄下
    title_font = fm.FontProperties(fname=font_path, size=24)  # 字體大小可依需求調整

    # 將 tt0 格式轉換為 "YYYY/MM/DD HH:MM" 格式，並加上標題內容
    tt0_date = tt0[:10].replace('-', '/')
    tt0_time = tt0[11:16]
    alert_title = f"{tt0_date} {tt0_time} 大雷雨影響範圍"
    fig.suptitle(alert_title, fontproperties=title_font, y=0.95)

    # 儲存圖片
    output_path = f"{figdir}/TS{tt0.replace(':', '').replace(' ', '').replace('-', '')}_R{ttR.replace(':', '').replace(' ', '').replace('-', '')}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"✅ 地圖已儲存至: {output_path}")
    return output_path



# =============================================================================
# 主流程
# =============================================================================

def main():
    """
    主要流程，不發送 LINE 訊息，而是將所有需要傳送的訊息記錄在本地終端與 log 檔案。
    """

    print("🚀 啟動大雷雨警報系統...")


    # 取得 CWA 警報資料
    cwa_url = 'https://cbph.cwa.gov.tw/api/cells/?order=asc&offset=0&limit=20'
    response = requests.get(cwa_url)
    wr = response.json()

    # 檢查是否為新警報
    t_lst_str, alarm_id = check_new_alarm(wr, ALARMFILE_DIR)
    figdir = prepare_output_directories(ALARMFILE_DIR, t_lst_str)


    # 解析警報範圍
    wpoly = parse_polygon(wr[0]['polygon'])
    tt0, tt1 = generate_time_strings(wr)

    # 取得雷達圖
    radar_image, ttR, radar_colorbar = load_radar_data()
    rail_map_image = load_rail_map_image()

    # **✅ 修正：確保 `station_data` 正確傳遞**
    output_image_path = plot_alarm_map(wpoly, radar_image, rail_map_image, radar_colorbar, figdir, tt0, ttR)

    print(f"📂 圖片儲存路徑: {output_image_path}")
    print("✅ 系統執行完成！")

if __name__ == '__main__':
    main()
