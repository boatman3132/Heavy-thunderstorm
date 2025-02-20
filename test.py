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
import csv
import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import pandas as pd


# =============================================================================
# 全域參數設定 (使用相對路徑)
# =============================================================================
EXFILE_DIR = "./exfile"
STATION_INFO_FILE = os.path.join(EXFILE_DIR, "station_info_UTF8.txt")
STATION_GRID_INFO_FILE = os.path.join(EXFILE_DIR, "grid_station_info_UTF8.txt")
RAILWAY_REGION_FILE = os.path.join(EXFILE_DIR, "railway_region.txt")
RAIL_MAP_IMAGE_FILE = os.path.join(EXFILE_DIR, "test-Photoroom.png")
# 警報記錄與輸出圖檔存放目錄
ALARMFILE_DIR = "./TS_alarm"

# LINE Notify Token 設定 (根據使用對象分內部與對外)
LINE_NOTIFY_TOKEN_PUBLIC = 'token_to_check'   # 對外群組
LINE_NOTIFY_TOKEN_INTERNAL = 'token_to_check'  # 台鐵值班群組

# 分支線 (僅用於判斷全線或區段)
BRANCH_LINES = np.array(["平溪線", "內灣線", "集集線", "深澳線", "六家線", "成追線", "沙崙線"])

# =============================================================================
# 輔助函式
# =============================================================================
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

import numpy as np



def load_station_data(station_file):
    """
    讀取臺鐵站點資訊，並解析經緯度資料。

    :param station_file: 車站資訊檔案路徑
    :return: station_data (NumPy 陣列)
    """

    print(f"📌 正在讀取車站資訊: {station_file}...")

    # 🚀 **讀取檔案，確保分隔符為 '\t'**
    station_info = np.loadtxt(station_file, delimiter='\t', skiprows=1, dtype=str)

    # ✅ **檢查欄位數是否符合預期**
    if station_info.shape[1] < 6:
        raise ValueError(f"❌ `{station_file}` 欄位數不足，應至少有 6 欄，實際有 {station_info.shape[1]} 欄！")

    # ✅ **正確解析車站資料**
    station_data = []
    for row in station_info:
        try:
            line_name = row[0]  # 路線名稱
            station_name = row[3]  # 站名
            lon = float(row[4])  # 經度
            lat = float(row[5])  # 緯度
            show = int(row[6])  # 是否顯示（0 or 1）
            station_data.append((line_name, station_name, lon, lat, show))
        except ValueError:
            print(f"⚠️ 跳過錯誤資料: {row}")

    print(f"✅ {station_file} 讀取完成，共 {len(station_data)} 站")

    return np.array(station_data, dtype=object)  # 🚀 **確保回傳 NumPy 陣列**

def compute_station_coordinates(station_info):
    """
    解析車站資訊，並將經緯度轉換為地圖上的座標。

    :param station_info: 車站資訊 NumPy 陣列
    :return: list[tuple] [(line_name, station_name, lon, lat, show)]
    """

    print("📌 正在處理車站座標資訊...")

    # 確保 `station_info` 至少有 5 欄 (lineName, stationName, Lon, Lat, Show)
    if station_info.shape[1] < 5:
        raise ValueError(f"❌ `station_info` 欄位數不足，應至少有 5 欄，實際有 {station_info.shape[1]} 欄！")

    station_data = []
    for row in station_info:
        try:
            line_name = row[0]  # 路線名稱
            station_name = row[1]  # 站名
            lon = float(row[2])  # 經度
            lat = float(row[3])  # 緯度
            show = int(row[4])  # 是否顯示

            station_data.append((line_name, station_name, lon, lat, show))
        except ValueError as e:
            print(f"⚠️ 車站座標錯誤: {row}，錯誤: {e}")

    print(f"✅ 車站座標處理完成，共 {len(station_data)} 站")
    return station_data



def load_grid_data(station_grid):
    """
    解析網格站點資料，並轉換經緯度座標 (僅供後續判斷多邊形包含關係)
    :param station_grid: 網格資料陣列
    :return: 字典，包含 stationName_grid, x_grid, y_grid
    """
    stationName_grid = station_grid[:, 3]
    x_grid = station_grid[:, 4].astype(float)
    y_grid = station_grid[:, 5].astype(float)
    return {
        "stationName_grid": stationName_grid,
        "x_grid": x_grid,
        "y_grid": y_grid
    }

def group_stations_by_line(station_info):
    """
    依據 lineName 將所有車站依序分組。
    :param station_info: 站點資訊陣列
    :return: 字典 {lineName: list of (index, stationName)}
    """
    line_stations = {}
    for i, line_name in enumerate(station_info[:, 0]):
        if line_name not in line_stations:
            line_stations[line_name] = []
        line_stations[line_name].append((i, station_info[i, 3]))
    return line_stations

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

def extract_alarm_message(wr, tt0, tt1):
    """
    從警報描述中擷取部分關鍵文字，並組合時間資訊成訊息。
    :param wr: 警報資料 (list)
    :param tt0: 生效時間字串
    :param tt1: 過期時間字串
    :return: 組合後的訊息字串 (linemsg_str)
    """
    msg0 = wr[0]['description']
    # 擷取關鍵字位置 (依據符號位置做字串切割)
    m1 = msg0.find('，')
    m2 = msg0.find('；')
    m3 = msg0.find('。')
    if m3 == -1:
        m3 = msg0.find('，', m2+1)
        if m3 == -1:
            m3 = len(msg0)
    msg1 = f"{tt0} {msg0[m1-12:m1+6]} {tt1}"  # 時間資訊
    msg2 = msg0[m2:m3]  # 潛在災害訊息
    linemsg_str = msg1 + msg2
    return linemsg_str

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

def generate_affected_string(line_stations, affected_indices):
    """
    根據分組後的線路車站資料與受影響站點索引，產生「影響鐵路區間」的文字訊息，
    同時判斷是否有臺鐵受影響。
    :param line_stations: 字典，格式 {lineName: list of (index, stationName)}
    :param affected_indices: 受影響站點索引列表
    :return: (affected_str, tra_affected) 其中 affected_str 為組合後文字訊息，
             tra_affected 為布林值 (True 表示有臺鐵受影響)
    """
    tra_affected = False
    affected_str = "\n\n影響鐵路區間："
    first_line_printed = False
    for line_name, stations in line_stations.items():
        # 篩選出該線中受影響的站點 (依 index)
        stations_in_polygon = [(idx, name) for idx, name in stations if idx in affected_indices]
        if stations_in_polygon:
            tra_affected = True
            # 若為分支線，僅顯示線名；若全部站點受影響則標註「全線」
            if line_name in BRANCH_LINES:
                if first_line_printed:
                    affected_str += "、"
                affected_str += f"{line_name}"
                first_line_printed = True
            elif len(stations_in_polygon) == len(stations):
                if first_line_printed:
                    affected_str += "、"
                affected_str += f"{line_name} (全線)"
                first_line_printed = True
            else:
                if first_line_printed:
                    affected_str += "、"
                sorted_stations = sorted(stations_in_polygon, key=lambda x: x[0])
                first_station = sorted_stations[0][1]
                last_station = sorted_stations[-1][1]
                if first_station == last_station:
                    affected_str += f"{line_name} ({first_station})"
                else:
                    affected_str += f"{line_name} ({first_station}-{last_station})"
                first_line_printed = True
    if not tra_affected:
        affected_str = "\n\n對臺鐵路無影響"
    return affected_str, tra_affected

def generate_qpf_message(tra_affected, qpf_time_str, QPF1, QPF2):
    """
    依據降雨預報數值與是否有臺鐵受影響，組合 QPF 訊息文字。
    :param tra_affected: 布林值，表示是否有臺鐵受影響
    :param qpf_time_str: 降雨時間字串 (例如 "12:30")
    :param QPF1: 第一百分位降雨量
    :param QPF2: 第二百分位降雨量
    :return: qpf_str (文字訊息)
    """
    if tra_affected:
        if QPF1 >= 0 and QPF2 >= 0 and QPF1 != QPF2:
            qpf_str = f"\n\n{qpf_time_str}起一小時內受影響路段降雨可能達{QPF1:.0f}~{QPF2:.0f}mm"
        elif QPF1 == QPF2:
            qpf_str = f"\n\n{qpf_time_str}起一小時內受影響路段降雨可能達{QPF1:.0f}mm"
        else:
            qpf_str = f"\n\n{qpf_time_str}起一小時內降雨無法被估計或受影響路段無顯著降雨"
    else:
        if QPF1 >= 0 and QPF2 >= 0 and QPF1 != QPF2:
            qpf_str = f"\n\n{qpf_time_str}起一小時內受影響區域降雨可能達{QPF1:.0f}~{QPF2:.0f}mm"
        elif QPF1 == QPF2:
            qpf_str = f"\n\n{qpf_time_str}起一小時內受影響區域降雨可能達{QPF1:.0f}mm"
        else:
            qpf_str = f"\n\n{qpf_time_str}起一小時內降雨無法被估計或無顯著降雨"
    return qpf_str

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import os



def plot_alarm_map(wpoly, radar_image, rail_map_image, radar_colorbar, figdir, tt0, ttR):
    """
    繪製警報範圍地圖，並儲存圖片。

    :param wpoly: 警報範圍多邊形座標
    :param radar_image: 雷達回波影像
    :param rail_map_image: 鐵路地圖影像
    :param radar_colorbar: 雷達顏色條
    :param figdir: 圖片儲存目錄
    :param tt0: 警報時間
    :param ttR: 雷達回波時間
    :return: 圖片儲存路徑
    """
    print("📌 正在繪製警報範圍地圖...")

    # 建立圖表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 處理警報範圍座標
    wpoly_mod = np.array(wpoly)
    wpoly_mod[:, 0] = (wpoly_mod[:, 0] - 118) * 600
    wpoly_mod[:, 1] = 3600 - (wpoly_mod[:, 1] - 20.5) * 600

    # 繪製雷達回波圖與鐵路地圖
    ax.imshow(radar_image, alpha=0.55)
    ax.imshow(rail_map_image, extent=[1800-480*1.69, 1800+480*1.69, 1800+640*1.6, 1800-640*1.785], alpha=0.8)

    # 繪製警報範圍多邊形
    poly = Polygon(wpoly_mod, closed=True, facecolor="red", alpha=0.3, edgecolor="darkred", linewidth=2)
    ax.add_patch(poly)

    # 隱藏橫軸與豎軸的數字
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
    cb_ax = fig.add_axes([0.25, 0.05, 0.5, 0.05])  # [left, bottom, width, height]，可依需求調整
    cb_ax.imshow(radar_colorbar)
    cb_ax.axis('off')

    # 儲存圖片
    output_path = f"{figdir}/TS{tt0.replace(':', '').replace(' ', '').replace('-', '')}_R{ttR.replace(':', '').replace(' ', '').replace('-', '')}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"✅ 地圖已儲存至: {output_path}")
    return output_path



def send_line_notification(message, image_path, token):
    """
    發送 LINE Notify 訊息 (圖片為可選)

    :param message: 文字訊息
    :param image_path: 圖檔路徑，若為 None 則不附圖
    :param token: LINE Notify 權杖
    """
    linemsg_url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + token}
    data = {'message': message}

    if image_path and os.path.isfile(image_path):
        with open(image_path, 'rb') as img:
            files = {'imageFile': img}
            r = requests.post(linemsg_url, headers=headers, data=data, files=files)
    else:
        r = requests.post(linemsg_url, headers=headers, data=data)

    print("LINE Notify 回應:", r.text)


# =============================================================================
# 主流程
# =============================================================================
import os
import logging

def main():
    """
    主要流程，不發送 LINE 訊息，而是將所有需要傳送的訊息記錄在本地終端與 log 檔案。
    """

    print("🚀 啟動大雷雨警報系統...")

    # 設定車站資訊檔案路徑
    station_info_file = "station_info_UTF8.txt"  # 請確保這個檔案存在
    station_grid_file = "grid_station_info_UTF8.txt"

    # 讀取站點資料 (✅ 正確傳入 `station_file`)
    station_info_raw = load_station_data(station_info_file)
    station_grid_raw = load_station_data(station_grid_file)

    print(f"✅ 車站資訊讀取完成，共 {len(station_info_raw)} 站")


    # 取得 CWA 警報資料
    cwa_url = 'https://cbph.cwa.gov.tw/api/cells/?order=asc&offset=0&limit=20'
    response = requests.get(cwa_url)
    wr = response.json()

    # 檢查是否為新警報
    t_lst_str, alarm_id = check_new_alarm(wr, ALARMFILE_DIR)
    figdir = prepare_output_directories(ALARMFILE_DIR, t_lst_str)

    # 讀取站點資料 (✅ 確保 `station_data` 變數存在)
    station_data = compute_station_coordinates(station_info_raw)  # 🚀 這一行很重要！

    # 解析警報範圍
    wpoly = parse_polygon(wr[0]['polygon'])
    tt0, tt1 = generate_time_strings(wr)
    linemsg_str = extract_alarm_message(wr, tt0, tt1)

    # 取得雷達圖
    radar_image, ttR, radar_colorbar = load_radar_data()
    rail_map_image = Image.open(RAIL_MAP_IMAGE_FILE)

    # **✅ 修正：確保 `station_data` 正確傳遞**
    output_image_path = plot_alarm_map(wpoly, radar_image, rail_map_image, radar_colorbar, figdir, tt0, ttR)

    print(f"📂 圖片儲存路徑: {output_image_path}")
    print("✅ 系統執行完成！")




if __name__ == '__main__':
    main()
