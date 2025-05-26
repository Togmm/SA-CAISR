import pandas as pd
import os
import csv

def split_by_data(file_path):
    df = pd.read_csv(file_path)
    # 确保时间列是 datetime 类型（只保留日期部分）
    df["date"] = pd.to_datetime(df["time"].astype(str).str[:10])  # 只取 YYYY-MM-DD

    # 创建存储目录
    output_dir = "E:\project\Rec\SASRec\SASRec\data"
    os.makedirs(output_dir, exist_ok=True)

    # 按日期分组并保存
    for date, group in df.groupby(df["date"]):
        date_str = date.strftime("%Y-%m-%d")  # 只保留 "YYYY-MM-DD" 格式
        file_name = os.path.join(output_dir, f"data_{date_str}.csv")  # 确保无非法字符
        group.drop(columns=["date"]).to_csv(file_name, index=False)
        print(f"保存文件: {file_name}")

def filter_user_item_id(file_path, out_path):
    info = []
    user_ids = set()
    item_ids = set()
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            info.append([row[0], row[1]]) 
    
    with open(out_path, 'w') as f:
        for user_id, item_id in info:
            user_ids.add(user_id)
            item_ids.add(item_id)
            f.write(user_id + '\t' + item_id + '\n')
    print(f'user_tol_num : {len(user_ids)}, item_tol_num : {len(item_ids)}')
        
    print('filter successful')

def count(file_path):
    user_ids = set()
    item_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            user_id, item_id = line.split(' ')
            user_ids.add(user_id)
            item_ids.add(item_id)
    print(f'user_tol_num : {len(user_ids)}, item_tol_num : {len(item_ids)}')


if __name__ == '__main__':
    file_path = 'E:\project\Rec\SASRec\SASRec\data\data_2014-11-18.csv'
    out_path = 'E:\project\Rec\SASRec\SASRec\data\data_2014-11-18.txt'
    # filter_user_item_id(file_path, out_path)
    count('E:\project\Rec\SASRec\SASRec\data\Beauty.txt')
    