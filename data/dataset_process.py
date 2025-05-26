import os
from glob import glob

# 指定目录路径（替换成你自己的）
input_dir = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/data/TAOBAO'
output_dir = '/data/wangxinru-slurm/project/Rec/SASRec/SASRec/data/TAOBAO_process'
os.makedirs(output_dir, exist_ok=True)

user2id = {}
item2id = {}
next_user_id = 1
next_item_id = 1

file_list = []
for i in range(6):
    file_list.append(f'/data/wangxinru-slurm/project/Rec/SASRec/SASRec/data/TAOBAO/period_{i}.txt')
# 首先扫描所有文件，建立全局 user/item 映射字典
for file in file_list:
    with open(file, 'r') as f:
        for line in f:
            user, item = line.strip().split()
            if user not in user2id:
                user2id[user] = next_user_id
                next_user_id += 1
            if item not in item2id:
                item2id[item] = next_item_id
                next_item_id += 1

# 应用映射，保存离散化文件
for file in file_list:
    output_file = os.path.join(output_dir, os.path.basename(file))
    with open(file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            user, item = line.strip().split()
            uid = user2id[user]
            iid = item2id[item]
            fout.write(f"{uid}\t{iid}\n")

print(f"离散化完成，共计 {len(user2id)} 个用户，{len(item2id)} 个物品。")
print(f"输出路径：{output_dir}")