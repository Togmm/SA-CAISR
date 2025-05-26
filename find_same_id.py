import os
from collections import defaultdict

import os
from collections import defaultdict

def find_userids_and_itemids_across_files(file_list):
    # 每个 userid 和 itemid 映射到出现的文件集合
    userid_to_files = defaultdict(set)
    itemid_to_files = defaultdict(set)

    for file_path in file_list:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                userid = parts[0]
                itemid = parts[1]
                userid_to_files[userid].add(file_path)
                itemid_to_files[itemid].add(file_path)

    # 出现在多个文件中的 userids 和 itemids（即重复）
    multi_file_userids = {
        userid: files for userid, files in userid_to_files.items()
        if len(files) > 1
    }

    multi_file_itemids = {
        itemid: files for itemid, files in itemid_to_files.items()
        if len(files) > 1
    }

    # 统计信息
    total_userids = len(userid_to_files)          # 总共的 userID 数量（唯一）
    total_itemids = len(itemid_to_files)          # 总共的 itemID 数量（唯一）
    total_userid_duplicates = len(multi_file_userids)   # 出现在多个文件中的重复 userID 数量
    total_itemid_duplicates = len(multi_file_itemids)   # 出现在多个文件中的重复 itemID 数量

    return multi_file_userids, multi_file_itemids, total_userids, total_itemids, total_userid_duplicates, total_itemid_duplicates

# ✅ 示例调用方式：
file_list = []
root_path = 'E:\project\Rec\SASRec\SASRec\data\TAOBAO'
lines = os.listdir(root_path)
for line in lines:
    file_list.append(os.path.join(root_path, line))

userids, itemids, total_userids, total_itemids, total_userid_duplicates, total_itemid_duplicates = find_userids_and_itemids_across_files(file_list)

print(f"📊 统计结果：")
print(f"  👉 总共出现的唯一 userID 数量：{total_userids}")
print(f"  👉 总共出现的唯一 itemID 数量：{total_itemids}")
print(f"  🔁 出现多个文件中的重复 userID 数量：{total_userid_duplicates}")
print(f"  🔁 出现多个文件中的重复 itemID 数量：{total_itemid_duplicates}\n")

# if userids:
#     print("🔁 以下 userID 出现在多个文件中：")
#     for userid, files in sorted(userids.items()):
#         print(f"UserID {userid} 出现在文件：{', '.join(sorted(files))}")

# if itemids:
#     print("\n🔁 以下 itemID 出现在多个文件中：")
#     for itemid, files in sorted(itemids.items()):
#         print(f"ItemID {itemid} 出现在文件：{', '.join(sorted(files))}")