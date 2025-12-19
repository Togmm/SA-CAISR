import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='DIGINETICA', type=str)
args = parser.parse_args()
dir_path = f'/home/aizoo/data/usershare/wangxinru/SASRec_bp/vision_0.98/SASRec/data/{args.dataset}'

file_paths = sorted(
    [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]
)

user_set_total = set()
item_set_total = set()
tol_interactions = 0
max_user = 0
max_item = 0

print(f"{'File':<20}{'Users':>10}{'New Users':>12}{'Items':>10}{'New Items':>12}"
      f"{'Interactions':>15}{'Avg Actions/User':>20}{'Avg Actions/Item':>20}")

for file in file_paths:
    user_set = set()
    item_set = set()
    len_file = 0

    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            sessId, itemId = line.split()
            user_set.add(sessId)
            item_set.add(itemId)
            max_user = max(int(sessId), max_user)
            max_item = max(int(itemId), max_item)
            tol_interactions += 1
            len_file += 1

    # 新用户和新物品
    new_users = len(user_set - user_set_total)
    new_items = len(item_set - item_set_total)

    print(f"{os.path.basename(file):<20}{len(user_set):>10}{new_users:>12}{len(item_set):>10}"
          f"{new_items:>12}{len_file:>15}{len_file/len(user_set):>20.2f}{len_file/len(item_set):>20.2f}")

    # 更新累计集合
    user_set_total.update(user_set)
    item_set_total.update(item_set)

print("\n" + "-"*115)
print(f"{'TOTAL':<20}{len(user_set_total):>10}{'-':>12}{len(item_set_total):>10}{'-':>12}"
      f"{tol_interactions:>15}{tol_interactions/len(user_set_total):>20.2f}{tol_interactions/len(item_set_total):>20.2f}")
print(f"Max User ID: {max_user}, Max Item ID: {max_item}")
