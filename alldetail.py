import pandas as pd
import pickle

# 加载数据
data_path = 'multihateclip_english_fold.csv'  # 您需要提供正确的文件路径
data = pd.read_csv(data_path)

data = pd.read_csv(data_path)

# 初始化数据结构
allDataAnnotation = {}
folds = ['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5']

# 提取每个fold的训练、验证和测试数据
for fold in folds:
    fold_data = data[data[fold].notna()]
    allDataAnnotation[fold] = {
        'train': (fold_data[fold_data[fold] == 'train']['Video_ID'].tolist(),
                  fold_data[fold_data[fold] == 'train']['Majority_Voting'].map({'Hateful': 0, 'Offensive': 1, 'Normal': 2}).tolist()),
        'val': (fold_data[fold_data[fold] == 'val']['Video_ID'].tolist(),
                fold_data[fold_data[fold] == 'val']['Majority_Voting'].map({'Hateful': 0, 'Offensive': 1, 'Normal': 2}).tolist()),
        'test': (fold_data[fold_data[fold] == 'test']['Video_ID'].tolist(),
                 fold_data[fold_data[fold] == 'test']['Majority_Voting'].map({'Hateful': 0, 'Offensive': 1, 'Normal': 2}).tolist())
    }


# 序列化并保存数据到pickle文件
pickle_path = 'multihateclip_allFoldDetails.p'
with open(pickle_path, 'wb') as fp:
    pickle.dump(allDataAnnotation, fp)

print("Data has been serialized to", pickle_path)
