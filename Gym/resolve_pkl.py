import pickle
# 替换 'your_file.pkl' 为你的文件名
with open('q_table.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)