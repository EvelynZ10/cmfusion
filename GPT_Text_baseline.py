import os
from openai import OpenAI
from tqdm import tqdm  # 导入 tqdm 库
import numpy as np
import json
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

client = OpenAI(api_key='')
# 清理文本函数，去除重复模式和冗余内容
def clean_text(text):
    # 去除重复的单词，保留首次出现的
    words = text.split()
    cleaned_text = ' '.join(sorted(set(words), key=words.index))
    
    # 去除多余空格
    cleaned_text = ' '.join(cleaned_text.split())
    
    # 限制文本长度为 1000 字符以内（你可以调整）
    return cleaned_text[:1000]

# 加载文本和标签
def load_texts_from_folder(folder_path, label):
    texts = []
    labels = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()  # 移除首尾空白符，避免空文本
                if content:  # 仅在文件不为空时添加
                    texts.append(content)
                    labels.append(label)
                    filenames.append(filename)  # 记录文件名
    return texts, labels, filenames

# 分类文本
def classify_content(text):
    # 清理输入文本
    text = clean_text(text)

    # 生成 prompt
    prompt = f"Please determine whether the following text contains hateful content. If it contains hateful content, please return 0; if it does not contain hateful content, please return 1:\n\n{text}"

    try:
        # 调用 OpenAI 接口
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 或者使用 "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100)

        result = response.choices[0].message.content.strip()

        # 检查返回值并处理非 0 或 1 的情况
        if result not in ['0', '1']:
            return None  # 如果无效返回值，返回 None
        return int(result)

    except Exception as e:
        print(f"Error processing text: {e}")
        return None  # 如果出错，返回 None

# 保存中间结果，包括不合规的文本
def save_results(results, invalid_texts, progress, filename="results.json"):
    data = {
        "progress": progress,  # 记录处理到哪个文本
        "results": results,  # 保存的分类结果
        "invalid_texts": invalid_texts  # 记录不合规文本文件名
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

# 读取保存的中间结果
def load_results(filename="results.json"):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data["progress"], data["results"], data.get("invalid_texts", [])
    except FileNotFoundError:
        return 0, [], []  # 如果没有保存的文件，从头开始

# 评估分类结果
def evalMetric(y_true, y_pred):
    try:
        # 输出 y_true 和 y_pred 进行检查
        #print(f"y_true: {y_true}")
        #print(f"y_pred: {y_pred}")

        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score = f1_score(y_true, y_pred)
        recallScore = recall_score(y_true, y_pred)
        precisionScore = precision_score(y_true, y_pred)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return dict({"accuracy": 0, 'mF1Score': 0, 'f1Score': 0, 'precision': 0, 'recall': 0})

    return dict({
        "accuracy": accuracy,
        'mF1Score': mf1Score,
        'f1Score': f1Score,
        'precision': precisionScore,
        'recall': recallScore
    })

# 主程序
def main():
    # 加载仇恨文本和非仇恨文本
    hate_texts, hate_labels, hate_filenames = load_texts_from_folder('/home/yz1031/HateMM/hate_text', 0)
    non_hate_texts, non_hate_labels, non_hate_filenames = load_texts_from_folder('/home/yz1031/HateMM/non_hate_text', 1)

    # 合并所有文本、标签和文件名
    all_texts = hate_texts + non_hate_texts
    all_labels = hate_labels + non_hate_labels
    all_filenames = hate_filenames + non_hate_filenames

    # 打乱文本、标签和文件名
    all_texts, all_labels, all_filenames = shuffle(all_texts, all_labels, all_filenames, random_state=42)

    # 创建文件名到真实标签和预测标签的映射表
    file_to_labels = {filename: {"true_label": label, "predicted_label": None} for filename, label in zip(all_filenames, all_labels)}

    # 加载之前保存的进度和结果
    start_index, results, invalid_texts = load_results()

    # 继续从上次进度开始进行分类
    processed_count = 0
    for i in tqdm(range(start_index, len(all_texts)), desc="Processing texts", unit="text"):
        text = all_texts[i]
        filename = all_filenames[i]
        
        if len(text) > 2000:
            text = text[:2000]  # 如果文本太长，截断处理
        
        prediction = classify_content(text)
        
        if prediction is not None:  # 确保有有效的返回值
            file_to_labels[filename]["predicted_label"] = prediction  # 更新预测标签
            processed_count += 1  # 记录成功处理的文本
        else:
            # 如果不合规，记录文件名并跳过
            print(f"Skipping invalid text in file: {filename}")
            invalid_texts.append(filename)

        # 每处理 10 个文本保存一次进度和结果
        if i % 10 == 0:
            save_results(results, invalid_texts, i)

    # 提取所有对齐的真实标签和预测标签
    valid_filenames = [filename for filename, labels in file_to_labels.items() if labels["predicted_label"] is not None]
    y_true = [file_to_labels[filename]["true_label"] for filename in valid_filenames]
    y_pred = [file_to_labels[filename]["predicted_label"] for filename in valid_filenames]


    # 评估分类结果
    metrics = evalMetric(np.array(y_true), np.array(y_pred))

    # 输出评估指标和处理的文本总数
    print(metrics)
    print(f"Total processed texts: {processed_count}")
    print(f"Invalid texts: {invalid_texts}")

    # 保存最终结果
    save_results(results, invalid_texts, len(all_texts))

if __name__ == "__main__":
    main()
