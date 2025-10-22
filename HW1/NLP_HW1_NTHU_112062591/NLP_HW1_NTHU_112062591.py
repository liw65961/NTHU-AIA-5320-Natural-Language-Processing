import pandas as pd
import numpy as np
import gensim.downloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import multiprocessing as mp
from pathlib import Path

import torch
print(torch.cuda.is_available())   # True 代表能用 GPU
print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱


## Part I: Data Pre-processing
file_name = "questions-words"
with open(f"{file_name}.txt", "r") as f:
    data = f.read().splitlines()

# check data from the first 10 entries
for entry in data[:10]:
    print(entry)

# TODO1: Write your code here for processing data to pd.DataFrame
# Please note that the first five mentions of ": " indicate `semantic`,
# and the remaining nine belong to the `syntatic` category.
# --- Parse `data` (list of lines) into pairs and build columns ---

questions = []
categories = []
sub_categories = []

block_idx = 0          # 已遇到的 ":" 段落數
current_cat = None     # 當前段落的分類
current_subcat = None  # 當前段落的子類別

for entry in data:
    s = str(entry).strip()
    if not s:
        continue

    if s.startswith(":"):  # 新段落
        block_idx += 1
        current_cat = "semantic" if block_idx <= 5 else "syntatic"  # 注意題目拼法
        current_subcat = s[:].strip()  
        # print(current_subcat)
        continue

    # 若尚未遇到任何 ":"，略過資料行
    if current_cat is None:
        continue

    # 這裡依你的設計：整行當作一個 Question
    questions.append(s)
    categories.append(current_cat)
    sub_categories.append(current_subcat)


# Create the dataframe
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

df.head()

df.to_csv(f"{file_name}.csv", index=False)

## Part II: Use pre-trained word embeddings

data = pd.read_csv("questions-words.csv")

MODEL_NAME = "glove-wiki-gigaword-100"
# You can try other models.
# https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models

# Load the pre-trained model (using GloVe vectors here)
model = gensim.downloader.load(MODEL_NAME)
print("The Gensim model loaded successfully!")

# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
      # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      a, b, c, d = analogy.strip().split()

      # 全部轉小寫，避免 OOV
      a, b, c, d = a.lower(), b.lower(), c.lower(), d.lower()

      try:
            pred = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
      except KeyError:
            pred = None  # 如果還是 OOV，就跳過
      preds.append(pred)
      golds.append(d)

      # word_a, word_b, word_c, word_d = map(str.lower, analogy.split())
      # this_pred = model.most_similar([word_b, word_c], word_a, topn=1)[0][0]
      
      # preds.append(this_pred)
      # golds.append(word_d)

# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO3: Plot t-SNE for the words in the SUB_CATEGORY `: family`
#(use Chatgpt to help write the code)
# 取出 family 子類別
sub_category_data = data[data.SubCategory == SUB_CATEGORY]
sub_category_data_str = " ".join(sub_category_data.Question)

# 先 split 再小寫 + 去重
words = np.unique(np.array(sub_category_data_str.split()))
words = [w.lower() for w in words]

# 只保留模型裡有的詞（避免 OOV）
words = [w for w in words if w in model.key_to_index]

print(f"[{SUB_CATEGORY}] words in model: {len(words)}")

# 向量矩陣
X = np.array([model[w] for w in words])

# t-SNE：perplexity 必須 < n_samples
perp = max(2, min(30, len(words) - 1))
embedded = TSNE(n_components=2, init="pca", random_state=42, perplexity=perp).fit_transform(X)

# 繪圖
plt.figure(figsize=(16, 12))
plt.scatter(embedded[:, 0], embedded[:, 1], s=20)

for idx, (x, y) in enumerate(embedded):
    plt.annotate(words[idx], (x + 0.04, y), fontsize=9)
    
plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")

### Part III: Train your own word embeddings

# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.

import random
wiki_txt_path = "wiki_texts_combined.txt"
output_path = "wiki_texts_20pct.txt"
p = 0.2
random.seed(42)  # for reproducibility

with open(wiki_txt_path, "r", encoding="utf-8") as f:
    with open(output_path, "w", encoding="utf-8") as output_file:
    # TODO4: Sample `20%` Wikipedia articles
    # Write your code here
        for line in f:
            if random.random() < p:  # ~20% Bernoulli sampling
                output_file.write(line)

print(f"Done. Wrote ~20% to: {output_path}")


# TODO5: Train your own word embeddings with the sampled articles
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# Hint: You should perform some pre-processing before training.

#Preprocess the sampled corpus and train Word2Vec embeddings(use Chatgpt to help write the code)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
STOP_NLTK = set(stopwords.words("english"))

CUSTOM_STOP = {"wikipedia", "http", "https", "www", "category", "ref"}
KEEP = {"not", "no"}  # 想保留的否定詞
STOP = (STOP_NLTK | CUSTOM_STOP) - KEEP

sampled_path = "wiki_texts_20pct.txt"

path = Path(sampled_path)  
# 計算總行數
with path.open("r", encoding="utf-8", errors="ignore") as f:
    total_lines = sum(1 for _ in f)
print(f"Total lines: {total_lines:,}")

max_lines = None  # 想全量就改 None

class SentenceIter:
    def __init__(self, path, max_lines=None):
        self.path = path
        self.max_lines = max_lines
    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_lines is not None and i >= self.max_lines:
                    break
                # 先做基本清理（小寫、去標點），再只留純英文 token
                toks = simple_preprocess(line, deacc=True, min_len=2, max_len=15)
                # 只留 ASCII 英文字母，且排除 NLTK stopwords
                toks = [t for t in toks if t.isascii() and t.isalpha() and t not in STOP]
                if toks:
                    yield toks

sentences = SentenceIter(sampled_path, max_lines=max_lines)  
w2v = Word2Vec(
    sentences=sentences,
    vector_size=150, 
    window=10, 
    min_count=20,
    sg=0, #sg=0用「周圍的字」去預測中心詞，速度快、較省資源；對高頻詞效果好；大語料常用作預設/baseline。sg=1用「中心詞」去預測周圍的字特性：較慢、但對低頻詞/小語料常比較好；語義/類比任務有時表現更佳。
    negative=20, 
    sample=1e-3,
    workers=max(1, mp.cpu_count()-1),
    epochs=2, 
    seed=42,
    batch_words=10000,
)
model = w2v.wv
print("vocab size:", len(model.key_to_index))
model.save("word2vec_wiki_20pct_ef.model")

data = pd.read_csv("questions-words.csv")
# Do predictions and preserve the gold answers (word_D)
from gensim.models import KeyedVectors
preds = []
golds = []

model = KeyedVectors.load("word2vec_wiki_20pct_ef.model")

for analogy in tqdm(data["Question"]):
      # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      a, b, c, d = analogy.strip().split()

      # 全部轉小寫，避免 OOV
      a, b, c, d = a.lower(), b.lower(), c.lower(), d.lower()

      try:
            pred = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
      except KeyError:
            pred = None  # 如果還是 OOV，就跳過
      preds.append(pred)
      golds.append(d)

# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO7: Plot t-SNE for the words in the SUB_CATEGORY `: family`
#(use Chatgpt to help write the code)
# 取出 family 子類別
sub_category_data = data[data.SubCategory == SUB_CATEGORY]
sub_category_data_str = " ".join(sub_category_data.Question)

# 先 split 再小寫 + 去重
words = np.unique(np.array(sub_category_data_str.split()))
words = [w.lower() for w in words]

# 只保留模型裡有的詞（避免 OOV）
words = [w for w in words if w in model.key_to_index]

print(f"[{SUB_CATEGORY}] words in model: {len(words)}")

# 向量矩陣
X = np.array([model[w] for w in words])

# t-SNE：perplexity 必須 < n_samples
perp = max(2, min(30, len(words) - 1))
embedded = TSNE(n_components=2, init="pca", random_state=42, perplexity=perp).fit_transform(X)

# 繪圖
plt.figure(figsize=(16, 12))
plt.scatter(embedded[:, 0], embedded[:, 1], s=20)

for idx, (x, y) in enumerate(embedded):
    plt.annotate(words[idx], (x + 0.04, y), fontsize=9)


plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")