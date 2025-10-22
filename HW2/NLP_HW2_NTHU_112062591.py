# ! pip install seaborn
# ! pip install opencc
# ! pip install -U scikit-learn
# train 2369251 eval 263251
# source ~/torch_env/bin/activate
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

data_path = './data'

df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
# df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_divide_train.csv'))
df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))
df_train.head()

# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))

char_to_id = {}
id_to_char = {}

# write your code here
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

def build_char_vocab():
    """Why: 最通用，對任何表達式(含空白)都適用。"""
    digits = [str(id) for id in range(10)]
    ops = ["+", "-", "*", "(", ")", "="]
    # ops = ["+", "-", "*", "/", "(", ")", "="]
    special_tokens = ["<pad>", "<eos>"]
    # space = [" "]  # 若你的資料沒有空白，可移除
    alphabet = digits + ops + special_tokens
    c2i = {ch: id for id, ch in enumerate(alphabet, start = 0)} # enumerate(alphabet, start = 0) 回傳 1. index 2. item, start 是 id 從 start 開始算
    i2c = {id: ch for ch, id in c2i.items()} # items() 回傳 key, value
    return c2i, i2c

char_to_id, id_to_char = build_char_vocab()

vocab_size = len(char_to_id)

print("Vocab size: {}".format(vocab_size))
print(char_to_id)
print(id_to_char)

# write your code here
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

src = df_train['src'].tolist()
tgt = df_train['tgt'].tolist()
length = df_train['len'].tolist()
char_id_list = []
label_id_list = []

for index, row in df_train.iterrows():
    src_seq = [char_to_id[char] for char in row['src']] + [char_to_id['<eos>']]
    idx = src_seq.index(char_to_id["="])
    tgt_seq = [char_to_id['<pad>']] * idx + src_seq[idx:]
    char_id_list.append(src_seq)
    label_id_list.append(tgt_seq)
    # print('label_id_list: {}, char_id_list: {}'.format(tgt_seq, src_seq))

df_train['char_id_list'] = char_id_list
df_train['label_id_list'] = label_id_list

df_train.head()

# hyperparameters
hidden_dim = 384
lstm_layers_1 = 1
lstm_layers_2 = 2
dropout = 0
embed_dim = 256
batch_size = 64
lr = 0.001
grad_clip = 1.0
epochs = 3

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        """
        sequences 可為：
          - pandas.DataFrame（含欄位 'char_id_list','label_id_list'）
          - List[Dict[str, List[int]]]（含同名鍵）
          - List[Tuple[List[int], List[int]]]（(x_ids, y_ids)）
        """
        self.sequences = sequences
    
    def __len__(self):
        # return the amount of data
        return len(self.sequences)
    
    def __getitem__(self, index):
        # Extract the input data x and the ground truth y from the data
        x = self.sequences["char_id_list"][index][:-1]
        y = self.sequences["label_id_list"][index][1:]
        return x, y

# collate function, used to build dataloader
# 把一個 batch 的不等長序列轉成 等長張量
def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch] # Tensor : 32 bit integer 張量
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x]) # LongTensor : 64 bit integer 張量
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y]) # embedding 與 CrossEntropyLoss 需要 long 索引
    
    # Pad the input sequence
    # 用 <pad> 右側補齊序列 → 產生 batch_first=True (pad_batch_x, pad_batch_y) 形狀 (B, T_max), 若果 batch_first=False(default)，則形狀為 (T_max, B)
    # B=此批次的樣本數（batch size），T_max=這個批次裡最長序列的長度（經 padding 後）
    # 原始序列長度各不相同：[L1, L2, ..., LB]，T_max = max(L1..LB)；短的右側補 <pad> 到長度 T_max。
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])
    
    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])
    
    #另回 (batch_x_lens, batch_y_lens) 保存每條原始長度。
    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])

# Build dataloader of train set and eval set, collate_fn is the collate function
dl_train = torch.utils.data.DataLoader(
    ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True
)# Write your code here

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()
        
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])
        
        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True,
                                        num_layers=lstm_layers_1,
                                        dropout=dropout)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True,
                                        num_layers=lstm_layers_2,
                                        dropout=dropout)
        
        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))
        
    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)
    
    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)
        
        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)
        
        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)
        
        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
                                                            batch_first=True)
        
        batch_x = self.linear(batch_x)
        
        return batch_x

    def generator(self, start_char, max_len=200):
        # Here use chatGPT to help writing the generator 
        # print(start_char)
        device = next(self.parameters()).device  # 獲取模型的設備
        char_list = [char_to_id[c] for c in start_char]

        next_char = None

        while len(char_list) < max_len:
            x = torch.tensor([char_list], dtype=torch.long, device=device)  # (1, T)
            lens = torch.tensor([len(char_list)], dtype=torch.long, device='cpu')  # (1,) 1D CPU int64

            logits = self.encoder(x, lens)  # (1, T, V)
            y = logits[:, -1, :]  # 取最後一個時間步的 logits (1, V)

            # print(y.shape)

            # print(y)

            next_char = torch.argmax(y).item()

            if next_char == char_to_id["<eos>"]:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]
    
torch.manual_seed(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# Write your code here. Specify a device (cuda or cpu)
print("cuda" if torch.cuda.is_available() else "cpu")

model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim)

criterion = torch.nn.CrossEntropyLoss(ignore_index = char_to_id["<pad>"]) # Write your code here. Cross-entropy loss function. The loss function should ignore <pad>
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2) # Write your code here. Use Adam or AdamW for Optimizer
# 若想用 Adam（無權重衰減）
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from tqdm import tqdm
from copy import deepcopy
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Here use chatGPT to help writing the training process 
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad(set_to_none=True)
    
        batch_pred_y = model(batch_x.to(device), batch_x_lens)
        
        # Write your code here
        # Input the prediction and ground truths to loss function
        V = batch_pred_y.size(-1)
        loss = criterion(
            batch_pred_y.view(-1, V),           # (B*T, V)
            batch_y.to(device).view(-1)         # (B*T,)
        )
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())
    
    # Evaluate your model
    matched = 0
    total = 0
    bar_eval = tqdm(df_eval.iterrows(), desc=f"Validation epoch {epoch}")
    for _, row in bar_eval:
        batch_x = row['src']
        batch_y = row['tgt']
        # An example of using generator: model.generator('1+1=')
        # Write your code here. Input the batch_x to the model and generate the predictions
        # 使用 model.generator 時，將 batch_x_tensor 傳入
        prediction = "".join(model.generator(batch_x))
        prediction = prediction.split("=", 1)[1] if "=" in prediction else prediction
        # prediction = prediction.replace("<eos>", "")  # 保險處理
        # Write your code here.
        # Check whether the prediction match the ground truths
        # print('src: {}, tgt: {}, pred: {}'.format(batch_x, batch_y, prediction))
        is_match = int(prediction == batch_y)
        matched += is_match
        total += 1
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total
        EM = matched/total if total > 0 else 0
        bar_eval.set_postfix(EM=EM)

        
    print(matched/total)

# torch.save(model, "LSTM_arithmetic_divide_train.pth")
# print("Full model saved as LSTM_arithmetic_divide_train.pth")
