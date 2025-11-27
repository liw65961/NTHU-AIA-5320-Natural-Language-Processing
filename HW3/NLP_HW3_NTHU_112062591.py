from transformers import BertTokenizer, BertModel
import datasets #x
from datasets import load_dataset
from evaluate import load
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
# source ~/torch_env/bin/activate
device = "cuda" if torch.cuda.is_available() else "cpu"
#  You can install and import any other libraries if needed
print(f"Using device: {device}")
print(datasets.__version__)

# Some Chinese punctuations will be tokenized as [UNK], so we replace them with English ones
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, trust_remote_code=True, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # Replace Chinese punctuations with English ones
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# Define the hyperparameters
# You can modify these values if needed
lr = 3e-5 #default 3e-5
epochs = 4 #default 3
train_batch_size = 8 #default 8
validation_batch_size = 8 #default 8
dropout = 0.1
# grad_clip = 0.1

# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn(batch):
    # Use Chatgpt to help  implementing this function
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    # batch: list[dict] with keys: premise, hypothesis, relatedness_score, entailment_judgement
    premise_list = [ex["premise"] for ex in batch]
    hypothesis_list = [ex["hypothesis"] for ex in batch]
    relatedness_score_list = torch.tensor([float(ex["relatedness_score"]) for ex in batch], dtype=torch.float32)
    entailment_judgment_list = torch.tensor([int(ex["entailment_judgment"]) for ex in batch], dtype=torch.long)

    toks = tokenizer(
        premise_list,
        hypothesis_list,
        padding="longest",   # dynamic padding per batch
        truncation=True,
        return_tensors="pt",
    )
    return toks, relatedness_score_list, entailment_judgment_list

# Use Chatgpt to help  implementing this function
# TODO1-2: Define your DataLoader
# Write your code here
train_dataset = SemevalDataset(split="train")
validation_dataset = SemevalDataset(split="validation")
test_dataset = SemevalDataset(split="test")

dl_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,  collate_fn=collate_fn)
dl_validation = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)
dl_test = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)

# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use Chatgpt to help  implementing this function
        # Write your code here
        # Define what modules you will use in the model
        # Please use "google-bert/bert-base-uncased" model (https://huggingface.co/google-bert/bert-base-uncased)
        # Besides the base model, you may design additional architectures by incorporating linear layers, activation functions, or other neural components.
        # Remark: The use of any additional pretrained language models is not permitted.
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # self.bert = BertModel.from_pretrained("roberta-base")
        hidden = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(p=dropout)
        self.regressor = torch.nn.Linear(hidden, 1)   # why: scalar similarity 1~5
        self.classifier = torch.nn.Linear(hidden, 3)  # why: 3-way entailment
        self.hidden_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout)
        )# adding tanh nonlinear transform
    def forward(self, **kwargs):
        # Use Chatgpt to help  implementing this function
        # Write your code here
        # Forward pass
        out = self.bert(
            input_ids=kwargs.get("input_ids"),
            attention_mask=kwargs.get("attention_mask"),
            token_type_ids=kwargs.get("token_type_ids", None),
        )# (B, L, H)，B=批次、L=序列長、H=隱層（bert-base 為 768）
        
        cls = out.last_hidden_state[:, 0]    # [CLS], 取每條序列第 0 個 token（[CLS]）的向量作整段句對的表示。形狀：(B, H)。
        cls = self.dropout(cls)
        cls = self.hidden_proj(cls)# adding tanh nonlinear transform
        reg = self.regressor(cls).squeeze(-1) #線性層把 (B, H) -> (B, 1)，squeeze(-1) 變 (B,)
        logits = self.classifier(cls) #(B, H) -> (B, 3)
        return {"regression": reg, "logits": logits}
    
# TODO3: Define your optimizer and loss function
model = MultiLabelModel().to(device)
# TODO3-1: Define your Optimizer
optimizer = AdamW(model.parameters(), lr=lr)
# TODO3-2: Define your loss functions (you should have two)
# Write your code here
loss_regression = torch.nn.MSELoss()
loss_classification = torch.nn.CrossEntropyLoss()

# scoring functions
psr = load("pearsonr")
acc = load("accuracy")    

best = 0.0
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # Use Chatgpt to help  implementing this function
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    # clear gradient
    # forward pass
    # compute loss
    # back-propagation
    # model optimization
    for enc, y_reg, y_cls in pbar:
        enc = {k: v.to(device) for k, v in enc.items()}
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(**enc)
        reg_pred = out["regression"]
        logits = out["logits"]
        alpha, beta = 0.8, 1.2  # regression : classification
        loss = alpha * loss_regression(reg_pred, y_reg) + beta * loss_classification(logits, y_cls)

        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=float(loss.detach().cpu()))

    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # Use Chatgpt to help  implementing this function
    # TODO5: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    # Output all the evaluation scores (PearsonCorr, Accuracy)
    reg_preds, reg_refs = [], []
    cls_preds, cls_refs = [], []
    with torch.no_grad():
        for enc, y_reg, y_cls in pbar:
            enc = {k: v.to(device) for k, v in enc.items()}
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            out = model(**enc)
            reg_pred = out["regression"]
            pred_cls = out["logits"].argmax(dim=-1)

            reg_preds.extend(reg_pred.detach().cpu().tolist())
            reg_refs.extend(y_reg.detach().cpu().tolist())
            cls_preds.extend(pred_cls.detach().cpu().tolist())
            cls_refs.extend(y_cls.detach().cpu().tolist())

    # Output all the evaluation scores (PearsonCorr, Accuracy)
    # Use Chatgpt to help  implementing this function
    # Write your code here
    # 先在指標區域新增一行
    pearson_corr = psr.compute(references=reg_refs, predictions=reg_preds)["pearsonr"]
    accuracy = acc.compute(references=cls_refs, predictions=cls_preds)["accuracy"]
    print(f"Validation — Pearson: {pearson_corr:.4f} | Accuracy: {accuracy:.4f}")
    
    if pearson_corr + accuracy > best:
        best = pearson_corr + accuracy
        torch.save(model.state_dict(), f'./saved_models/best_model.ckpt')

# Load the model
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load(f"./saved_models/best_model.ckpt", weights_only=True))
# Test Loop
pbar = tqdm(dl_test, desc="Test")
model.eval()

# TODO6: Write the test loop
# Write your code here
# We have loaded the best model with the highest evaluation score for you
# Please implement the test loop to evaluate the model on the test dataset
# We will have 10% of the total score for the test accuracy and pearson correlation
reg_preds, reg_refs = [], []
cls_preds, cls_refs = [], []
with torch.no_grad():
    for enc, y_reg, y_cls in pbar:
        enc = {k: v.to(device) for k, v in enc.items()}
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        out = model(**enc)
        reg_pred = out["regression"]
        pred_cls = out["logits"].argmax(dim=-1)

        reg_preds.extend(reg_pred.detach().cpu().tolist())
        reg_refs.extend(y_reg.detach().cpu().tolist())
        cls_preds.extend(pred_cls.detach().cpu().tolist())
        cls_refs.extend(y_cls.detach().cpu().tolist())

test_pearson = psr.compute(references=reg_refs, predictions=reg_preds)["pearsonr"]
test_accuracy = acc.compute(references=cls_refs, predictions=cls_preds)["accuracy"]
print(f"Test — Pearson: {test_pearson:.4f} | Accuracy: {test_accuracy:.4f}")
