from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


class RankingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]['query']
        document = self.data.iloc[idx]['document']
        score = self.data.iloc[idx]['score']
        inputs = self.tokenizer(
            query + " [SEP] " + document,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'score': torch.tensor(score, dtype=torch.float)
        }


column_names = ['query', 'document', 'score']
df = pd.read_csv('../data/train/top100_train.csv', names=column_names)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = RankingDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


class RankingModel(nn.Module):
    def __init__(self):
        super(RankingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.score_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        score = self.score_classifier(pooled_output)
        return score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RankingModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = MSELoss()
num_epochs = 5
accumulation_steps = 8  # Adjust based on your memory capacity and batch size
scaler = GradScaler()  # For mixed precision training

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)):
        with autocast():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['score'].to(device)
            predictions = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(predictions.squeeze(), scores) / accumulation_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0 or step + 1 == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (step + 1) % 10 == 0:
            print(f'Step {step + 1}, Current Loss: {loss.item()}')

    print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}')


