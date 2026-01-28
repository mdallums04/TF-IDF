# train_export.py
import json, joblib, numpy as np, torch, torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
rng = np.random.default_rng(SEED)

print("1")
# 1) Data
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
print("1.1")
X_raw, y = data.data, data.target
print("1.2")
num_classes = len(data.target_names)

print("2")
# 2) Vectorizer (fit once)
vectorizer = TfidfVectorizer(
    max_features=5000, lowercase=True, stop_words='english',
    strip_accents='unicode', token_pattern=r"(?u)\b\w\w+\b"
)
X = vectorizer.fit_transform(X_raw).toarray()

# 3) Split
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 4) Model
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsMLP(Xtr.shape[1], num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 5) Torch training (minimal)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)

train_ds = TensorDataset(Xtr_t, ytr_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

for epoch in range(25):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = crit(logits, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * xb.size(0)

    print(f"Epoch {epoch+1}/25 | loss={total_loss / len(train_ds):.4f}")



# 5b) Evaluation
model.eval()
with torch.no_grad():
    # Convert test data to torch tensor on the correct device
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    # Forward pass
    logits = model(Xte_t)

    # Get predicted class indices
    preds = logits.argmax(dim=1).cpu().numpy()  # move to CPU for sklearn

# Accuracy
acc = accuracy_score(yte, preds)
print(f"Test Accuracy: {acc:.4f}")

# Detailed classification report (precision, recall, F1)
report = classification_report(yte, preds, target_names=data.target_names)
print("Classification Report:\n", report)


acc = accuracy_score(yte, preds)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(yte, preds, target_names=data.target_names))

# 6) Export artifacts
torch.save(model.state_dict(), "model_state_dict.pt")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("label_names.json","w") as f: json.dump(data.target_names, f)
print("Exported: model_state_dict.pt, vectorizer.pkl, label_names.json")

