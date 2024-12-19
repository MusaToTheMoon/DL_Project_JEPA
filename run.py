import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Optional
from tqdm import tqdm
from dataset import create_wall_dataloader
from models import NonRecurrentJEPA

def train_jepa(
    data_path,
    epochs=1,
    batch_size=64,
    lr=1e-3,
    device="cuda",
    save_path="best_model.pth" 
):
    train_loader = create_wall_dataloader(data_path=data_path, probing=False, device=device, batch_size=batch_size, train=False)

    model = NonRecurrentJEPA(dropout_rate=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")  
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader):
            states, _, actions = batch.states, batch.locations, batch.actions

            optimizer.zero_grad()
            loss = model.compute_loss(states, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path)
                print(f"Checkpoint saved at epoch {epoch+1} with loss {avg_loss:.4f}")

    print("Training complete. Best model saved to:", save_path)
    return model

path = "../../DL24FA/train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = train_jepa(data_path=path, epochs=1, batch_size=64, lr=1e-3, device=device, save_path="best_model.pth")

