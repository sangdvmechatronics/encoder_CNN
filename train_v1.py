import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_v2 import train_loaders, val_loaders, test_loaders
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_v1 import *
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Chọn device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# Khởi tạo model và chuyển về device
model = CNN1D().to(device)

# Sử dụng CrossEntropyLoss cho bài toán phân loại 100 lớp
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
# Sử dụng ReduceLROnPlateau để giảm learning rate khi validation loss không giảm
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Tính tổng số mẫu trong train_loaders và val_loaders
total_train_samples = sum(len(loader.dataset) for _, loader in train_loaders)
total_val_samples = sum(len(loader.dataset) for _, loader in val_loaders)

print(f"Số file trong train_loaders: {len(train_loaders)}")
print(f"Tổng số mẫu trong train_loaders: {total_train_samples}")
print(f"Số file trong val_loaders: {len(val_loaders)}")
print(f"Tổng số mẫu trong val_loaders: {total_val_samples}")

# Hàm train_model đã sửa đổi để xử lý hai đầu ra (i_x và i_y)
def train_model(model, train_loaders, val_loaders, criterion, optimizer,
                scheduler=None, num_epochs=35, save_path='best_model.pth', device='cuda'):
    model.to(device)
    train_losses, val_losses = [], []
    best_loss = float('inf')
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # Vòng lặp qua train và validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloaders = train_loaders
                total_samples = total_train_samples
            else:
                model.eval()
                dataloaders = val_loaders
                total_samples = total_val_samples

            running_loss = 0.0

            for _, dataloader in dataloaders:
                for inputs, targets in dataloader:
                    inputs = inputs.to(device)
                    # targets: (batch_size, 2) - [i_x, i_y], đã là số nguyên trong [0, 100]
                    i_x_labels = targets[:, 0].to(device)  # (batch_size,)
                    i_y_labels = targets[:, 1].to(device)  # (batch_size,)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        i_x_pred, i_y_pred = model(inputs)  # i_x_pred, i_y_pred: (batch_size, 101)
                        loss_x = criterion(i_x_pred, i_x_labels)
                        loss_y = criterion(i_y_pred, i_y_labels)
                        loss = (loss_x + loss_y) / 2  # Trung bình hai mất mát

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / total_samples
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), save_path)  # Lưu mô hình tốt nhất

                scheduler.step(epoch_loss)

    print(f'Best Validation Loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

# Thực thi train
if __name__ == "__main__":
    # Load lại weight từ file nếu có
    save_path = 'best_model.pth'
    try:
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded pre-trained weights from {save_path}")
    except FileNotFoundError:
        print("No pre-trained weights found, training from scratch.")
    
    model, train_losses, val_losses = train_model(
        model, train_loaders, val_loaders,
        criterion, optimizer, scheduler,
        num_epochs = 35, save_path=save_path, device=device
    )

    # Vẽ biểu đồ Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.savefig("training_validation_loss.png", dpi=300, bbox_inches='tight')
    plt.show()