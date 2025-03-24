import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_v2 import train_loader, val_loader, test_loader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Chọn device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Định nghĩa mô hình 1D CNN + FC với softmax tùy chọn và thêm Dropout
class DNN1D(nn.Module):
    def __init__(self):
        super(DNN1D, self).__init__()
        # Giữ nguyên số channels nhưng thay đổi các thông số Conv1d để phù hợp với data length = 6
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Thêm dropout sau các lớp Conv (tỷ lệ 25% là ví dụ, bạn có thể điều chỉnh)
        self.dropout_conv = nn.Dropout(p=0.25)
        
        # Sau conv3, dữ liệu có shape: (batch, 64, 6)
        # Nên số đầu vào của fc1 là 6 * 64 = 384
        self.fc1 = nn.Linear(6 * 64, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 100)  # logits cho 100 lớp

        # Thêm dropout cho các lớp FC (tỷ lệ 50% là ví dụ)
        self.dropout_fc = nn.Dropout(p=0.2)

        # Bộ lọc làm mượt (smoothing) giữ nguyên
        self.smoothing = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7,
                                   stride=1, padding=3, bias=False)
        # Nếu bạn muốn sử dụng smoothing kernel tùy chỉnh, uncomment đoạn dưới:
        smoothing_kernel = torch.tensor([0, 0, 0.15, 0.7, 0.15, 0, 0], dtype=torch.float32).view(1, 1, 7)
        self.smoothing.weight = nn.Parameter(smoothing_kernel, requires_grad=False)

    def forward(self, x, apply_softmax=False):
        # Input x có shape: (batch, 8, 6)
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        
        # Phẳng hoá: (batch, 64*6 = 384)
        x = x.view(x.size(0), -1)
        
        # Qua các lớp Fully Connected với Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)             # (batch, 100) -> raw logits
        
        # Áp dụng smoothing (reshape để dùng Conv1d)
        x = x.unsqueeze(1)          # (batch, 1, 100)
        x = self.smoothing(x)       # (batch, 1, 100)
        x = x.squeeze(1)            # (batch, 100)
        
        if apply_softmax:
            x = F.softmax(x, dim=1)
        return x

# Khởi tạo model và chuyển về device
model = DNN1D().to(device)

# Sử dụng CrossEntropyLoss cho bài toán phân loại 100 lớp
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
# Sử dụng ReduceLROnPlateau để giảm learning rate khi validation loss không giảm
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

print(f"Số batch trong train_loader: {len(train_loader)}")
print(f"Số batch trong val_loader: {len(val_loader)}")

# Hàm train_model với chuyển đổi target từ giá trị liên tục sang chỉ số lớp
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, num_epochs=20, save_path='best_model.pth', device='cuda'):
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
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0

            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                # Giả sử targets ban đầu có giá trị liên tục trong khoảng [-1, 1],
                # chuyển đổi chúng thành chỉ số lớp [0, 99]
                targets = ((targets + 1) / 2 * 99).round().clamp(0, 99).long().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Không áp dụng softmax trong training vì CrossEntropyLoss xử lý nội bộ
                    outputs = model(inputs, apply_softmax=False)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                # Lưu model tốt nhất dựa trên validation loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), save_path)

                # Update scheduler dựa trên validation loss
                if scheduler:
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
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        num_epochs=20, device=device
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
