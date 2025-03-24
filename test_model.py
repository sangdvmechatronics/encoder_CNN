import numpy as np
import torch
import torch.nn as nn
from dataset_v2 import test_loaders  # Import test_loaders từ dataset_v2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_v1 import CNN1D  # Import mô hình CNN1D từ model_v1
import os

# Đặt biến môi trường để debug lỗi CUDA (nếu cần)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Chọn device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Khởi tạo mô hình và chuyển về device
model = CNN1D().to(device)

# Đường dẫn đến file trọng số đã huấn luyện
save_path = 'best_model.pth'

# Load trọng số đã huấn luyện
try:
    model.load_state_dict(torch.load(save_path))
    print(f"Loaded pre-trained weights from {save_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"Pre-trained weights not found at {save_path}. Please train the model first.")

# Đặt mô hình ở chế độ đánh giá (evaluation)
model.eval()

# Tính tổng số mẫu trong test_loaders
total_test_samples = sum(len(loader.dataset) for _, loader in test_loaders)
print(f"Số file trong test_loaders: {len(test_loaders)}")
print(f"Tổng số mẫu trong test_loaders: {total_test_samples}")

# Hàm test và vẽ biểu đồ
def test_model(model, test_loaders, device='cuda'):
    model.to(device)
    
    # Duyệt qua từng file trong test_loaders
    for file_idx, (file_name, dataloader) in enumerate(test_loaders):
        print(f"\nTesting on file: {file_name}")
        
        all_i_x_labels = []
        all_i_y_labels = []
        all_i_x_preds = []
        all_i_y_preds = []

        # Duyệt qua từng batch trong dataloader
        with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm bộ nhớ
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                i_x_labels = targets[:, 0].to(device)  # Nhãn thực tế i_x
                i_y_labels = targets[:, 1].to(device)  # Nhãn thực tế i_y

                # Dự đoán
                i_x_pred, i_y_pred = model(inputs)  # i_x_pred, i_y_pred: (batch_size, 101)
                _, i_x_pred = torch.max(i_x_pred, 1)  # Lấy chỉ số lớp có xác suất cao nhất
                _, i_y_pred = torch.max(i_y_pred, 1)

                # Chuyển về CPU và lưu lại để vẽ biểu đồ
                all_i_x_labels.extend(i_x_labels.cpu().numpy())
                all_i_y_labels.extend(i_y_labels.cpu().numpy())
                all_i_x_preds.extend(i_x_pred.cpu().numpy())
                all_i_y_preds.extend(i_y_pred.cpu().numpy())

        # Vẽ biểu đồ so sánh nhãn thực tế và nhãn dự đoán
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Biểu đồ cho i_x
        ax1.plot(all_i_x_labels, label='True i_x', color='blue', marker='o', linestyle='-', markersize=5)
        ax1.plot(all_i_x_preds, label='Predicted i_x', color='red', marker='x', linestyle='--', markersize=5)
        ax1.set_title(f'File: {file_name} - i_x: True vs Predicted')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('i_x Value')
        ax1.legend()
        ax1.grid(True)

        # Biểu đồ cho i_y
        ax2.plot(all_i_y_labels, label='True i_y', color='blue', marker='o', linestyle='-', markersize=5)
        ax2.plot(all_i_y_preds, label='Predicted i_y', color='red', marker='x', linestyle='--', markersize=5)
        ax2.set_title(f'File: {file_name} - i_y: True vs Predicted')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('i_y Value')
        ax2.legend()
        ax2.grid(True)

        # Điều chỉnh layout và lưu biểu đồ
        plt.tight_layout()
        output_file = f"test_predictions_{file_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.show()

# Thực thi test
if __name__ == "__main__":
    test_model(model, test_loaders, device=device)
    print("Testing completed!")