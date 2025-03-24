import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

# Đường dẫn tới folder train
train_folder = "processed_data\\test"  # Thay bằng đường dẫn thực tế

# Hàm đọc nhãn từ file và mô phỏng train_loader
def read_labels_from_file(filepath):
    # Đọc file với delimiter là dấu phẩy
    data = np.loadtxt(filepath, delimiter=',')
    # Giả định cột cuối cùng là nhãn (có thể thay đổi tùy cấu trúc file)
    labels = data[:, -1]
    # Chuyển thành tensor PyTorch để giống train_loader
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return labels_tensor

# Hàm thu thập nhãn giống cách xử lý train_loader
def collect_labels_from_file(filepath):
    labels_tensor = read_labels_from_file(filepath)
    all_labels = []
    time_index = []
    current_time = 0
    
    # Mô phỏng batch từ train_loader (giả lập batch size = 1)
    for label in labels_tensor:
        all_labels.append(label.item())  # Chuyển từ tensor sang scalar
        time_index.append(current_time)
        current_time += 1
    
    return time_index, all_labels

# Thu thập nhãn từ tất cả file trong folder
def collect_labels_from_folder(folder_path):
    all_file_labels = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Chỉ xử lý file .txt
            filepath = os.path.join(folder_path, filename)
            time_index, labels = collect_labels_from_file(filepath)
            all_file_labels[filename] = (time_index, labels)
    return all_file_labels

# Vẽ tất cả nhãn vào các subplot
def plot_labels_in_subplots(all_file_labels):
    num_files = len(all_file_labels)
    if num_files == 0:
        print("Không tìm thấy file .txt nào trong folder.")
        return
    
    # Xác định số hàng và cột cho subplot
    num_cols = int(np.ceil(np.sqrt(num_files)))
    num_rows = int(np.ceil(num_files / num_cols))
    
    # Tạo figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4), squeeze=False)
    fig.suptitle("Đồ thị Label theo Thời Gian từ Các File", fontsize=16)
    
    # Vẽ từng subplot
    for idx, (filename, (time_index, labels)) in enumerate(all_file_labels.items()):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        
        ax.plot(time_index, labels, marker='o', linestyle='-', color='blue')
        # ax.set_xlabel('Thời gian (sample index)')
        # ax.set_ylabel('Label')
        ax.set_title(f'{filename}')
        ax.grid(True)
    
    # Ẩn subplot dư
    for idx in range(num_files, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Hàm chính
def main():
    all_file_labels = collect_labels_from_folder(train_folder)
    plot_labels_in_subplots(all_file_labels)

if __name__ == "__main__":
    main()