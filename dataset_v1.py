import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, file_path, block_size=16, r=0.05):
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Bỏ dòng đầu tiên
            
        self.data_blocks = []
        self.label_blocks = []
        num_lines = len(lines)
        
        # Trùng lấp 50% -> bước nhảy là block_size / 2
        step = block_size // 2  # Bước nhảy: 8 (16 / 2)
        num_blocks = (num_lines - block_size) // step + 1 if num_lines >= block_size else 0
        
        # Chia block: mỗi block có độ dài 16, bắt đầu từ i, cách nhau 8 dòng
        for i in range(num_blocks):
            start_idx = i * step
            end_idx = start_idx + block_size
            if end_idx > num_lines:  # Bỏ qua nếu block không đủ độ dài
                break
            block_lines = lines[start_idx:end_idx]

            block_data = []
            for j, line in enumerate(block_lines):
                values = list(map(float, line.strip().split()))
                inputs = values[1:9]  # (8,)
                block_data.append(inputs)

            # Tính nhãn từ dòng đầu và dòng cuối của block
            last_values = list(map(float, block_lines[-1].strip().split()))
            x_last, y_last = last_values[9], last_values[10]  # Lấy x, y thực (hector)
            x_enc_last, y_enc_last = last_values[12], last_values[13]  # Lấy x, y encoder
            first_values = list(map(float, block_lines[0].strip().split()))
            x_first, y_first = first_values[9], first_values[10]  # Lấy x, y thực (hector)
            x_enc_first, y_enc_first = first_values[12], first_values[13]  # Lấy x, y encoder

            # Tính độ chênh lệch
            dx = x_last - x_first
            dy = y_last - y_first
            dx_enc = x_enc_last - x_enc_first
            dy_enc = y_enc_last - y_enc_first

            # Tính độ trượt theo x
            if dx_enc == 0:  # Tránh chia cho 0
                lambda_x = 0
            else:
                if dx_enc > dx:
                    lambda_x = (dx_enc - dx) / dx_enc
                else:
                    lambda_x = -(dx - dx_enc) / dx if dx != 0 else 0

            # Tính độ trượt theo y
            if dy_enc == 0:  # Tránh chia cho 0
                lambda_y = 0
            else:
                if dy_enc > dy:
                    lambda_y = (dy_enc - dy) / dy_enc
                else:
                    lambda_y = -(dy - dy_enc) / dy if dy != 0 else 0

            # Cắt tỉa để đảm bảo giá trị nằm trong [-0.75, 0.75]
            lambda_x = max(-0.75, min(0.75, lambda_x))
            lambda_y = max(-0.75, min(0.75, lambda_y))

            # Lưu block và nhãn
            self.data_blocks.append(np.array(block_data, dtype=np.float32))
            self.label_blocks.append(np.array([lambda_x, lambda_y], dtype=np.float32))

        # Chuyển sang numpy array
        self.data_blocks = np.array(self.data_blocks)  # (num_blocks, block_size, 8)
        self.label_blocks = np.array(self.label_blocks)  # (num_blocks, 2)

    def __len__(self):
        return len(self.data_blocks)

    def __getitem__(self, idx):
        x = torch.tensor(self.data_blocks[idx], dtype=torch.float32)  # (block_size, 8)
        x = x.permute(1, 0)  # Chuyển thành (8, block_size) đúng định dạng cho Conv1d
        y = torch.tensor(self.label_blocks[idx], dtype=torch.float32)  # (2,)
        return x, y

def get_dataloaders(input_folder, batch_size=16, block_size=16, r=0.05):
    """
    Tạo danh sách DataLoader cho từng file trong thư mục.
    Chia file: 70% train, 20% val, 10% test.
    """
    # Lấy danh sách tất cả file trong thư mục
    all_files = [f for f in os.listdir(input_folder) if f.startswith('data_output_') and f.endswith('_full.txt')]
    all_files.sort()  # Sắp xếp file theo tên để đảm bảo thứ tự nhất quán

    # Kiểm tra số lượng file
    num_files = len(all_files)
    if num_files == 0:
        raise ValueError("Không tìm thấy file nào trong thư mục.")

    # Chia file: 70% train, 20% val, 10% test
    train_size = int(0.7 * num_files)
    val_size = int(0.2 * num_files)
    test_size = num_files - train_size - val_size  # Phần còn lại cho test

    # Điều chỉnh để đảm bảo đủ file cho train và val
    if train_size + val_size > num_files:
        val_size = num_files - train_size  # Ưu tiên train
    test_size = num_files - train_size - val_size

    # Chia danh sách file
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]

    # Tạo danh sách DataLoader cho từng tập
    train_loaders = []
    val_loaders = []
    test_loaders = []

    # Train
    for f in train_files:
        dataset = CustomDataset(os.path.join(input_folder, f), block_size, r)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append((f, loader))

    # Val
    for f in val_files:
        dataset = CustomDataset(os.path.join(input_folder, f), block_size, r)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        val_loaders.append((f, loader))

    # Test
    for f in test_files:
        dataset = CustomDataset(os.path.join(input_folder, f), block_size, r)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_loaders.append((f, loader))

    return train_loaders, val_loaders, test_loaders

# Đường dẫn thư mục chứa các file dữ liệu
input_folder = "data_19_3\\output"
output_folder = "processed_data_full"

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tạo danh sách DataLoader
train_loaders, val_loaders, test_loaders = get_dataloaders(
    input_folder, batch_size=16, block_size=16, r=0.05
)

# Lưu dữ liệu cho từng tập
# Train
for file_name, loader in train_loaders:
    file_number = file_name.split('_')[2]  # Lấy số từ tên file (data_output_i_full.txt)
    train_path = os.path.join(output_folder, f"train_data_{file_number}.txt")
    if os.path.exists(train_path):
        os.remove(train_path)
        print(f"Đã xóa file cũ: {train_path}")

# Val
for file_name, loader in val_loaders:
    file_number = file_name.split('_')[2]
    val_path = os.path.join(output_folder, f"val_data_{file_number}.txt")
    if os.path.exists(val_path):
        os.remove(val_path)
        print(f"Đã xóa file cũ: {val_path}")

# Test
for file_name, loader in test_loaders:
    file_number = file_name.split('_')[2]
    test_path = os.path.join(output_folder, f"test_data_{file_number}.txt")
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"Đã xóa file cũ: {test_path}")

print("Dữ liệu đã được lưu vào thư mục processed_data.")
print(f"Train files: {[f for f, _ in train_loaders]}")
print(f"Val files: {[f for f, _ in val_loaders]}")
print(f"Test files: {[f for f, _ in test_loaders]}")

# In 5 giá trị đầu tiên của train_loaders
print("=== In 5 giá trị đầu tiên của train_loaders ===")
for file_name, loader in train_loaders:
    print(f"\nFile: {file_name}")
    dataset = loader.dataset
    num_samples = len(dataset)
    print(f"  Tổng số block: {num_samples}")
    
    # In 5 mẫu đầu tiên từ dataset (không shuffle để đảm bảo lấy đúng 5 mẫu đầu)
    for i in range(min(5, num_samples)):
        data, label = dataset[i]  # Lấy mẫu thứ i
        print(f"  Mẫu {i + 1}:")
        print(f"    Dữ liệu (dòng đầu tiên trong block): {data[:, 0].numpy()}")
        print(f"    Nhãn (lambda_x, lambda_y): {label.numpy()}")

print("=== Kết thúc in train_loaders ===")