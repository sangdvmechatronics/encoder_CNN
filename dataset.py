import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

file_number = input("Nhập số cho tên file: ")  # Người dùng nhập số

class CustomDataset(Dataset):
    def __init__(self, file_path, block_size=32, r=0.05):

        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Bỏ dòng đầu tiên
            
        self.data_blocks = []
        self.label_blocks = []
        # Tính số block tối đa
        num_lines = len(lines)
        num_blocks = num_lines // block_size

        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_lines = lines[start_idx:end_idx]

            block_data = []
            dencoder = 0  # Khởi tạo dencoder
            pre_t = 0  # Giá trị thời gian trước đó (t_i-1)
            
            
            for j, line in enumerate(block_lines):
                values = list(map(float, line.strip().split()))
                inputs = values[1:9]  # (8,)
                # print(inputs)
                block_data.append(inputs)
            # Nhãn từ dòng cuối cùng của block
            last_values = list(map(float, block_lines[-1].strip().split()))
            x_last, y_last = last_values[9], last_values[10]  # Lấy x, y từ dòng cuối
            x_enc_last, y_enc_last = last_values[12], last_values[13]  # Lấy x, y từ dòng cuối
            first_values = list(map(float, block_lines[0].strip().split()))
            x_first, y_first = first_values[9], first_values[10]  # Lấy x, y từ dòng đầu
            x_enc_first, y_enc_first = first_values[12], first_values[13]  # Lấy x, y từ dòng đầu

            dx = x_last - x_first
            dy = y_last - y_first
            dx_enc = x_enc_last - x_enc_first
            dy_enc = y_enc_last - y_enc_first
            dthuc = np.sqrt(dx**2 + dy**2)  # Khoảng cách thực
            dencoder = np.sqrt(dx_enc**2 + dy_enc**2)  # Khoảng cách thực
            # dencoder *= r / 2  # Hiệu chỉnh dencoder
            # Tránh chia cho 0
            if dencoder != 0:
                block_label = (dencoder - dthuc) / dencoder if dthuc <= dencoder else -(dthuc - dencoder) / dthuc
            else:
                block_label = 0  # Nếu dencoder = 0, tránh lỗi chia cho 0


            block_label = round((block_label-0.005 + 0.5)*100)
            #print('block_label: ',block_label)
            self.data_blocks.append(np.array(block_data, dtype=np.float32))
            self.label_blocks.append(np.array(block_label, dtype=np.float32))

        # Chuyển sang numpy array
        self.data_blocks = np.array(self.data_blocks)  # (num_blocks, 32, 8)
        self.label_blocks = np.array(self.label_blocks)  # (num_blocks,)

    def __len__(self):
        return len(self.data_blocks)

    def __getitem__(self, idx):
        x = torch.tensor(self.data_blocks[idx], dtype=torch.float32)  # (32, 8)
        x = x.permute(1, 0)  # Chuyển thành (8, 32) đúng định dạng cho Conv1d
        y = torch.tensor(self.label_blocks[idx], dtype=torch.float32)
        return x, y

def save_dataset_to_txt(dataloader, save_path):
    with open(save_path, 'w') as f:
        for batch in dataloader:
            data, labels = batch  # Lấy dữ liệu và nhãn từ batch
            data = data.numpy()  # Chuyển tensor sang numpy
            labels = labels.numpy()

            for i in range(len(labels)):  # Duyệt từng sample trong batch
                first_row = data[i, :, 0]  # Lấy dòng đầu tiên (sequence index = 0)
                last_row = data[i, :, -1]  # Lấy dòng cuối cùng (sequence index = -1)
                label = labels[i]  # Nhãn của block

                # Chuyển dữ liệu thành chuỗi
                first_row_str = ','.join(map(str, first_row))
                last_row_str = ','.join(map(str, last_row))

                # Ghi vào file: dòng đầu, dòng cuối, label
                f.write(f"{first_row_str},{last_row_str},{label}\n")

def get_dataloaders(file_path, batch_size= 16, block_size =6, train_ratio=0.8, val_ratio=0.1, r=0.05):
    """
    Tạo DataLoader từ file dữ liệu.
    """
    dataset = CustomDataset(file_path, block_size, r)

    # Chia dữ liệu thành train, val, test
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Định nghĩa đường dẫn file dữ liệu
file_path = f"data_19_3\\output\\data_output_processed_{file_number}.txt"

# Tạo DataLoader
train_loader, val_loader, test_loader = get_dataloaders(file_path, batch_size=16, block_size= 6, r=0.05)

train_path = f"processed_data/train_data_{file_number}.txt"
val_path = f"processed_data/val_data_{file_number}.txt"
test_path = f"processed_data/test_data_{file_number}.txt"

# Xóa file nếu tồn tại
for path in [train_path, val_path, test_path]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Đã xóa file cũ: {path}")

# Lưu dữ liệu mới
save_dataset_to_txt(train_loader, train_path)
save_dataset_to_txt(val_loader, val_path)
save_dataset_to_txt(test_loader, test_path)

print("Dữ liệu đã được lưu vào file.")

# Lưu dữ liệu train, val, test vào các file khác nhau
