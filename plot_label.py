import matplotlib.pyplot as plt
import numpy as np
from dataset import train_loader  # hoặc dùng train_loader, val_loader, test_loader tùy nhu cầu

# Thu thập label theo thứ tự (theo thời gian)
all_labels = []
time_index = []  # Giả sử mỗi block là 1 điểm thời gian
current_time = 0

for batch in train_loader:
    # Nếu train_loader trả về tuple (inputs, labels)
    if isinstance(batch, (list, tuple)):
        _, labels = batch
    else:
        labels = batch
    labels_np = labels.cpu().numpy()
    # Giả sử mỗi sample là một block theo thứ tự thời gian
    for label in labels_np:
        all_labels.append(label)
        time_index.append(current_time)
        current_time += 1

# Vẽ đồ thị label theo thời gian
plt.figure(figsize=(12, 6))
plt.plot(time_index, all_labels, marker='o', linestyle='-', color='blue')
plt.xlabel('Thời gian (block index)')
plt.ylabel('Label')
plt.title('Đồ thị Label theo Thời Gian')
plt.grid(True)
plt.show()
