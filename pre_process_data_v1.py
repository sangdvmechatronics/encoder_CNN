import numpy as np
import matplotlib.pyplot as plt
import os

def read_data(filename):
    data = np.loadtxt(filename, skiprows=1)  # Bỏ qua dòng tiêu đề
    t, w1, w2, x, y = data[:, 0], data[:, 7], data[:, 8], data[:, 9], data[:, 10]
    return data, t, w1, w2, x, y

def compute_trajectory(t, w1, w2, r, l):
    dt = np.diff(t, prepend=t[0])
    x, y, theta = [0], [0], [0]
    
    for i in range(1, len(t)):
        v = r * (w1[i] + w2[i]) / 2
        omega = r * (w1[i] - w2[i]) / l
        theta.append(theta[-1] + omega * dt[i])
        x.append(x[-1] + v * np.cos(theta[-1]) * dt[i])
        y.append(y[-1] + v * np.sin(theta[-1]) * dt[i])
    
    return np.array(x), np.array(y)

def save_data(filename, x_calc, y_calc):
    # Đọc lại file gốc (bao gồm tiêu đề)
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Lấy tiêu đề (dòng đầu tiên)
    header = lines[0].strip()
    if header.startswith('#'):
        header = header[1:].strip()  # Bỏ ký tự '#' nếu có
    # Thêm 2 cột mới vào tiêu đề
    new_header = header + " x_encoder y_encoder\n"
    
    # Đọc dữ liệu từ file gốc (bỏ qua tiêu đề)
    data = np.loadtxt(filename, skiprows=1)
    
    # Ghép dữ liệu gốc với x_calc, y_calc
    new_data = np.column_stack((data, x_calc, y_calc))
    
    # Tạo tên file mới: tên cũ + "full.txt"
    new_filename = filename.replace('.txt', '_full.txt')
    
    # Lưu vào file mới
    with open(new_filename, 'w') as file:
        file.write('# ' + new_header)  # Ghi tiêu đề
        np.savetxt(file, new_data, fmt='%.6f', delimiter=' ')  # Ghi dữ liệu

def plot_trajectory(x, y, x_hec, y_hec, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(x, -y, marker='o', linestyle='-', markersize=3, label='Quỹ đạo encoder')
    plt.plot(x_hec, y_hec, marker='x', linestyle='--', markersize=3, label='Quỹ đạo hector')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Dữ liệu - {filename}')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def main():
    # Đường dẫn tới thư mục chứa các file
    folder_path = "data_19_3\\output"
    
    # Tham số robot
    r = 0.05  # Bán kính bánh xe (m)
    l = 0.417  # Khoảng cách giữa hai bánh xe (m)
    
    # Duyệt qua tất cả file trong thư mục
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and 'full' not in filename:  # Chỉ xử lý file .txt và không phải file _full.txt
            filepath = os.path.join(folder_path, filename)
            print(f"Đang xử lý file: {filename}")
            
            # Đọc dữ liệu
            original_data, t, w1, w2, x_hec, y_hec = read_data(filepath)
            
            # Tính quỹ đạo
            x_calc, y_calc = compute_trajectory(t, w1, w2, r, l)
            
            # Lưu dữ liệu
            save_data(filepath, x_calc, y_calc)
            
            # Vẽ biểu đồ
            plot_trajectory(x_calc, y_calc, x_hec, y_hec, filename)

if __name__ == "__main__":
    main()