import numpy as np
import matplotlib.pyplot as plt

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

def save_data(filename, original_data, x, y):
    header = " " .join(["col" + str(i) for i in range(original_data.shape[1])]) + " x_calc y_calc"
    data = np.column_stack((original_data, x, y))
    np.savetxt(filename, data, header=header, comments='')

def plot_trajectory(x, y, x_hec, y_hec):
    plt.figure(figsize=(8, 6))
    plt.plot(x, -y, marker='o', linestyle='-', markersize=3, label='Quỹ đạo encoder')
    plt.plot(x_hec, y_hec, marker='x', linestyle='--', markersize=3, label='Quỹ đạo hector')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Dữ liệu')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def main():
    file_number = input("Nhập số cho tên file: ")  # Người dùng nhập số
    # for i in range(1, 10):
        # file_number = i
    filename = f"data_19_3\\output\\data_output_{file_number}.txt" 
    save_path = f"data_19_3\\output\\data_output_processed_{file_number}.txt"

    r = 0.05  # Bán kính bánh xe (m)
    l = 0.417  # Khoảng cách giữa hai bánh xe (m)
    
    original_data, t, w1, w2, x_hec, y_hec = read_data(filename)
    x, y = compute_trajectory(t, w1, w2, r, l)
    
    save_data(save_path, original_data, x, y)
    plot_trajectory(x, y, x_hec, y_hec)

if __name__ == "__main__":
    main()
