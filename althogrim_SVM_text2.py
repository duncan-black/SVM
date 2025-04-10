import csv
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu từ file huấn luyện (để tìm w và b)
def load_training_data(filename):
    """Đọc dữ liệu huấn luyện từ file CSV."""
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            try:
                weight = float(row[0])
                size = float(row[1])
                label = row[2]

                X.append([weight, size])
                if label == 'orange':
                    y.append(-1)
                elif label == 'apple':
                    y.append(1)
                else:
                    print(f"Nhãn không hợp lệ: {label}")
                    continue
            except ValueError as e:
                print(f"Lỗi chuyển đổi kiểu dữ liệu: {e}, dòng: {row}")
                continue
    return np.array(X), np.array(y)

# 2. Tính w và b từ dữ liệu huấn luyện (sử dụng trung bình)
def calculate_w_and_b(X, y):
    """Tính w và b dựa trên trung bình của dữ liệu."""
    apple_indices = np.where(y == 1)
    orange_indices = np.where(y == -1)

    if len(apple_indices[0]) == 0 or len(orange_indices[0]) == 0:
        print("Không đủ dữ liệu để tính w và b.")
        return None, None

    mean_apple = np.mean(X[apple_indices], axis=0)
    mean_orange = np.mean(X[orange_indices], axis=0)

    w = mean_apple - mean_orange
    b = np.dot(w, (mean_apple + mean_orange) / 2)

    return w, b

# 3. Đọc dữ liệu kiểm tra (chỉ weight và size)
def load_data_no_labels(filename):
    """Đọc dữ liệu kiểm tra từ file CSV (chỉ weight và size)."""
    X = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            try:
                weight = float(row[0])
                size = float(row[1])
                X.append([weight, size])
            except ValueError as e:
                print(f"Lỗi chuyển đổi kiểu dữ liệu: {e}, dòng: {row}")
                continue
    return np.array(X)

# 4. Gán nhãn dựa trên w và b
def assign_labels(X, w, b):
    """Gán nhãn dựa trên w và b."""
    y = []
    for x_i in X:
        prediction = np.dot(x_i, w) - b
        label = 1 if prediction > 0 else -1
        y.append(label)
    return np.array(y)

# 5. Lưu dữ liệu đã gán nhãn vào file
def save_labeled_data(X, y, output_filename='labeled_data.csv'):
    """Lưu dữ liệu đã gán nhãn vào file CSV."""
    with open(output_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['weight', 'size', 'label'])  # Ghi header

        for i in range(X.shape[0]):
            weight, size = X[i, 0], X[i, 1]
            label = 'orange' if y[i]==-1 else 'apple'
            writer.writerow([weight, size, label])

# 6. Vẽ hình ảnh và thống kê (cần sửa để vẽ đường phân chia)
def plot_data_with_margin(X, y, w, b, filename='data_visualization.png'):
    """Vẽ dữ liệu đã chia lớp và đường phân chia."""
    plt.figure(figsize=(8, 6))

    # Tách dữ liệu theo nhãn
    apple_indices = np.where(y == 1)
    orange_indices = np.where(y == -1)

    # Vẽ scatter plot cho từng lớp
    plt.scatter(X[apple_indices, 0], X[apple_indices, 1], marker='o', color='red', label='Apple')
    plt.scatter(X[orange_indices, 0], X[orange_indices, 1], marker='x', color='orange', label='Orange')

    # Vẽ đường phân chia (cần tìm 2 điểm để vẽ đường thẳng)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1  # Margin cho x
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1  # Margin cho y

    # Tìm hai điểm trên đường thẳng decision boundary
    x_plot = np.linspace(x_min, x_max, 100)
    y_plot = (b - w[0] * x_plot) / w[1] # w[0]*x + w[1]*y = b  => y = (b - w[0]*x) / w[1]

    # Lọc các giá trị y_plot nằm trong khoảng y_min và y_max
    y_plot = np.clip(y_plot, y_min, y_max)

    plt.plot(x_plot, y_plot, color='blue', linestyle='-', label='Decision Boundary')

    # Thêm label và title
    plt.xlabel('Weight')
    plt.ylabel('Size')
    plt.title('Phân loại Apple và Orange với Decision Boundary')
    plt.legend()

    # Hiển thị số lượng mẫu mỗi lớp
    num_apples = np.sum(y == 1)
    num_oranges = np.sum(y == -1)
    plt.text(0.05, 0.95, f'Số lượng Apple: {num_apples}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.05, 0.90, f'Số lượng Orange: {num_oranges}', transform=plt.gca().transAxes, verticalalignment='top')

    # Giới hạn trục x và y
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Lưu hình ảnh
    plt.savefig(filename)
    plt.show()

# Main
if __name__ == "__main__":
    # 1. Tải dữ liệu huấn luyện
    X_train, y_train = load_training_data('veriseti.csv')

    # 2. Tính w và b
    w, b = calculate_w_and_b(X_train, y_train)

    if w is not None and b is not None:
        # 3. Tải dữ liệu kiểm tra
        X_test = load_data_no_labels('veriseti_no_labels.csv')

        # 4. Gán nhãn cho dữ liệu kiểm tra
        y_pred = assign_labels(X_test, w, b)

        # 5. Lưu dữ liệu đã gán nhãn vào file
        save_labeled_data(X_test, y_pred)

        # 6. Vẽ hình ảnh và thống kê
        plot_data_with_margin(X_test, y_pred, w, b)
    else:
        print("Không thể tính w và b. Kiểm tra dữ liệu huấn luyện.")