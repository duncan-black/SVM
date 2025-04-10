import csv
import numpy as np

# 1. Đọc dữ liệu từ file CSV
def load_data(filename):
    """Đọc dữ liệu từ file CSV và trả về X (đặc trưng) và y (nhãn)."""
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Bỏ qua header
        for row in reader:
            try:
                weight = float(row[0])
                size = float(row[1])
                label = row[2]

                X.append([weight, size])  # Đặc trưng: Weight và Size
                if label == 'orange':
                    y.append(-1)  # Gán nhãn -1 cho orange
                elif label == 'apple':
                    y.append(1)   # Gán nhãn +1 cho apple
                else:
                    print(f"Nhãn không hợp lệ: {label}")
                    continue  # Bỏ qua dòng này
            except ValueError as e:
                print(f"Lỗi chuyển đổi kiểu dữ liệu: {e}, dòng: {row}")
                continue  # Bỏ qua dòng này
    return np.array(X), np.array(y)

# 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Chia dữ liệu thành tập huấn luyện và tập kiểm tra."""
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(test_size * X.shape[0])
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

# 3. Chuẩn hóa dữ liệu (tùy chọn, nhưng nên làm)
def standardize(X):
    """Chuẩn hóa dữ liệu về mean=0 và std=1."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# 4. Lớp SVM (Linear SVM)
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weights
        self.b = None  # Bias

    def fit(self, X, y):
        """Huấn luyện mô hình SVM."""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # Khởi tạo weights
        self.b = 0  # Khởi tạo bias

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    # Cập nhật weights nếu điểm nằm ngoài margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Cập nhật weights và bias nếu điểm nằm trong hoặc trên margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        """Dự đoán nhãn cho dữ liệu X."""
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)  # Trả về +1 hoặc -1

# 5. Đánh giá mô hình
def accuracy(y_true, y_pred):
    """Tính độ chính xác."""
    return np.sum(y_true == y_pred) / len(y_true)

# 6. Hàm main
if __name__ == "__main__":
    # 1. Tải dữ liệu
    X, y = load_data('veriseti.csv')

    # 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Chuẩn hóa dữ liệu
    X_train = standardize(X_train)
    X_test = standardize(X_test)

    # 4. Huấn luyện mô hình SVM
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    # 5. Dự đoán trên tập kiểm tra
    y_pred = svm.predict(X_test)

    # 6. Đánh giá mô hình
    acc = accuracy(y_test, y_pred)
    print(f"Độ chính xác: {acc}")