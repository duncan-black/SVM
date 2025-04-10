import os
import cv2
import numpy as np
import pickle  # Dùng để lưu model

def extract_hog_features(img, filename):
    """Tính toán HOG descriptors cho một ảnh và trả về vector đặc trưng."""
    try:
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        img = cv2.resize(img, winSize)
        hogs = hog.compute(img)
        return hogs.flatten()
    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")
        return None


def load_training_data(train_dir):
    """Đọc dữ liệu từ thư mục huấn luyện và trả về features và labels."""
    X = []
    y = []
    cat_dir = os.path.join(train_dir, "cats")
    dog_dir = os.path.join(train_dir, "dogs")
    # Đọc ảnh mèo
    if os.path.exists(cat_dir):
        for filename in os.listdir(cat_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cat_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Không thể đọc ảnh: {filename}, bỏ qua.")
                        continue
                    features = extract_hog_features(img, filename)
                    if features is not None:
                        X.append(features)
                        y.append(-1)  # Mèo được gán nhãn -1
                except Exception as e:
                    print(f"Lỗi xử lý ảnh: {filename}, {e}")
    # Đọc ảnh chó
    if os.path.exists(dog_dir):
        for filename in os.listdir(dog_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dog_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Không thể đọc ảnh: {filename}, bỏ qua.")
                        continue
                    features = extract_hog_features(img, filename)
                    if features is not None:
                        X.append(features)
                        y.append(1)  # Chó được gán nhãn 1
                except Exception as e:
                    print(f"Lỗi xử lý ảnh: {filename}, {e}")
    return np.array(X), np.array(y)

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

def load_data_no_labels(input_dir):
    """Đọc dữ liệu kiểm tra từ một thư mục (chỉ ảnh)."""
    X = []
    filenames = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            try:
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Không thể đọc ảnh: {filename}, bỏ qua.")
                    continue
                features = extract_hog_features(img, filename)
                if features is not None:
                    X.append(features)
                    filenames.append(filename)  # Lưu tên tệp
                else:
                    print(f"Không thể trích xuất đặc trưng từ: {filename}, bỏ qua.")
            except Exception as e:
                print(f"Lỗi xử lý ảnh: {filename}, {e}")
    return np.array(X), filenames

def assign_labels(X, w, b):
    """Gán nhãn dựa trên w và b."""
    y = []
    for x_i in X:
        prediction = np.dot(x_i, w) - b
        label = 1 if prediction > 0 else -1
        y.append(label)
    return np.array(y)

def save_labeled_data(X, y, filenames, output_dir):
    """Lưu dữ liệu đã gán nhãn vào thư mục đầu ra."""
    cat_output_dir = os.path.join(output_dir, 'cats')
    dog_output_dir = os.path.join(output_dir, 'dogs')

    os.makedirs(cat_output_dir, exist_ok=True)  # Tạo thư mục "cats" nếu không tồn tại
    os.makedirs(dog_output_dir, exist_ok=True)  # Tạo thư mục "dogs" nếu không tồn tại

    for i in range(X.shape[0]):
        # Đọc lại ảnh gốc (vì X chỉ chứa vector đặc trưng)
        # Lấy tên tệp từ filenames
        filename = filenames[i]

        img_path = os.path.join(input_test_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh gốc: {filename}, bỏ qua.")
            continue

        label = 'orange' if y[i]==-1 else 'apple'
        if label == 'orange':
            output_path = os.path.join(cat_output_dir, filename)  # Lưu vào thư mục mèo
        else:
            output_path = os.path.join(dog_output_dir, filename)  # Lưu vào thư mục chó

        cv2.imwrite(output_path, img)  # Lưu ảnh vào thư mục
        print(f"Đã lưu ảnh {filename} vào {output_path}")


def train_data(train_dir, input_test_dir, output_dir):
  # 1. Load training data
    X_train, y_train = load_training_data(train_dir)

    # 2. Calculate w and b
    w, b = calculate_w_and_b(X_train, y_train)
    if w is not None and b is not None:

        # 3. Load data to classify
        X_test, filenames = load_data_no_labels(input_test_dir)

        # 4. Assign labels based on w and b
        y_pred = assign_labels(X_test, w, b)

        # 5. Save labeled data to file
        save_labeled_data(X_test, y_pred, filenames, output_dir)
    else:
      print("Không thể tính w và b. Kiểm tra dữ liệu huấn luyện.")

# Main
if __name__ == "__main__":
    train_dir = './training_data'  # Đường dẫn đến thư mục huấn luyện (cats và dogs)
    input_test_dir = './image' # Đường dẫn đến thư mục chứa ảnh cần phân loại
    output_dir = './classified_images' # Đường dẫn đến thư mục lưu kết quả

    train_data(train_dir, input_test_dir, output_dir)