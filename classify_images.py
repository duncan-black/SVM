import os
import cv2
import numpy as np
import pickle

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

def predict(x, w):
    """Dự đoán nhãn cho một mẫu sử dụng mô hình SVM đã huấn luyện."""
    return 1 if np.dot(w, x) >= 0 else -1

def classify_and_save_images(input_dir, output_dir, model_file='svm_model.pkl'):
    """Phân loại ảnh trong thư mục đầu vào và lưu vào thư mục đầu ra tương ứng."""

    # Tải model
    try:
        with open(model_file, 'rb') as f:
            w = pickle.load(f)
        print("Đã tải mô hình SVM từ svm_model.pkl")
    except FileNotFoundError:
        print(f"Không tìm thấy file mô hình: {model_file}")
        return
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cat_output_dir = os.path.join(output_dir, 'cats')
    dog_output_dir = os.path.join(output_dir, 'dogs')
    os.makedirs(cat_output_dir, exist_ok=True)  # Tạo thư mục "cats"
    os.makedirs(dog_output_dir, exist_ok=True)  # Tạo thư mục "dogs"

    # Duyệt qua các ảnh trong thư mục đầu vào
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            try:
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Không thể đọc ảnh: {filename}, bỏ qua.")
                    continue

                hog_features = extract_hog_features(img, filename)
                if hog_features is not None:
                    prediction = predict(hog_features, w)
                    if prediction == 1:
                        output_path = os.path.join(cat_output_dir, filename)
                        print(f"Ảnh {filename} được phân loại là mèo.")
                    else:
                        output_path = os.path.join(dog_output_dir, filename)
                        print(f"Ảnh {filename} được phân loại là chó.")
                    cv2.imwrite(output_path, img)
                    print(f"Đã lưu ảnh {filename} vào {output_path}")
                else:
                    print(f"Không thể trích xuất đặc trưng HOG từ {filename}, bỏ qua.")
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")

    print("Hoàn thành phân loại và lưu ảnh.")

if __name__ == "__main__":
    input_dir = './test_images' # Thay đổi đường dẫn
    output_dir = './classified_images' # Thay đổi đường dẫn
    classify_and_save_images(input_dir, output_dir)