import os
import cv2
import numpy as np

def extract_hog_features(img, filename):
    """
    Tính toán HOG descriptors cho một ảnh và trả về vector đặc trưng.
    """
    try:
        # Khởi tạo HOGDescriptor (giữ nguyên các tham số)
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        # Resize ảnh (bắt buộc)
        img = cv2.resize(img, winSize)

        # Tính toán HOG descriptors
        hogs = hog.compute(img)

        return hogs.flatten()  # Trả về vector đặc trưng đã làm phẳng

    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")
        return None

def process_images_in_folder(input_dir):
    """
    Đọc các ảnh từ một thư mục và trích xuất vector đặc trưng HOG cho mỗi ảnh.

    Args:
        input_dir (str): Đường dẫn đến thư mục chứa ảnh đầu vào.

    Returns:
        tuple: Một tuple chứa hai mảng NumPy:
            - features: Một mảng 2D trong đó mỗi hàng là vector đặc trưng HOG cho một ảnh.
            - filenames: Một mảng 1D chứa tên của các tệp ảnh tương ứng.
        Trả về (None, None) nếu không có ảnh nào được xử lý thành công.
    """

    feature_list = []  # Danh sách để lưu trữ vector đặc trưng
    filename_list = [] # Danh sách để lưu trữ tên tệp

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
                    feature_list.append(hog_features)
                    filename_list.append(filename)
                    print(f"Đã trích xuất đặc trưng HOG từ: {filename}")
                else:
                    print(f"Không thể trích xuất đặc trưng HOG từ {filename}, bỏ qua.")

            except Exception as e:
                print(f"Lỗi khi đọc/xử lý {filename}: {e}")

    if feature_list:
        features = np.array(feature_list)  # Chuyển đổi danh sách thành mảng NumPy
        filenames = np.array(filename_list)
        return features, filenames
    else:
        print("Không có đặc trưng HOG nào được trích xuất.")
        return None, None

# Cài đặt đường dẫn
input_dir = './anh_hoa'  # Thay đổi đường dẫn
features, filenames = process_images_in_folder(input_dir) #Không cần thư mục đầu ra nữa

# In ra vector đặc trưng nếu có
if features is not None:
    print("Mảng vector đặc trưng HOG:")
    print(features)

    print("\nTên các tệp ảnh tương ứng:")
    print(filenames)

    #Bạn có thể làm gì đó với 'features' và 'filenames' ở đây,
    #ví dụ như huấn luyện mô hình SVM

else:
    print("Không có đặc trưng HOG nào được trích xuất.")