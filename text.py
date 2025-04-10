import os

import skimage
import skimage.io as io
from skimage.feature import hog
from skimage import exposure
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt #Để vẽ

def process_image(img, filename):
    """
    Tính toán HOG cho một ảnh, và trả về cả hình ảnh HOG, độ lớn gradient, và hướng gradient.
    """
    try:
        # Chuyển đổi ảnh sang thang độ xám
        gray = skimage.color.rgb2gray(img) if len(img.shape) == 3 else img
        print(len(img.shape))

        # Tính toán độ lớn và hướng gradient sử dụng Sobel filters
        # gx = filters.sobel_h(gray) #Gradient theo chiều ngang
        # gy = filters.sobel_v(gray) #Gradient theo chiều dọc

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        for y in range(1, gray.shape[0] - 1):  # Bỏ qua hàng đầu và hàng cuối
            for x in range(1, gray.shape[1] - 1):  # Bỏ qua cột đầu và cột cuối
                gx[y, x] = gray[y, x + 1] - gray[y, x - 1]
                gy[y, x] = gray[y + 1, x] - gray[y - 1, x]
        magnitude = np.sqrt(gx ** 2 + gy ** 2)  # Độ lớn
        orientation = np.arctan2(gy, gx)  # Hướng (radian)
        print(orientation)

        # Tính toán HOG (chỉ để hiển thị hình ảnh HOG)
        fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)

        # Chỉnh độ tương phản cho hình ảnh HOG để hiển thị tốt hơn
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return hog_image_rescaled, magnitude, orientation

    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")
        return None, None, None

def process_images_in_folder(input_dir, output_dir):
    """
    Đọc, xử lý (HOG), và lưu ảnh trong một thư mục.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_filename = f'hog_{filename}.png'
            output_path = os.path.join(output_dir, output_filename)

            try:
                img = io.imread(input_path)
                hog_image, magnitude, orientation = process_image(img, filename) #Gọi process_image

                if hog_image is not None:
                    #In một vài thông tin để xem
                    print(f"Độ lớn gradient (hình dạng): {magnitude.shape}")
                    print(f"Hướng gradient (hình dạng): {orientation.shape}")

                    #Bạn có thể vẽ độ lớn và hướng gradient (tùy chọn)
                    #Để vẽ, cần chia tỷ lệ giá trị để hiển thị
                    plt.figure(figsize=(12,6))

                    plt.subplot(1,3,1)
                    plt.imshow(img, cmap='gray')
                    plt.title('Ảnh gốc')

                    plt.subplot(1,3,2)
                    plt.imshow(magnitude, cmap='jet')  # 'jet' hoặc 'viridis'
                    plt.title('Độ lớn Gradient')

                    plt.subplot(1,3,3)
                    plt.imshow(orientation, cmap='hsv')  # 'hsv' phù hợp để hiển thị góc
                    plt.title('Hướng Gradient')

                    plt.show()


                    io.imsave(output_path, hog_image)
                    print(f"Đã xử lý và lưu: {output_path}")
                else:
                    print(f"Không thể xử lý {filename}, bỏ qua.")

            except Exception as e:
                print(f"Lỗi khi đọc/lưu {filename}: {e}")

    print("Hoàn thành.")

# Cài đặt đường dẫn
input_dir = './anh_hoa'  # Thay đổi đường dẫn
output_dir ='./anh_hoa_xuat'  # Thay đổi đường dẫn
process_images_in_folder(input_dir, output_dir)