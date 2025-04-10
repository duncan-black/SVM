import os
import skimage.io as io
from skimage.feature import hog
from skimage import exposure
import numpy as np

def process_image(img, filename):
    """
    Tính toán HOG cho một ảnh và trả về hình ảnh HOG đã điều chỉnh.
    """
    try:
        # Tính toán HOG
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)

        # Chỉnh độ tương phản cho hình ảnh HOG để hiển thị tốt hơn
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # Chuyển đổi kiểu dữ liệu nếu cần thiết
        if hog_image_rescaled.dtype == np.float64:
            hog_image_rescaled = (hog_image_rescaled * 255).astype(np.uint8)

        return hog_image_rescaled

    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")
        return None

def process_images_in_folder(input_dir, output_dir):
    """
    Đọc, xử lý (HOG), và lưu ảnh trong một thư mục.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_filename = f'hog_{filename}.png' #Đảm bảo đuôi tệp là png
            output_path = os.path.join(output_dir, output_filename)

            try:
                img = io.imread(input_path)
                hog_image = process_image(img, filename) #Gọi process_image ở đây

                if hog_image is not None:
                    io.imsave(output_path, hog_image, quality=95) #Thêm quality
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