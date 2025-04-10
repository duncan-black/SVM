import csv


def read_csv_file(filename):
    """Đọc file CSV và trả về một danh sách các dòng."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Đọc dòng header (nếu có)
        print(f"Header: {header}")

        for row in reader:
            data.append(row)

    return data
if __name__ == "__main__":
    filename = "veriseti.csv"  # Thay bằng tên file CSV của bạn
    data = read_csv_file(filename)

    if data:
        print("\nDữ liệu đã đọc:")
        for row in data:
            print(row)

        # --- Ví dụ xử lý dữ liệu (chuyển đổi kiểu dữ liệu) ---

            # Lấy cột Weight và Size, chuyển sang float
            weights = [float(row[0]) for row in data]
            sizes = [float(row[1]) for row in data]
            classes = [row[2] for row in data]

            print("Weights:", weights)
            print("Sizes:", sizes)
            print("Classes:", classes)