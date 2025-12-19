# Product Shape & Color Search Demo

Ứng dụng desktop (GUI) giúp **lọc/tìm ảnh sản phẩm tương tự** theo:
- **Hình dạng (shape)**: trích đặc trưng deep feature bằng **ResNet50 (ImageNet weights)** và so sánh cosine similarity.
- **Màu sắc (color)**: histogram **Hue** trên vùng center-crop và so sánh similarity. 

Tool phù hợp cho workflow “chọn 1 ảnh mẫu → tìm các ảnh giống nhất trong gallery → chọn & lưu kết quả”, kèm **history** (đã xem nhưng chưa lưu) để tránh lặp ảnh. 

---

## Tính năng chính

- **Build Index** từ nhiều thư mục gallery (quét đệ quy theo đuôi ảnh `.jpg/.jpeg/.png/.bmp/.webp`).
- 2 chế độ lọc:
  - **Shape only**: chỉ dựa trên similarity deep feature. 
  - **Shape + color**: kết hợp shape + màu (lọc theo `color_threshold`, chấm điểm tổng theo `alpha`). 
- GUI theo 3 phase:
  1) Import gallery + chọn ảnh mẫu + build index  
  2) Chọn mode + Top K + Run/Refresh  
  3) Xem kết quả dạng grid thumbnail, zoom, chọn ảnh, lưu ảnh, xem history 
- **Image Viewer**: xem ảnh lớn auto-fit, zoom +/- và reset 100%. 

---

## Cách hoạt động (tóm tắt)

1. **Indexing**
   - Với mỗi ảnh trong gallery:  
     - Trích deep feature (2048-d) từ ResNet50 (bỏ FC), L2-normalize.
     - Tính Hue histogram (NUM_BINS bins) trên center crop.  
   - Lưu `deep_matrix (N×2048)` và `color_hists (N×NUM_BINS)` đã normalize.  

2. **Search**
   - `Shape only`: score = `shape_sim` 
   - `Shape + color`: lọc ứng viên theo `color_sim >= color_threshold`, score tổng:
     ```
     total = alpha * shape_sim + (1 - alpha) * color_sim
     ```
 

---

## Yêu cầu hệ thống

- Python 3.10+ (khuyến nghị)
- Thư viện chính: `torch`, `torchvision`, `numpy`, `Pillow`, `PySide6` 
- Có thể chạy CPU hoặc GPU (tự nhận CUDA nếu có). 

---

## Cài đặt

 1) Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
 2) Cài dependencies

Nếu dùng CPU:

pip install numpy pillow pyside6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

**## Hướng dẫn sử dụng (UI)**
**Phase 1: Import**

- Bấm “Thêm thư mục hình (gallery)” để chọn 1 hoặc nhiều folder ảnh. 

- Bấm Build Index để tạo index (có progress). 

- Bấm “Chọn hình mẫu (input)” để chọn ảnh query. 


**Phase 2: Chọn chế độ lọc**

- Filter mode: Shape only hoặc Shape + color 

- Top K: số lượng kết quả hiển thị mỗi lượt 

- Bấm Run Search để chạy. 


**Phase 3: Kết quả**

- Xem grid thumbnail, zoom +/-, click ảnh để tick chọn. 

- Save selected images… để copy ảnh đã chọn sang thư mục khác. 

- Refresh: đưa các ảnh “đã xem nhưng chưa lưu” vào history để tránh lặp khi lướt tiếp. 

- View history: xem danh sách đã xem, chọn và lưu lại. 
