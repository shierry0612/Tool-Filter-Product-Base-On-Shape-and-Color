import sys
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QFileDialog, QScrollArea, QGridLayout, QCheckBox,
    QMessageBox, QGroupBox, QLineEdit, QSpinBox, QComboBox, QProgressDialog,
    QDialog
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


# =========================
# GLOBAL CONFIG & MODEL
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_BINS = 16  # bins cho Hue

# Preprocess cho ResNet
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


def create_feature_extractor():
    weights = ResNet50_Weights.IMAGENET1K_V1
    base = resnet50(weights=weights)
    model = nn.Sequential(*list(base.children())[:-1])  # bỏ FC
    model.eval()
    model.to(DEVICE)
    return model


FEATURE_EXTRACTOR = create_feature_extractor()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def deep_feature_roi(img: Image.Image) -> torch.Tensor:
    """
    Trích deep feature (2048-d) từ ảnh (center crop), L2-normalize.
    """
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    feat = FEATURE_EXTRACTOR(x).view(-1)
    feat = feat / (feat.norm(p=2) + 1e-8)
    return feat.cpu()


def hue_hist_center(img: Image.Image, num_bins: int = NUM_BINS) -> np.ndarray:
    """
    Histogram Hue trên vùng center crop (224x224).
    """
    img_resized = img.resize((256, 256))
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    img_crop = img_resized.crop((left, top, left + 224, top + 224))

    hsv = img_crop.convert("HSV")
    h, s, v = hsv.split()
    h_np = np.array(h).astype(np.float32)
    hist, _ = np.histogram(h_np, bins=num_bins, range=(0, 255))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# =========================
# INDEX
# =========================

class ImageIndex:
    """
    Lưu index cho gallery:
    - paths         : list[Path]
    - deep_matrix   : torch.Tensor (N, 2048)  (L2-norm theo hàng)
    - color_hists   : np.ndarray (N, NUM_BINS) (L2-norm theo hàng)
    """
    def __init__(self):
        self.paths: list[Path] = []
        self.deep_matrix: torch.Tensor | None = None
        self.color_hists: np.ndarray | None = None

    def is_built(self) -> bool:
        return (
            self.deep_matrix is not None
            and self.color_hists is not None
            and len(self.paths) > 0
        )

    def build_from_folders(self, folders: list[Path], progress_callback=None):
        files = []
        for folder in folders:
            for root, _, names in os.walk(folder):
                for name in names:
                    if Path(name).suffix.lower() in IMG_EXT:
                        files.append(Path(root) / name)

        files = sorted(files)
        if not files:
            raise RuntimeError("Không tìm thấy ảnh hợp lệ trong các thư mục.")

        deep_feats = []
        color_hists = []

        total = len(files)
        print(f"Indexing {total} images...")

        for i, p in enumerate(files):
            img = load_image(p)

            f_deep = deep_feature_roi(img)
            deep_feats.append(f_deep.unsqueeze(0))

            hist = hue_hist_center(img, num_bins=NUM_BINS)
            color_hists.append(hist[None, :])

            if progress_callback is not None:
                progress_callback(i + 1, total)

            if (i + 1) % 20 == 0 or i == len(files) - 1:
                print(f"  processed {i+1}/{total}")

        deep_matrix = torch.cat(deep_feats, dim=0)
        deep_matrix = deep_matrix / (deep_matrix.norm(dim=1, keepdim=True) + 1e-8)

        color_hists_np = np.concatenate(color_hists, axis=0).astype(np.float32)
        norms = np.linalg.norm(color_hists_np, axis=1, keepdims=True) + 1e-8
        color_hists_np = color_hists_np / norms

        self.paths = files
        self.deep_matrix = deep_matrix
        self.color_hists = color_hists_np

        print("Index built:", deep_matrix.shape, color_hists_np.shape)

    def build_from_folder(self, folder: Path, progress_callback=None):
        self.build_from_folders([folder], progress_callback=progress_callback)


# =========================
# SEARCH
# =========================

@torch.no_grad()
def search_shape_only(q_img: Image.Image, index: ImageIndex):
    deep_matrix = index.deep_matrix
    paths = index.paths

    q_feat = deep_feature_roi(q_img).unsqueeze(1)
    shape_sims = (deep_matrix @ q_feat).squeeze(1)

    q_hist = hue_hist_center(q_img, num_bins=NUM_BINS).astype(np.float32)
    if np.linalg.norm(q_hist) > 0:
        q_hist = q_hist / (np.linalg.norm(q_hist) + 1e-8)
    color_sims = index.color_hists @ q_hist

    results = []
    for i in range(shape_sims.shape[0]):
        shape_sim = float(shape_sims[i])
        color_sim = float(color_sims[i])
        total = shape_sim
        results.append((paths[i], shape_sim, color_sim, shape_sim, total))

    results.sort(key=lambda x: x[-1], reverse=True)
    return results


@torch.no_grad()
def search_shape_plus_color(
    q_img: Image.Image,
    index: ImageIndex,
    topM: int | None = None,
    color_threshold: float = 0.75,
    alpha: float = 0.7,
):
    deep_matrix = index.deep_matrix
    paths = index.paths

    q_feat = deep_feature_roi(q_img).unsqueeze(1)
    shape_sims = (deep_matrix @ q_feat).squeeze(1)

    q_hist = hue_hist_center(q_img, num_bins=NUM_BINS).astype(np.float32)
    if np.linalg.norm(q_hist) > 0:
        q_hist = q_hist / (np.linalg.norm(q_hist) + 1e-8)
    color_sims_all = index.color_hists @ q_hist

    N = shape_sims.shape[0]
    if topM is None:
        M = N
        idxs = torch.arange(N)
        shape_top = shape_sims
    else:
        M = min(topM, N)
        shape_top, idxs = torch.topk(shape_sims, k=M)

    candidates = []
    for rank in range(M):
        i = int(idxs[rank])
        shape_sim = float(shape_top[rank])
        color_sim = float(color_sims_all[i])
        if color_sim < color_threshold:
            continue
        total = alpha * shape_sim + (1.0 - alpha) * color_sim
        candidates.append((paths[i], shape_sim, color_sim, shape_sim, total))

    candidates.sort(key=lambda x: x[-1], reverse=True)
    return candidates


# =========================
# IMAGE VIEWER DIALOG (auto-fit)
# =========================

class ImageViewerDialog(QDialog):
    """
    Dialog xem 1 ảnh lớn:
    - Mặc định: auto-fit toàn bộ ảnh vào cửa sổ.
    - Nút + / - để zoom tay -> tắt auto-fit.
    - Nút 100% để bật lại auto-fit.
    """
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(image_path).name)
        self.scale_factor = 1.0
        self.auto_fit = True

        layout = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.label)
        layout.addWidget(self.scroll)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_in = QPushButton("+")
        self.btn_reset = QPushButton("100%")
        btn_row.addWidget(self.btn_zoom_out)
        btn_row.addWidget(self.btn_zoom_in)
        btn_row.addWidget(self.btn_reset)
        layout.addLayout(btn_row)

        self.pix_original = QPixmap(image_path)

        self.resize(1000, 700)
        self.update_pixmap()

        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_reset.clicked.connect(self.reset_zoom)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.update_pixmap()

    def update_pixmap(self):
        if self.pix_original.isNull():
            return

        viewport = self.scroll.viewport().size()
        if self.auto_fit and viewport.width() > 0 and viewport.height() > 0:
            w_ratio = viewport.width() / self.pix_original.width()
            h_ratio = viewport.height() / self.pix_original.height()
            self.scale_factor = max(0.1, min(w_ratio, h_ratio))

        w = int(self.pix_original.width() * self.scale_factor)
        h = int(self.pix_original.height() * self.scale_factor)
        if w <= 0 or h <= 0:
            return
        scaled = self.pix_original.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)

    def zoom_in(self):
        self.auto_fit = False
        self.scale_factor = min(self.scale_factor * 1.25, 10.0)
        self.update_pixmap()

    def zoom_out(self):
        self.auto_fit = False
        self.scale_factor = max(self.scale_factor / 1.25, 0.1)
        self.update_pixmap()

    def reset_zoom(self):
        self.auto_fit = True
        self.update_pixmap()


# =========================
# HISTORY DIALOG
# =========================

class HistoryDialog(QDialog):
    def __init__(self, history_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nhật ký đã xem (chưa lưu)")
        self.history_paths = list(history_paths)
        self.saved_paths = []

        layout = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.scroll.setWidget(self.inner)
        layout.addWidget(self.scroll)

        cols = 4
        thumb = 200
        for i, p in enumerate(self.history_paths):
            row = i // cols
            col = i % cols

            cell = QWidget()
            v = QVBoxLayout(cell)

            chk = QCheckBox(Path(p).name)
            chk.setObjectName(f"hist_chk_{i}")
            v.addWidget(chk)

            lbl_img = QLabel()
            lbl_img.setFixedSize(thumb, thumb)
            lbl_img.setStyleSheet("border: 1px solid gray;")
            pix = QPixmap(str(p))
            if not pix.isNull():
                pix = pix.scaled(lbl_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_img.setPixmap(pix)
            v.addWidget(lbl_img)

            btn_view = QPushButton("View")
            btn_view.clicked.connect(lambda checked, path=str(p): self.on_view_image(path))
            v.addWidget(btn_view)

            self.grid.addWidget(cell, row, col)

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save selected images...")
        self.btn_save.clicked.connect(self.on_save_selected)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.resize(1000, 700)

    def on_view_image(self, path: str):
        dlg = ImageViewerDialog(path, self)
        dlg.exec()

    def on_save_selected(self):
        indices = []
        for i, _ in enumerate(self.history_paths):
            chk = self.inner.findChild(QCheckBox, f"hist_chk_{i}")
            if chk and chk.isChecked():
                indices.append(i)

        if not indices:
            QMessageBox.information(self, "Info", "Chưa chọn hình nào trong history.")
            return

        dest_dir_str = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục để lưu hình đã chọn từ history"
        )
        if not dest_dir_str:
            return
        dest_dir = Path(dest_dir_str)

        count = 0
        for idx in indices:
            p = self.history_paths[idx]
            shutil.copy2(p, dest_dir / Path(p).name)
            if p not in self.saved_paths:
                self.saved_paths.append(p)
            count += 1

        QMessageBox.information(self, "Done", f"Đã lưu {count} hình từ history.")


# =========================
# MAIN WINDOW
# =========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Product Shape & Color Search Demo")
        self.resize(1200, 800)

        self.gallery_dirs: list[Path] = []
        self.query_path: Path | None = None
        self.index = ImageIndex()

        self.all_results = []
        self.current_results = []
        self.page_size = 100

        # thumbnail to hơn mặc định
        self.thumb_size = 220
        self.last_clicked_index: int | None = None

        self.saved_paths: set[Path] = set()
        self.history_paths: set[Path] = set()

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # ----- Phase 1 -----
        phase1 = QGroupBox("Phase 1: Import")
        p1 = QVBoxLayout(phase1)

        row_g = QHBoxLayout()
        self.btn_gallery = QPushButton("Thêm thư mục hình (gallery)...")
        self.btn_gallery.clicked.connect(self.on_add_gallery_folder)
        self.txt_gallery = QLineEdit()
        self.txt_gallery.setReadOnly(True)
        self.btn_build = QPushButton("Build Index")
        self.btn_build.clicked.connect(self.on_build_index)
        row_g.addWidget(self.btn_gallery)
        row_g.addWidget(self.txt_gallery)
        row_g.addWidget(self.btn_build)
        p1.addLayout(row_g)

        row_q = QHBoxLayout()
        self.btn_query = QPushButton("Chọn hình mẫu (input)...")
        self.btn_query.clicked.connect(self.on_choose_query)
        self.lbl_query_path = QLabel("Chưa chọn")
        self.lbl_query_preview = QLabel()
        self.lbl_query_preview.setFixedSize(150, 150)
        self.lbl_query_preview.setStyleSheet("border: 1px solid gray;")
        row_q.addWidget(self.btn_query)
        row_q.addWidget(self.lbl_query_path)
        row_q.addWidget(self.lbl_query_preview)
        p1.addLayout(row_q)

        main_layout.addWidget(phase1)

        # ----- Phase 2 -----
        phase2 = QGroupBox("Phase 2: Chọn cách lọc  điều khiển")
        p2 = QHBoxLayout(phase2)

        p2.addWidget(QLabel("Filter mode:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Shape only", "Shape + color"])
        p2.addWidget(self.cmb_mode)

        p2.addWidget(QLabel("Top K:"))
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 1000)
        self.spin_topk.setValue(100)
        p2.addWidget(self.spin_topk)

        self.btn_run = QPushButton("Run Search")
        self.btn_run.clicked.connect(self.on_run_search)
        p2.addWidget(self.btn_run)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.on_refresh)
        p2.addWidget(self.btn_refresh)

        self.lbl_status = QLabel("Ready")
        p2.addWidget(self.lbl_status)

        p2.addStretch()
        main_layout.addWidget(phase2)

        # ----- Phase 3 -----
        phase3 = QGroupBox("Phase 3: Kết quả, zoom  chọn hình")
        p3 = QVBoxLayout(phase3)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom thumbnail:"))
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.clicked.connect(self.on_zoom_out)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.clicked.connect(self.on_zoom_in)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addStretch()
        p3.addLayout(zoom_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.scroll.setWidget(self.results_widget)
        p3.addWidget(self.scroll)

        row_save = QHBoxLayout()
        self.lbl_selected_info = QLabel("Selected (page): 0/0 | Saved total: 0 | Seen (not saved): 0")
        self.btn_history = QPushButton("View history")
        self.btn_history.clicked.connect(self.on_view_history)
        self.btn_save = QPushButton("Save selected images...")
        self.btn_save.clicked.connect(self.on_save_selected)
        row_save.addWidget(self.lbl_selected_info)
        row_save.addStretch()
        row_save.addWidget(self.btn_history)
        row_save.addWidget(self.btn_save)
        p3.addLayout(row_save)

        main_layout.addWidget(phase3)

        self.setCentralWidget(central)

    # ----- events -----

    def on_add_gallery_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục hình (gallery)")
        if folder:
            p = Path(folder)
            if p not in self.gallery_dirs:
                self.gallery_dirs.append(p)
            self.txt_gallery.setText("; ".join(str(d) for d in self.gallery_dirs))

            self.index = ImageIndex()
            self.all_results = []
            self.current_results = []
            self.saved_paths.clear()
            self.history_paths.clear()
            self.clear_results()
            self.lbl_status.setText("Đã thêm thư mục, cần Build Index.")
            self.update_selected_label()

    def on_build_index(self):
        if not self.gallery_dirs:
            QMessageBox.warning(self, "Warning", "Chưa chọn thư mục gallery nào.")
            return
        try:
            progress = QProgressDialog("Loading images...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Building index...")
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            def progress_cb(done, total):
                progress.setMaximum(total)
                progress.setValue(done)
                progress.setLabelText(f"Loading... {done}/{total}")
                QApplication.processEvents()
                if progress.wasCanceled():
                    raise RuntimeError("User cancelled")

            self.index.build_from_folders(self.gallery_dirs, progress_callback=progress_cb)
            progress.close()
            QMessageBox.information(self, "OK", "Build index xong.")
            self.lbl_status.setText(f"Index built: {len(self.index.paths)} images")
        except RuntimeError:
            QMessageBox.information(self, "Cancelled", "Đã hủy build index.")
            self.lbl_status.setText("Index build cancelled")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Lỗi build index:\n{e}")
            self.lbl_status.setText("Index build error")

    def on_choose_query(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Chọn hình mẫu (input)", filter="Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if file:
            self.query_path = Path(file)
            self.lbl_query_path.setText(str(self.query_path))
            pix = QPixmap(str(self.query_path))
            if not pix.isNull():
                pix = pix.scaled(self.lbl_query_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_query_preview.setPixmap(pix)

            self.all_results = []
            self.current_results = []
            self.saved_paths.clear()
            self.history_paths.clear()
            self.clear_results()
            self.lbl_status.setText("Ready for search")
            self.update_selected_label()

    def _compute_all_results(self):
        if not self.index.is_built() or not self.query_path:
            return []

        img_q = load_image(self.query_path)
        mode = self.cmb_mode.currentText()

        if mode == "Shape only":
            return search_shape_only(img_q, self.index)
        else:
            return search_shape_plus_color(
                img_q, self.index,
                topM=None,
                color_threshold=0.75,
                alpha=0.7,
            )

    def _update_main_results(self):
        if not self.all_results:
            self.current_results = []
            self.clear_results()
            self.lbl_status.setText("No results")
            self.update_selected_label()
            return

        excluded = self.saved_paths.union(self.history_paths)
        remaining = [r for r in self.all_results if r[0] not in excluded]

        self.current_results = remaining[:self.page_size]

        if not self.current_results:
            self.clear_results()
            self.lbl_status.setText("Không còn ảnh mới. Xem thêm trong history nếu cần.")
            self.update_selected_label()
            return

        self.show_results()
        self.lbl_status.setText(f"Remaining: {len(remaining)} results")

    def on_run_search(self):
        if not self.index.is_built():
            QMessageBox.warning(self, "Warning", "Index chưa build.")
            return
        if not self.query_path:
            QMessageBox.warning(self, "Warning", "Chưa chọn hình mẫu.")
            return

        try:
            self.page_size = self.spin_topk.value()

            self.saved_paths.clear()
            self.history_paths.clear()
            self.lbl_status.setText("Waiting...")
            QApplication.processEvents()

            self.all_results = self._compute_all_results()

            if not self.all_results:
                QMessageBox.information(self, "Info", "Không có kết quả phù hợp.")
                self.clear_results()
                self.lbl_status.setText("No results")
                self.update_selected_label()
                return

            self._update_main_results()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Lỗi khi search:\n{e}")
            self.lbl_status.setText("Search error")

    def on_refresh(self):
        if not self.all_results:
            return

        for i, (path, *_rest) in enumerate(self.current_results):
            if path in self.saved_paths:
                continue
            chk = self.results_widget.findChild(QCheckBox, f"chk_{i}")
            if chk and not chk.isChecked():
                self.history_paths.add(path)

        self.page_size = self.spin_topk.value()
        self._update_main_results()

    def on_view_history(self):
        if not self.history_paths:
            QMessageBox.information(self, "Info", "Chưa có ảnh nào trong nhật ký đã xem.")
            return

        dlg = HistoryDialog(sorted(self.history_paths, key=lambda p: str(p)), self)
        dlg.exec()

        if dlg.saved_paths:
            for p in dlg.saved_paths:
                self.saved_paths.add(p)
                if p in self.history_paths:
                    self.history_paths.remove(p)
        self.update_selected_label()

    def on_save_selected(self):
        if not self.current_results:
            QMessageBox.information(self, "Info", "Chưa có kết quả để lưu.")
            return

        dest_dir_str = QFileDialog.getExistingDirectory(self, "Chọn thư mục để lưu hình đã chọn")
        if not dest_dir_str:
            return
        dest_dir = Path(dest_dir_str)

        count = 0
        for i, (path, *_rest) in enumerate(self.current_results):
            chk = self.results_widget.findChild(QCheckBox, f"chk_{i}")
            if chk and chk.isChecked():
                shutil.copy2(path, dest_dir / path.name)
                self.saved_paths.add(path)
                count += 1

        QMessageBox.information(self, "Done", f"Đã lưu {count} hình vào:\n{dest_dir}")
        self.update_selected_label()

    # ----- zoom & key -----

    def on_zoom_in(self):
        self.thumb_size = min(self.thumb_size + 40, 500)
        if self.current_results:
            self.show_results()

    def on_zoom_out(self):
        self.thumb_size = max(self.thumb_size - 40, 100)
        if self.current_results:
            self.show_results()

    def make_image_click_handler(self, idx: int):
        def handler(event):
            chk = self.results_widget.findChild(QCheckBox, f"chk_{idx}")
            if chk:
                chk.setChecked(not chk.isChecked())
                self.last_clicked_index = idx
                self.update_selected_label()
        return handler

    def on_view_image_main(self, image_path: str):
        dlg = ImageViewerDialog(image_path, self)
        dlg.exec()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.current_results and self.last_clicked_index is not None:
                chk = self.results_widget.findChild(QCheckBox, f"chk_{self.last_clicked_index}")
                if chk:
                    chk.setChecked(not chk.isChecked())
                    self.update_selected_label()
        else:
            super().keyPressEvent(event)

    # ----- helpers -----

    def clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.last_clicked_index = None

    def update_selected_label(self):
        selected_current = 0
        for i, _ in enumerate(self.current_results):
            chk = self.results_widget.findChild(QCheckBox, f"chk_{i}")
            if chk and chk.isChecked():
                selected_current += 1

        saved_total = len(self.saved_paths)
        seen_total = len(self.history_paths)

        self.lbl_selected_info.setText(
            f"Selected (page): {selected_current}/{len(self.current_results)} | "
            f"Saved total: {saved_total} | Seen (not saved): {seen_total}"
        )

    def show_results(self):
        self.clear_results()
        thumb = self.thumb_size
        cols = 4
        for i, (path, shape_sim, color_sim, global_sim, total_score) in enumerate(self.current_results):
            row = i // cols
            col = i % cols

            cell = QWidget()
            v = QVBoxLayout(cell)

            chk = QCheckBox(path.name)
            chk.setObjectName(f"chk_{i}")
            chk.stateChanged.connect(lambda state, idx=i: self._on_checkbox_toggled(idx))
            v.addWidget(chk)

            lbl_img = QLabel()
            lbl_img.setFixedSize(thumb, thumb)
            lbl_img.setStyleSheet("border: 1px solid gray;")
            pix = QPixmap(str(path))
            if not pix.isNull():
                pix = pix.scaled(lbl_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_img.setPixmap(pix)
            lbl_img.mousePressEvent = self.make_image_click_handler(i)
            v.addWidget(lbl_img)

            btn_view = QPushButton("View")
            btn_view.clicked.connect(lambda checked, p=str(path): self.on_view_image_main(p))
            v.addWidget(btn_view)

            self.results_layout.addWidget(cell, row, col)

        self.update_selected_label()

    def _on_checkbox_toggled(self, idx: int):
        self.last_clicked_index = idx
        self.update_selected_label()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    print("Starting tool (history mode)...")
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e))
        input("Press Enter to exit...")
