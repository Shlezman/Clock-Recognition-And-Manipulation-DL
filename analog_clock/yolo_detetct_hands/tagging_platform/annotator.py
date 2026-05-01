"""
Clock Hand Segmentation Annotation Tool
========================================
A PyQt5-based tool for creating YOLO-Seg datasets for analog clock hand detection.

Classes:
    0 = hour hand
    1 = minute hand
    2 = second hand

YOLO-Seg format per line:
    <class_id> x1 y1 x2 y2 ... xn yn  (normalized 0-1 coordinates)
"""

import sys
import os
import json
import shutil
import random
import math
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QGroupBox, QSlider, QSplitter, QFrame, QScrollArea, QCheckBox,
    QProgressDialog, QSpinBox, QDoubleSpinBox, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QCursor,
    QPolygonF, QFont, QPainterPath
)
import numpy as np
from annotation_logic import AnnotationManager, Annotation
from yolo_export import YOLOExporter


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

CLASS_INFO = {
    0: {"name": "Hour",   "color": QColor(255,  80,  80, 180)},   # red
    1: {"name": "Minute", "color": QColor( 80, 180, 255, 180)},   # blue
    2: {"name": "Second", "color": QColor( 80, 255, 130, 180)},   # green
}

DRAW_MODE_POLYGON  = "polygon"
DRAW_MODE_FREEHAND = "freehand"


# ──────────────────────────────────────────────────────────────────────────────
# Canvas Widget
# ──────────────────────────────────────────────────────────────────────────────

class AnnotationCanvas(QWidget):
    """
    The central drawing canvas.

    Responsibilities:
    - Render the current image with zoom/pan.
    - Overlay existing annotations (filled polygons).
    - Handle mouse events for drawing polygons or freehand paths.
    - Emit signals when annotations change.
    """

    annotationChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)

        # Image state
        self._pixmap: Optional[QPixmap] = None
        self._img_w = 1
        self._img_h = 1

        # View transform
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._pan_start: Optional[QPoint] = None
        self._pan_offset_start = QPoint(0, 0)

        # Drawing state
        self._draw_mode = DRAW_MODE_POLYGON
        self._active_class = 0
        self._drawing = False
        self._poly_points: list[QPoint] = []          # image-space points
        self._freehand_points: list[QPoint] = []

        # Annotation data (set from outside)
        self._annotations: dict[int, list[tuple[float,float]]] = {}

        # Overlay opacity (0-255)
        self._opacity = 180

        # Eraser
        self._eraser_radius = 20
        self._erase_mode = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._img_w = pixmap.width()
        self._img_h = pixmap.height()
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._drawing = False
        self._poly_points = []
        self._freehand_points = []
        self._fit_to_window()
        self.update()

    def set_annotations(self, annotations: dict):
        """annotations: {class_id: [(nx,ny), ...]}"""
        self._annotations = annotations
        self.update()

    def set_active_class(self, cls: int):
        self._active_class = cls

    def set_draw_mode(self, mode: str):
        self._draw_mode = mode
        self._cancel_drawing()

    def set_opacity(self, v: int):
        self._opacity = v
        for info in CLASS_INFO.values():
            info["color"].setAlpha(v)
        self.update()

    def set_erase_mode(self, enabled: bool):
        self._erase_mode = enabled
        self._cancel_drawing()
        self.setCursor(Qt.ArrowCursor if enabled else Qt.CrossCursor)

    def cancel_current(self):
        self._cancel_drawing()

    # ── Fit / zoom helpers ────────────────────────────────────────────────────

    def _fit_to_window(self):
        if self._pixmap is None:
            return
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        scale_x = w / self._img_w
        scale_y = h / self._img_h
        self._zoom = min(scale_x, scale_y) * 0.95
        self._center_image()

    def _center_image(self):
        scaled_w = self._img_w * self._zoom
        scaled_h = self._img_h * self._zoom
        self._offset = QPoint(
            int((self.width()  - scaled_w) / 2),
            int((self.height() - scaled_h) / 2),
        )

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _widget_to_image(self, wp: QPoint) -> QPoint:
        """Convert widget pixel → image pixel (unclipped)."""
        ix = (wp.x() - self._offset.x()) / self._zoom
        iy = (wp.y() - self._offset.y()) / self._zoom
        return QPoint(int(ix), int(iy))

    def _image_to_widget(self, ip: QPoint) -> QPoint:
        wx = ip.x() * self._zoom + self._offset.x()
        wy = ip.y() * self._zoom + self._offset.y()
        return QPoint(int(wx), int(wy))

    def _image_pt_to_normalized(self, ip: QPoint) -> tuple[float, float]:
        return (
            max(0.0, min(1.0, ip.x() / self._img_w)),
            max(0.0, min(1.0, ip.y() / self._img_h)),
        )

    def _normalized_to_widget(self, nx: float, ny: float) -> QPoint:
        ix = nx * self._img_w
        iy = ny * self._img_h
        return self._image_to_widget(QPoint(int(ix), int(iy)))

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a folder to begin")
            return

        # Draw image
        scaled = QPixmap(int(self._img_w * self._zoom), int(self._img_h * self._zoom))
        scaled = self._pixmap.scaled(
            scaled.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(self._offset, scaled)

        # Draw saved annotations
        for cls_id, norm_pts in self._annotations.items():
            if not norm_pts:
                continue
            color = QColor(CLASS_INFO[cls_id]["color"])
            self._draw_polygon_overlay(painter, norm_pts, color)

        # Draw in-progress polygon
        if self._drawing and self._draw_mode == DRAW_MODE_POLYGON and self._poly_points:
            color = QColor(CLASS_INFO[self._active_class]["color"])
            self._draw_in_progress_polygon(painter, color)

        # Draw in-progress freehand
        if self._drawing and self._draw_mode == DRAW_MODE_FREEHAND and self._freehand_points:
            color = QColor(CLASS_INFO[self._active_class]["color"])
            pts_n = [self._image_pt_to_normalized(p) for p in self._freehand_points]
            self._draw_polygon_overlay(painter, pts_n, color)

        # Eraser cursor
        if self._erase_mode:
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            painter.setPen(QPen(QColor(255, 100, 100), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            r = self._eraser_radius
            painter.drawEllipse(cursor_pos, r, r)

    def _draw_polygon_overlay(self, painter: QPainter, norm_pts, color: QColor):
        if len(norm_pts) < 2:
            return
        widget_pts = [self._normalized_to_widget(nx, ny) for nx, ny in norm_pts]
        poly = QPolygonF([p for p in widget_pts])

        fill_color = QColor(color)
        painter.setBrush(QBrush(fill_color))
        border_color = QColor(color.red(), color.green(), color.blue(), 255)
        painter.setPen(QPen(border_color, 2))
        painter.drawPolygon(poly)

        # Draw vertex dots
        painter.setBrush(QBrush(border_color))
        for p in widget_pts:
            painter.drawEllipse(p, 4, 4)

    def _draw_in_progress_polygon(self, painter: QPainter, color: QColor):
        pts = [self._image_to_widget(p) for p in self._poly_points]
        if len(pts) >= 2:
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i+1])
            # Close preview line to first point
            painter.drawLine(pts[-1], pts[0])

        painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(QBrush(color))
        for p in pts:
            painter.drawEllipse(p, 5, 5)

        # Closing circle on first point
        if pts:
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(pts[0], 8, 8)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return

        if event.button() == Qt.MiddleButton:
            self._pan_start = event.pos()
            self._pan_offset_start = QPoint(self._offset)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            self._cancel_drawing()
            return

        if event.button() != Qt.LeftButton:
            return

        img_pt = self._widget_to_image(event.pos())

        # Erase mode
        if self._erase_mode:
            self._erase_at(event.pos())
            return

        # Freehand: start drawing
        if self._draw_mode == DRAW_MODE_FREEHAND:
            self._drawing = True
            self._freehand_points = [img_pt]
            return

        # Polygon mode
        if self._draw_mode == DRAW_MODE_POLYGON:
            if not self._drawing:
                self._drawing = True
                self._poly_points = [img_pt]
            else:
                # Check if clicking near first point → close polygon
                if len(self._poly_points) >= 3:
                    first_w = self._image_to_widget(self._poly_points[0])
                    dist = math.hypot(
                        event.pos().x() - first_w.x(),
                        event.pos().y() - first_w.y()
                    )
                    if dist < 12:
                        self._finish_polygon()
                        return
                self._poly_points.append(img_pt)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._draw_mode == DRAW_MODE_POLYGON:
            if self._drawing and len(self._poly_points) >= 3:
                self._finish_polygon()

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._offset = self._pan_offset_start + delta
            self.update()
            return

        if self._drawing and self._draw_mode == DRAW_MODE_FREEHAND:
            img_pt = self._widget_to_image(event.pos())
            self._freehand_points.append(img_pt)
            self.update()
            return

        if self._draw_mode == DRAW_MODE_POLYGON and self._drawing:
            self.update()   # redraw preview line

        if self._erase_mode:
            self.update()   # redraw eraser circle

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor if self._erase_mode else Qt.CrossCursor)
            return

        if event.button() == Qt.LeftButton and self._draw_mode == DRAW_MODE_FREEHAND:
            if self._drawing and len(self._freehand_points) >= 3:
                self._finish_freehand()
            else:
                self._drawing = False
                self._freehand_points = []

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        old_zoom = self._zoom
        self._zoom = max(0.05, min(30.0, self._zoom * factor))

        # Zoom toward cursor
        cursor = event.pos()
        self._offset = QPoint(
            int(cursor.x() - (cursor.x() - self._offset.x()) * self._zoom / old_zoom),
            int(cursor.y() - (cursor.y() - self._offset.y()) * self._zoom / old_zoom),
        )
        self.update()

    def resizeEvent(self, event):
        if self._pixmap is not None:
            self._fit_to_window()
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._cancel_drawing()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self._drawing and self._draw_mode == DRAW_MODE_POLYGON and len(self._poly_points) >= 3:
                self._finish_polygon()

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _finish_polygon(self):
        norm_pts = [self._image_pt_to_normalized(p) for p in self._poly_points]
        self._annotations[self._active_class] = norm_pts
        self._drawing = False
        self._poly_points = []
        self.update()
        self.annotationChanged.emit()

    def _finish_freehand(self):
        # Simplify path using Ramer-Douglas-Peucker
        simplified = _rdp_simplify(self._freehand_points, epsilon=2.0)
        if len(simplified) >= 3:
            norm_pts = [self._image_pt_to_normalized(p) for p in simplified]
            self._annotations[self._active_class] = norm_pts
        self._drawing = False
        self._freehand_points = []
        self.update()
        self.annotationChanged.emit()

    def _cancel_drawing(self):
        self._drawing = False
        self._poly_points = []
        self._freehand_points = []
        self.update()

    def _erase_at(self, widget_pos: QPoint):
        """Remove annotation for the class whose polygon contains the click point."""
        to_remove = []
        for cls_id, norm_pts in self._annotations.items():
            if not norm_pts:
                continue
            # Check if any vertex is within eraser radius
            for nx, ny in norm_pts:
                wp = self._normalized_to_widget(nx, ny)
                dist = math.hypot(widget_pos.x() - wp.x(), widget_pos.y() - wp.y())
                if dist <= self._eraser_radius:
                    to_remove.append(cls_id)
                    break
            else:
                # Check if click is inside polygon
                wpts = [self._normalized_to_widget(nx, ny) for nx, ny in norm_pts]
                if _point_in_polygon(widget_pos, wpts):
                    to_remove.append(cls_id)

        for cls_id in to_remove:
            del self._annotations[cls_id]

        if to_remove:
            self.update()
            self.annotationChanged.emit()


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rdp_simplify(points: list[QPoint], epsilon: float) -> list[QPoint]:
    """Ramer-Douglas-Peucker line simplification."""
    if len(points) < 3:
        return points
    pts = np.array([(p.x(), p.y()) for p in points], dtype=float)

    def rdp(pts, eps):
        if len(pts) < 3:
            return list(range(len(pts)))
        start, end = pts[0], pts[-1]
        seg = end - start
        seg_len = np.linalg.norm(seg)
        if seg_len == 0:
            dists = np.linalg.norm(pts - start, axis=1)
        else:
            dists = np.abs(np.cross(seg, start - pts)) / seg_len
        idx = np.argmax(dists)
        if dists[idx] > eps:
            left  = rdp(pts[:idx+1], eps)
            right = rdp(pts[idx:],   eps)
            return left[:-1] + [i + idx for i in right]
        return [0, len(pts)-1]

    indices = rdp(pts, epsilon)
    return [points[i] for i in sorted(set(indices))]


def _point_in_polygon(point: QPoint, polygon: list[QPoint]) -> bool:
    """Ray-casting algorithm."""
    x, y = point.x(), point.y()
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x(), polygon[i].y()
        xj, yj = polygon[j].x(), polygon[j].y()
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ──────────────────────────────────────────────────────────────────────────────
# Export Dialog
# ──────────────────────────────────────────────────────────────────────────────

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Dataset")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        # Train split
        split_group = QGroupBox("Train / Val Split")
        split_layout = QHBoxLayout(split_group)
        split_layout.addWidget(QLabel("Train %:"))
        self.train_spin = QSpinBox()
        self.train_spin.setRange(50, 95)
        self.train_spin.setValue(80)
        self.train_spin.setSuffix(" %")
        split_layout.addWidget(self.train_spin)
        layout.addWidget(split_group)

        # Output folder
        out_group = QGroupBox("Output Folder")
        out_layout = QHBoxLayout(out_group)
        self.out_label = QLabel("(not set)")
        self.out_label.setWordWrap(True)
        out_btn = QPushButton("Choose…")
        out_btn.clicked.connect(self._choose_folder)
        out_layout.addWidget(self.out_label, 1)
        out_layout.addWidget(out_btn)
        layout.addWidget(out_group)

        self._out_path = ""

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._out_path = path
            self.out_label.setText(path)

    def get_values(self):
        return self.train_spin.value() / 100.0, self._out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clock Hand Annotator  –  YOLO-Seg Dataset Builder")
        self.resize(1280, 800)

        # State
        self._image_folder: Optional[Path] = None
        self._image_files: list[Path] = []
        self._current_index = -1
        self._ann_manager = AnnotationManager()
        self._autosave = True

        self._build_ui()
        self._build_shortcuts()
        self._update_ui_state()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # ── Left sidebar ──
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setFrameShape(QFrame.StyledPanel)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(8)

        # Open folder
        open_btn = QPushButton("📂  Open Image Folder")
        open_btn.clicked.connect(self._open_folder)
        sl.addWidget(open_btn)

        # Image counter
        self.img_label = QLabel("No folder loaded")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setWordWrap(True)
        sl.addWidget(self.img_label)

        # Navigation
        nav_group = QGroupBox("Navigation")
        nl = QHBoxLayout(nav_group)
        self.prev_btn = QPushButton("◀  Prev")
        self.next_btn = QPushButton("Next  ▶")
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn.clicked.connect(self._next_image)
        nl.addWidget(self.prev_btn)
        nl.addWidget(self.next_btn)
        sl.addWidget(nav_group)

        # Class selection
        class_group = QGroupBox("Active Class  (1/2/3)")
        cl = QVBoxLayout(class_group)
        self._class_btns = []
        for cls_id, info in CLASS_INFO.items():
            btn = QPushButton(f"{cls_id+1}. {info['name']} Hand")
            color = info["color"]
            btn.setStyleSheet(
                f"background-color: rgba({color.red()},{color.green()},{color.blue()},200);"
                "color: white; font-weight: bold; padding: 6px;"
            )
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, c=cls_id: self._select_class(c))
            cl.addWidget(btn)
            self._class_btns.append(btn)
        self._class_btns[0].setChecked(True)
        sl.addWidget(class_group)

        # Draw mode
        mode_group = QGroupBox("Draw Mode")
        ml = QVBoxLayout(mode_group)
        self.poly_btn = QPushButton("🔷  Polygon  (click vertices)")
        self.free_btn = QPushButton("✏️  Freehand  (drag)")
        self.erase_btn = QPushButton("🧹  Eraser")
        self.poly_btn.setCheckable(True)
        self.free_btn.setCheckable(True)
        self.erase_btn.setCheckable(True)
        self.poly_btn.setChecked(True)
        self.poly_btn.clicked.connect(lambda: self._set_draw_mode(DRAW_MODE_POLYGON))
        self.free_btn.clicked.connect(lambda: self._set_draw_mode(DRAW_MODE_FREEHAND))
        self.erase_btn.clicked.connect(self._toggle_eraser)
        ml.addWidget(self.poly_btn)
        ml.addWidget(self.free_btn)
        ml.addWidget(self.erase_btn)
        sl.addWidget(mode_group)

        # Opacity
        opacity_group = QGroupBox("Overlay Opacity")
        ol = QVBoxLayout(opacity_group)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(20, 255)
        self.opacity_slider.setValue(180)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        ol.addWidget(self.opacity_slider)
        sl.addWidget(opacity_group)

        # Annotation status
        ann_group = QGroupBox("Annotations")
        al = QVBoxLayout(ann_group)
        self.ann_status_labels = {}
        for cls_id, info in CLASS_INFO.items():
            lbl = QLabel(f"  {info['name']}: ✗")
            lbl.setStyleSheet("font-size: 12px;")
            al.addWidget(lbl)
            self.ann_status_labels[cls_id] = lbl
        sl.addWidget(ann_group)

        # Save / Clear
        self.save_btn = QPushButton("💾  Save Annotation")
        self.save_btn.clicked.connect(self._save_annotation)
        self.clear_btn = QPushButton("🗑️  Clear All")
        self.clear_btn.clicked.connect(self._clear_annotations)
        sl.addWidget(self.save_btn)
        sl.addWidget(self.clear_btn)

        # Export
        export_btn = QPushButton("📦  Export Dataset…")
        export_btn.clicked.connect(self._export_dataset)
        sl.addWidget(export_btn)

        sl.addStretch()

        # Autosave checkbox
        self.autosave_cb = QCheckBox("Auto-save on navigate")
        self.autosave_cb.setChecked(True)
        self.autosave_cb.stateChanged.connect(lambda v: setattr(self, "_autosave", bool(v)))
        sl.addWidget(self.autosave_cb)

        # ── Canvas ──
        self.canvas = AnnotationCanvas()
        self.canvas.annotationChanged.connect(self._on_annotation_changed)

        # ── Splitter ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.canvas)
        splitter.setSizes([220, 1060])
        splitter.setCollapsible(0, False)
        root_layout.addWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready  |  Open a folder to begin")

    def _build_shortcuts(self):
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence("1"), self, lambda: self._select_class(0))
        QShortcut(QKeySequence("2"), self, lambda: self._select_class(1))
        QShortcut(QKeySequence("3"), self, lambda: self._select_class(2))
        QShortcut(QKeySequence("D"), self, self._next_image)
        QShortcut(QKeySequence("A"), self, self._prev_image)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_annotation)
        QShortcut(QKeySequence("Escape"), self, self.canvas.cancel_current)
        QShortcut(QKeySequence("P"), self, lambda: self._set_draw_mode(DRAW_MODE_POLYGON))
        QShortcut(QKeySequence("F"), self, lambda: self._set_draw_mode(DRAW_MODE_FREEHAND))
        QShortcut(QKeySequence("E"), self, self._toggle_eraser)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self._image_folder = Path(folder)
        self._image_files = sorted([
            p for p in self._image_folder.iterdir()
            if p.suffix.lower() in SUPPORTED_FORMATS
        ])
        if not self._image_files:
            QMessageBox.warning(self, "No Images", "No supported images found in the selected folder.")
            return
        self._current_index = 0
        self._load_current_image()

    def _load_current_image(self):
        if self._current_index < 0 or self._current_index >= len(self._image_files):
            return
        path = self._image_files[self._current_index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.status.showMessage(f"Failed to load: {path.name}")
            return
        self.canvas.load_image(pixmap)

        # Load existing annotations if any
        existing = self._ann_manager.load(path)
        self.canvas.set_annotations(existing)
        self._update_ui_state()
        self._update_annotation_status(existing)

    def _save_annotation(self):
        if self._current_index < 0:
            return
        path = self._image_files[self._current_index]
        anns = self.canvas._annotations
        self._ann_manager.save(path, anns)
        self._update_annotation_status(anns)
        self.status.showMessage(f"Saved: {path.name}", 3000)

    def _clear_annotations(self):
        reply = QMessageBox.question(self, "Clear", "Clear all annotations for this image?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.set_annotations({})
            if self._current_index >= 0:
                path = self._image_files[self._current_index]
                self._ann_manager.delete(path)
            self._update_annotation_status({})

    def _prev_image(self):
        if self._current_index <= 0:
            return
        if self._autosave:
            self._save_annotation()
        self._current_index -= 1
        self._load_current_image()

    def _next_image(self):
        if self._current_index >= len(self._image_files) - 1:
            return
        if self._autosave:
            self._save_annotation()
        self._current_index += 1
        self._load_current_image()

    def _select_class(self, cls: int):
        self._active_class = cls
        self.canvas.set_active_class(cls)
        for i, btn in enumerate(self._class_btns):
            btn.setChecked(i == cls)
        self.status.showMessage(f"Active class: {CLASS_INFO[cls]['name']} Hand", 2000)

    def _set_draw_mode(self, mode: str):
        self.canvas.set_draw_mode(mode)
        # Uncheck eraser
        self.erase_btn.setChecked(False)
        self.canvas.set_erase_mode(False)
        self.poly_btn.setChecked(mode == DRAW_MODE_POLYGON)
        self.free_btn.setChecked(mode == DRAW_MODE_FREEHAND)
        msg = "Polygon mode: click to add vertices, double-click or click first point to close"
        if mode == DRAW_MODE_FREEHAND:
            msg = "Freehand mode: click and drag to draw"
        self.status.showMessage(msg, 4000)

    def _toggle_eraser(self):
        enabled = self.erase_btn.isChecked()
        if not enabled:
            # Was unchecked — toggle on
            self.erase_btn.setChecked(True)
            enabled = True
        self.canvas.set_erase_mode(enabled)
        self.poly_btn.setChecked(False)
        self.free_btn.setChecked(False)
        if enabled:
            self.status.showMessage("Eraser: click on an annotation to remove it", 3000)

    def _on_opacity_changed(self, v):
        self.canvas.set_opacity(v)

    def _on_annotation_changed(self):
        self._update_annotation_status(self.canvas._annotations)
        if self._autosave and self._current_index >= 0:
            self._save_annotation()

    def _update_annotation_status(self, anns: dict):
        for cls_id, lbl in self.ann_status_labels.items():
            info = CLASS_INFO[cls_id]
            if cls_id in anns and anns[cls_id]:
                pts = anns[cls_id]
                color = info["color"]
                lbl.setText(f"  {info['name']}: ✓  ({len(pts)} pts)")
                lbl.setStyleSheet(
                    f"color: rgb({color.red()},{color.green()},{color.blue()}); font-weight: bold;"
                )
            else:
                lbl.setText(f"  {info['name']}: ✗")
                lbl.setStyleSheet("color: #aaa;")

    def _update_ui_state(self):
        has_images = bool(self._image_files)
        self.prev_btn.setEnabled(has_images and self._current_index > 0)
        self.next_btn.setEnabled(has_images and self._current_index < len(self._image_files) - 1)
        if has_images:
            path = self._image_files[self._current_index]
            self.img_label.setText(
                f"{path.name}\n{self._current_index+1} / {len(self._image_files)}"
            )
        else:
            self.img_label.setText("No folder loaded")

    def _export_dataset(self):
        if not self._image_files:
            QMessageBox.warning(self, "No Images", "Load images first.")
            return

        # Auto-save current
        if self._autosave and self._current_index >= 0:
            self._save_annotation()

        dlg = ExportDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        train_ratio, out_path = dlg.get_values()
        if not out_path:
            QMessageBox.warning(self, "No Output", "Please choose an output folder.")
            return

        exporter = YOLOExporter(self._ann_manager)
        try:
            stats = exporter.export(
                image_files=self._image_files,
                output_dir=Path(out_path),
                train_ratio=train_ratio,
            )
            QMessageBox.information(
                self, "Export Complete",
                f"Dataset exported to:\n{out_path}\n\n"
                f"Train images: {stats['train']}\n"
                f"Val images:   {stats['val']}\n"
                f"Skipped (no annotations): {stats['skipped']}\n\n"
                f"data.yaml created ✓"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(45,  45,  45))
    palette.setColor(QPalette.WindowText,      Qt.white)
    palette.setColor(QPalette.Base,            QColor(30,  30,  30))
    palette.setColor(QPalette.AlternateBase,   QColor(55,  55,  55))
    palette.setColor(QPalette.ToolTipBase,     Qt.white)
    palette.setColor(QPalette.ToolTipText,     Qt.white)
    palette.setColor(QPalette.Text,            Qt.white)
    palette.setColor(QPalette.Button,          QColor(55,  55,  55))
    palette.setColor(QPalette.ButtonText,      Qt.white)
    palette.setColor(QPalette.BrightText,      Qt.red)
    palette.setColor(QPalette.Highlight,       QColor(70, 130, 200))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
