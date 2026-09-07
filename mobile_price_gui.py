"""
📱 Mobile Price Classification - Modern PyQt5 GUI
Algorithm 1: Random Forest | Algorithm 2: SVM
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton,
    QScrollArea, QFrame, QMessageBox, QSizePolicy, QGroupBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ── Constants ──────────────────────────────────────────────
PRICE_LABELS = {
    0: 'Low Cost', 1: 'Medium Cost',
    2: 'High Cost', 3: 'Very High Cost'
}
PRICE_EMOJI = {0: '💰', 1: '💰💰', 2: '💰💰💰', 3: '💰💰💰💰'}
PRICE_COLORS_HEX = {0: '#3498db', 1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}

STYLESHEET = """
QMainWindow {
    background-color: #f0f2f5;
}
QLabel#headerTitle {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
}
QLabel#headerSub {
    font-size: 11px;
    color: #a0b4c8;
}
QFrame#sidebar {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #e0e4e8;
}
QFrame#resultCard {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #e0e4e8;
}
QLabel#fieldLabel {
    font-size: 12px;
    color: #4a5568;
    font-weight: 500;
}
QLineEdit {
    padding: 7px 10px;
    border: 1.5px solid #d1d5db;
    border-radius: 6px;
    font-size: 12px;
    background-color: #f9fafb;
    color: #1f2937;
}
QLineEdit:focus {
    border-color: #3b82f6;
    background-color: #ffffff;
}
QComboBox {
    padding: 6px 8px;
    border: 1.5px solid #d1d5db;
    border-radius: 6px;
    font-size: 12px;
    background-color: #f9fafb;
    color: #1f2937;
}
QComboBox:focus {
    border-color: #3b82f6;
}

/* ── PREDICT button (main action) ── */
QPushButton#predictBtn {
    background-color: #2563eb;
    color: #ffffff;
    font-size: 14px;
    font-weight: bold;
    padding: 12px;
    border-radius: 8px;
    border: none;
}
QPushButton#predictBtn:hover {
    background-color: #1d4ed8;
}
QPushButton#predictBtn:pressed {
    background-color: #1e40af;
}

/* ── CLEAR button ── */
QPushButton#clearBtn {
    background-color: #ffffff;
    color: #dc2626;
    font-size: 12px;
    font-weight: bold;
    padding: 9px;
    border-radius: 8px;
    border: 2px solid #dc2626;
}
QPushButton#clearBtn:hover {
    background-color: #fef2f2;
}
QPushButton#clearBtn:pressed {
    background-color: #fee2e2;
}

/* ── PRESET buttons (Budget / Mid / Flag) ── */
QPushButton#presetBtn {
    background-color: #1e293b;
    color: #ffffff;
    font-size: 12px;
    font-weight: bold;
    padding: 9px 4px;
    border-radius: 8px;
    border: none;
}
QPushButton#presetBtn:hover {
    background-color: #334155;
}
QPushButton#presetBtn:pressed {
    background-color: #0f172a;
}

QLabel#predTitle {
    font-size: 11px;
    font-weight: bold;
    color: #6b7280;
    letter-spacing: 1px;
}
QLabel#predClass {
    font-size: 18px;
    font-weight: bold;
}
QLabel#predConf {
    font-size: 11px;
    color: #6b7280;
}
QLabel#statusLabel {
    font-size: 12px;
    font-weight: bold;
    padding: 8px 14px;
    border-radius: 6px;
}
QLabel#placeholderLabel {
    font-size: 15px;
    color: #9ca3af;
}
"""


class PredictionCard(QFrame):
    """Reusable card widget showing one model's prediction."""

    def __init__(self, model_name, icon, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setMinimumHeight(110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self.title_lbl = QLabel(f"{icon}  {model_name}")
        self.title_lbl.setObjectName("predTitle")
        layout.addWidget(self.title_lbl)

        self.class_lbl = QLabel("—")
        self.class_lbl.setObjectName("predClass")
        self.class_lbl.setStyleSheet("color: #9ca3af;")
        layout.addWidget(self.class_lbl)

        self.conf_lbl = QLabel("")
        self.conf_lbl.setObjectName("predConf")
        layout.addWidget(self.conf_lbl)

    def update_prediction(self, pred_class, confidence, color):
        emoji = PRICE_EMOJI.get(pred_class, '')
        label = PRICE_LABELS.get(pred_class, 'Unknown')
        self.class_lbl.setText(f"{emoji}  {label}")
        self.class_lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        self.conf_lbl.setText(f"Confidence: {confidence:.1f}%")


class MobilePriceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📱 Mobile Price Classifier")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 780)
        self.setStyleSheet(STYLESHEET)

        self.load_models()
        self.build_ui()

    # ── Model Loading ──────────────────────────────────────
    def load_models(self):
        try:
            self.rf_model = joblib.load(os.path.join('models', 'rf_model.pkl'))
            self.svm_model = joblib.load(os.path.join('models', 'svm_model.pkl'))
            self.scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
            with open(os.path.join('models', 'model_metadata.json'), 'r') as f:
                self.meta = json.load(f)
            self.features = self.meta['selected_features']
            self.rf_acc = self.meta.get('rf_test_accuracy', 0)
            self.svm_acc = self.meta.get('svm_test_accuracy', 0)
            self.winner = self.meta.get('winner', 'SVM')
        except Exception as e:
            QMessageBox.critical(self, "Error",
                f"Cannot load models.\nRun Steps 7-9 first.\n\n{e}")
            sys.exit(1)

    # ── UI Construction ────────────────────────────────────
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header ──
        header = QFrame()
        header.setStyleSheet("background-color: #1e293b;")
        header.setFixedHeight(64)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("📱  Mobile Price Classifier")
        title.setObjectName("headerTitle")
        h_layout.addWidget(title)

        h_layout.addStretch()

        sub = QLabel("Random Forest  vs  SVM")
        sub.setObjectName("headerSub")
        h_layout.addWidget(sub)

        root_layout.addWidget(header)

        # ── Body ──
        body = QWidget()
        body.setStyleSheet("background-color: #f0f2f5;")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(16)

        # Left sidebar
        body_layout.addWidget(self.build_sidebar(), 0)
        # Right results
        self.right_panel = self.build_right_panel()
        body_layout.addWidget(self.right_panel, 1)

        root_layout.addWidget(body, 1)

    def build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(320)

        main_v = QVBoxLayout(sidebar)
        main_v.setContentsMargins(16, 16, 16, 16)
        main_v.setSpacing(10)

        # Scrollable form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        form_widget = QWidget()
        form_widget.setStyleSheet("background: transparent;")
        form_layout = QGridLayout(form_widget)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(8)

        self.inputs = {}
        fields = [
            ("RAM (MB)", "ram", "256-4000"),
            ("Battery (mAh)", "battery_power", "500-2000"),
            ("Pixel Width", "px_width", "500-1998"),
            ("Pixel Height", "px_height", "0-1960"),
            ("Memory (GB)", "int_memory", "2-64"),
            ("Clock (GHz)", "clock_speed", "0.5-3.0"),
            ("CPU Cores", "n_cores", "1-8"),
            ("Weight (g)", "mobile_wt", "80-200"),
            ("Rear Cam (MP)", "pc", "0-20"),
            ("Front Cam (MP)", "fc", "0-20"),
            ("Screen H (cm)", "sc_h", "5-19"),
            ("Screen W (cm)", "sc_w", "1-18"),
            ("Talk Time (h)", "talk_time", "2-20"),
            ("Depth (cm)", "m_dep", "0.1-1.0"),
        ]
        combos = [
            ("4G", "four_g"), ("3G", "three_g"), ("Dual SIM", "dual_sim"),
            ("Bluetooth", "blue"), ("WiFi", "wifi"), ("Touch", "touch_screen"),
        ]

        row = 0
        for label_text, key, ph in fields:
            lbl = QLabel(label_text)
            lbl.setObjectName("fieldLabel")
            form_layout.addWidget(lbl, row, 0)
            entry = QLineEdit()
            entry.setPlaceholderText(ph)
            form_layout.addWidget(entry, row, 1)
            self.inputs[key] = ('entry', entry, ph)
            row += 1

        for label_text, key in combos:
            lbl = QLabel(label_text)
            lbl.setObjectName("fieldLabel")
            form_layout.addWidget(lbl, row, 0)
            cb = QComboBox()
            cb.addItems(["1 - Yes", "0 - No"])
            form_layout.addWidget(cb, row, 1)
            self.inputs[key] = ('combo', cb, None)
            row += 1

        scroll.setWidget(form_widget)
        main_v.addWidget(scroll, 1)

        # ── PRESET BUTTONS ──
        btn_grid = QHBoxLayout()
        btn_grid.setSpacing(8)

        preset_style = """
            QPushButton {{
                background-color: {bg};
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                padding: 10px 6px;
                border-radius: 8px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

        presets = [
            ("Budget",    "budget",   "#3b82f6", "#2563eb"),
            ("Mid-Range", "midrange", "#8b5cf6", "#7c3aed"),
            ("Flagship",  "flagship", "#f59e0b", "#d97706"),
        ]
        for name, key, bg, hover in presets:
            b = QPushButton(name)
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(38)
            b.setStyleSheet(preset_style.format(bg=bg, hover=hover))
            b.clicked.connect(lambda checked, p=key: self.load_preset(p))
            btn_grid.addWidget(b)

        main_v.addLayout(btn_grid)

        # ── PREDICT BUTTON ──
        predict_btn = QPushButton("⚡  PREDICT")
        predict_btn.setCursor(Qt.PointingHandCursor)
        predict_btn.setMinimumHeight(48)
        predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                padding: 12px;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #15803d;
            }
            QPushButton:pressed {
                background-color: #166534;
            }
        """)
        predict_btn.clicked.connect(self.run_prediction)
        main_v.addWidget(predict_btn)

        # ── CLEAR BUTTON ──
        clear_btn = QPushButton("Clear All")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setMinimumHeight(38)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #dc2626;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
                border: 2px solid #dc2626;
            }
            QPushButton:hover {
                background-color: #fef2f2;
            }
            QPushButton:pressed {
                background-color: #fee2e2;
            }
        """)
        clear_btn.clicked.connect(self.clear_all)
        main_v.addWidget(clear_btn)


        return sidebar

    def build_right_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        self.right_layout = QVBoxLayout(panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(12)

        # Placeholder
        self.placeholder = QLabel("👈  Enter specs and click PREDICT")
        self.placeholder.setObjectName("placeholderLabel")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.placeholder, 1)

        return panel

    # ── Presets ────────────────────────────────────────────
    def load_preset(self, name):
        presets = {
            "budget": {
                'ram': '850', 'battery_power': '800', 'px_width': '650',
                'px_height': '400', 'int_memory': '8', 'clock_speed': '1.0',
                'n_cores': '2', 'mobile_wt': '180', 'pc': '5', 'fc': '2',
                'sc_h': '10', 'sc_w': '4', 'talk_time': '6', 'm_dep': '0.8',
                'four_g': 1, 'three_g': 0, 'dual_sim': 0, 'blue': 1,
                'wifi': 0, 'touch_screen': 1
            },
            "midrange": {
                'ram': '2100', 'battery_power': '1400', 'px_width': '1200',
                'px_height': '900', 'int_memory': '32', 'clock_speed': '1.8',
                'n_cores': '4', 'mobile_wt': '145', 'pc': '12', 'fc': '5',
                'sc_h': '14', 'sc_w': '7', 'talk_time': '12', 'm_dep': '0.5',
                'four_g': 0, 'three_g': 0, 'dual_sim': 0, 'blue': 0,
                'wifi': 0, 'touch_screen': 0
            },
            "flagship": {
                'ram': '3800', 'battery_power': '1950', 'px_width': '1850',
                'px_height': '1600', 'int_memory': '64', 'clock_speed': '2.8',
                'n_cores': '8', 'mobile_wt': '115', 'pc': '20', 'fc': '16',
                'sc_h': '18', 'sc_w': '10', 'talk_time': '19', 'm_dep': '0.2',
                'four_g': 0, 'three_g': 0, 'dual_sim': 0, 'blue': 0,
                'wifi': 0, 'touch_screen': 0
            }
        }
        data = presets[name]
        for k, v in data.items():
            ftype, widget, _ = self.inputs[k]
            if ftype == 'entry':
                widget.setText(str(v))
            else:
                widget.setCurrentIndex(v)

    def clear_all(self):
        for k, (ftype, widget, ph) in self.inputs.items():
            if ftype == 'entry':
                widget.clear()
            else:
                widget.setCurrentIndex(0)
        self.show_placeholder()

    def show_placeholder(self):
        for i in reversed(range(self.right_layout.count())):
            w = self.right_layout.itemAt(i).widget()
            if w and w is not self.placeholder:
                w.deleteLater()
        self.placeholder.show()

    # ── Input Parsing ──────────────────────────────────────
    def get_values(self):
        raw = {}
        for k, (ftype, widget, ph) in self.inputs.items():
            if ftype == 'entry':
                txt = widget.text().strip()
                if not txt:
                    QMessageBox.warning(self, "Missing", f"Fill in '{k}'")
                    return None
                try:
                    raw[k] = float(txt)
                except ValueError:
                    QMessageBox.warning(self, "Invalid", f"'{k}' must be a number")
                    return None
            else:
                raw[k] = int(widget.currentText().split(' - ')[0])
        return raw

    def engineer(self, r):
        f = dict(r)
        f['total_camera_mp'] = r['pc'] + r['fc']
        f['screen_area'] = r['sc_h'] * r['sc_w']
        f['aspect_ratio'] = r['px_width'] / (r['px_height'] + 1)
        f['pixel_density'] = np.sqrt(r['px_width']**2 + r['px_height']**2) / (r['sc_h'] + 1)
        f['ram_per_core'] = r['ram'] / (r['n_cores'] + 1)
        f['battery_wt_ratio'] = r['battery_power'] / (r['mobile_wt'] + 1)
        f['storage_ram_ratio'] = r['int_memory'] / ((r['ram'] / 1024) + 0.1)
        f['connectivity_score'] = r['blue'] + r['dual_sim'] + r['four_g'] + r['three_g'] + r['touch_screen'] + r['wifi']
        f['performance_score'] = (r['ram'] * r['clock_speed'] * r['n_cores']) / 1000
        ram = r['ram']
        f['ram_category'] = 0 if ram < 1000 else (1 if ram < 2000 else (2 if ram < 3000 else 3))
        bp = r['battery_power']
        f['battery_category'] = 0 if bp < 1000 else (1 if bp < 1500 else 2)
        return f

    # ── Prediction ─────────────────────────────────────────
    def run_prediction(self):
        raw = self.get_values()
        if raw is None:
            return
        try:
            eng = self.engineer(raw)
            df_in = pd.DataFrame([eng])
            for feat in self.features:
                if feat not in df_in.columns:
                    df_in[feat] = 0
            df_in = df_in[self.features]
            scaled = self.scaler.transform(df_in)

            rf_pred = int(self.rf_model.predict(scaled)[0])
            rf_proba = self.rf_model.predict_proba(scaled)[0]
            svm_pred = int(self.svm_model.predict(scaled)[0])
            svm_proba = self.svm_model.predict_proba(scaled)[0]

            self.show_results(rf_pred, rf_proba, svm_pred, svm_proba)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ── Results Dashboard ──────────────────────────────────
    def show_results(self, rf_pred, rf_proba, svm_pred, svm_proba):
        self.show_placeholder()  # clear old
        self.placeholder.hide()

        # ── Top row: two prediction cards + status ──
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        rf_card = PredictionCard("RANDOM FOREST", "🌲")
        rf_card.update_prediction(rf_pred, rf_proba[rf_pred] * 100,
                                  PRICE_COLORS_HEX[rf_pred])
        top_row.addWidget(rf_card, 1)

        svm_card = PredictionCard("SVM CLASSIFIER", "🎯")
        svm_card.update_prediction(svm_pred, svm_proba[svm_pred] * 100,
                                   PRICE_COLORS_HEX[svm_pred])
        top_row.addWidget(svm_card, 1)

        self.right_layout.addLayout(top_row)

        # ── Status bar ──
        agree = rf_pred == svm_pred
        status = QLabel()
        status.setObjectName("statusLabel")
        status.setAlignment(Qt.AlignCenter)
        if agree:
            status.setText(f"✅  Both models agree  •  🏆 Best model: {self.winner}")
            status.setStyleSheet(
                "background-color: #ecfdf5; color: #065f46; "
                "border: 1px solid #a7f3d0; border-radius: 6px; padding: 8px;")
        else:
            status.setText(f"⚠️  Models disagree  •  🏆 Best model: {self.winner}")
            status.setStyleSheet(
                "background-color: #fffbeb; color: #92400e; "
                "border: 1px solid #fde68a; border-radius: 6px; padding: 8px;")
        self.right_layout.addWidget(status)

        # ── Charts ──
        chart_frame = QFrame()
        chart_frame.setObjectName("resultCard")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(8, 8, 8, 8)

        fig = Figure(figsize=(8, 5.5), facecolor='white', dpi=100)

        # 1) Probability bar chart
        ax1 = fig.add_subplot(221)
        cats = ['Low', 'Med', 'High', 'Flag']
        x = np.arange(4)
        w = 0.35
        ax1.bar(x - w/2, rf_proba * 100, w, label='RF', color='#3b82f6',
                edgecolor='white', linewidth=0.5)
        ax1.bar(x + w/2, svm_proba * 100, w, label='SVM', color='#10b981',
                edgecolor='white', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats, fontsize=8)
        ax1.set_title("Class Probabilities", fontsize=9, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.legend(fontsize=7)
        ax1.tick_params(labelsize=7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2) RF pie
        ax2 = fig.add_subplot(222)
        colors_pie = list(PRICE_COLORS_HEX.values())
        ax2.pie(rf_proba, labels=cats, autopct='%1.0f%%', colors=colors_pie,
                startangle=90, textprops={'fontsize': 7},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        ax2.set_title("Random Forest", fontsize=9, fontweight='bold')

        # 3) SVM pie
        ax3 = fig.add_subplot(223)
        ax3.pie(svm_proba, labels=cats, autopct='%1.0f%%', colors=colors_pie,
                startangle=90, textprops={'fontsize': 7},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        ax3.set_title("SVM Classifier", fontsize=9, fontweight='bold')

        # 4) Benchmark accuracy
        ax4 = fig.add_subplot(224)
        models = ['Random\nForest', 'SVM\nClassifier']
        accs = [self.rf_acc, self.svm_acc]
        bars = ax4.bar(models, accs, color=['#3b82f6', '#10b981'],
                       width=0.45, edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, accs):
            ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                     f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax4.set_ylim(0, 115)
        ax4.set_title("Benchmark Accuracy", fontsize=9, fontweight='bold')
        ax4.tick_params(labelsize=8)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        fig.tight_layout(pad=1.5)

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(canvas)

        self.right_layout.addWidget(chart_frame, 1)


# ── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MobilePriceApp()
    window.show()
    sys.exit(app.exec_())
