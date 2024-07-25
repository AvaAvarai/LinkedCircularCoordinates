import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QMessageBox
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QFont, QTransform
from OpenGL.GL import *
from OpenGL.GLU import *
import colorsys

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.data = None
        self.labels = None
        self.feature_names = []
        self.scaler = None
        self.class_order = None
        self.feature_order = None
        self.highlighted_index = None
        self.class_colors = None

    def initializeGL(self):
        glClearColor(1, 1, 1, 1)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(-1.5, 1.5, -1.5, 1.5)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.data is not None:
            self.draw_circular_coordinates()

    def draw_circular_coordinates(self):
        num_classes = len(np.unique(self.labels))
        num_features = self.data.shape[1]
        
        angles = np.linspace(2 * np.pi, 0, num_features + 1, endpoint=True)
        radii = np.linspace(0.2, 1, num_classes)
        
        for radius in radii:
            self.draw_circle(radius)
            for angle in angles:
                x, y = radius * np.cos(angle), radius * np.sin(angle)
                glBegin(GL_LINES)
                glVertex2f(0, 0)
                glVertex2f(x, y)
                glEnd()
        
        label_positions = self.calculate_label_positions(num_features, 1.2)
        self.draw_labels(label_positions, self.feature_order, self.feature_names, self.scaler)
        self.draw_scatter_points(self.data, self.labels, self.feature_order, angles, radii, self.class_order, self.highlighted_index)

    def draw_circle(self, radius):
        glBegin(GL_LINE_LOOP)
        for i in range(100):
            theta = 2.0 * np.pi * i / 100
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            glVertex2f(x, y)
        glEnd()

    def draw_labels(self, label_positions, feature_order, feature_names, scaler):
        for i, feature_idx in enumerate(feature_order):
            x, y = label_positions[i]
            angle = np.arctan2(y, x)
            angle_degrees = np.degrees(angle)
            original_min = scaler.data_min_[feature_idx]
            original_max = scaler.data_max_[feature_idx]
            normalized_range = f"[{0:.2f} - {1:.2f}]"
            original_range = f"[{original_min:.2f} - {original_max:.2f}]"
            label = f"{feature_names[feature_idx]}\nNorm: {normalized_range}\nOrig: {original_range}"
            self.render_text(float(x), float(y), label, angle_degrees)

    def render_text(self, x, y, text, angle):
        painter = QPainter(self)
        painter.setPen(Qt.GlobalColor.black)
        painter.setFont(QFont("Arial", 10))
        transform = QTransform()
        transform.translate(x * 100, y * 100)
        transform.rotate(angle)
        painter.setTransform(transform)
        painter.drawText(0, 0, text)
        painter.end()

    def draw_scatter_points(self, data, labels, feature_order, angles, radii, class_order, highlighted_index):
        scatter_data = []
        glPointSize(6)  # Increase point size
        for j in range(len(data)):
            class_label = labels[j]
            radius = radii[class_order.index(class_label)]
            scatter_points = []
            for i, feature_idx in enumerate(feature_order):
                data_angle = np.interp(data[j, feature_idx], [0, 1], [angles[i + 1], angles[i]])
                x, y = radius * np.cos(data_angle), radius * np.sin(data_angle)
                scatter_points.append((x, y))
                if highlighted_index is not None and j == highlighted_index:
                    x_max, y_max = radii[-1] * np.cos(data_angle), radii[-1] * np.sin(data_angle)
                    glBegin(GL_LINES)
                    glVertex2f(0, 0)
                    glVertex2f(x_max, y_max)
                    glEnd()
                    glPointSize(10)
                    glColor3f(1, 0, 0)
                    glBegin(GL_POINTS)
                    glVertex2f(x, y)
                    glEnd()
            scatter_data.append((scatter_points, class_label, j))
        
        for points, class_label, j in scatter_data:
            xs, ys = zip(*points)
            glColor3f(*self.class_colors[class_label])
            glBegin(GL_POINTS)
            for x, y in zip(xs, ys):
                glVertex2f(x, y)
            glEnd()

    def calculate_label_positions(self, num_features, radius):
        positions = []
        for i in range(num_features):
            angle = 2 * np.pi * i / num_features - np.pi / 2  # Start from the top and move clockwise
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))
        return positions

    def update_data(self, data, labels, feature_names, scaler, class_order, feature_order, class_colors, highlighted_index):
        self.data = data
        self.labels = labels
        self.feature_names = feature_names
        self.scaler = scaler
        self.class_order = class_order
        self.feature_order = feature_order
        self.class_colors = class_colors
        self.highlighted_index = highlighted_index
        self.update()

class ParallelCoordinatesWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(ParallelCoordinatesWidget, self).__init__(parent)
        self.data = None
        self.labels = None
        self.feature_names = []
        self.class_colors = None
        self.highlighted_index = None

    def initializeGL(self):
        glClearColor(1, 1, 1, 1)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.data is not None:
            self.draw_parallel_coordinates()

    def draw_parallel_coordinates(self):
        num_features = self.data.shape[1]
        num_samples = self.data.shape[0]

        # Define the plot area within the widget
        plot_left = -0.5
        plot_right = num_features - 0.5
        plot_top = 1.0
        plot_bottom = 0.0

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(plot_left, plot_right, plot_bottom, plot_top)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw axes for each feature
        glColor3f(0, 0, 0)
        glBegin(GL_LINES)
        for i in range(num_features):
            glVertex2f(i, 0)
            glVertex2f(i, 1)
        glEnd()

        for i in range(num_samples):
            class_label = self.labels[i]
            color = self.class_colors[class_label]
            glColor3f(*color)
            glBegin(GL_LINE_STRIP)
            for j in range(num_features):
                glVertex2f(j, self.data[i, j])
            glEnd()

            if self.highlighted_index is not None and i == self.highlighted_index:
                glColor3f(1, 0, 0)
                glBegin(GL_LINE_STRIP)
                for j in range(num_features):
                    glVertex2f(j, self.data[i, j])
                glEnd()

    def update_data(self, data, labels, feature_names, class_colors, highlighted_index):
        self.data = data
        self.labels = labels
        self.feature_names = feature_names
        self.class_colors = class_colors
        self.highlighted_index = highlighted_index
        self.update()

def load_and_normalize_file(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, delimiter='\t')
    else:
        raise ValueError("Unsupported file type. Please select a CSV or TXT file.")
    
    if 'class' not in df.columns:
        raise ValueError("The selected file does not contain a 'class' column.")

    labels = df['class'].astype(str)
    data = df.drop(columns=['class'])
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    feature_names = list(data.columns)
    class_names = labels.unique().tolist()
    class_names.sort()
    label_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    labels = labels.map(label_to_index)
    return normalized_data, labels, feature_names, scaler, class_names, df

def get_class_colors(labels, class_names):
    unique_labels = np.unique(labels)
    class_colors = {}
    for i, label in enumerate(unique_labels):
        hue = i / len(unique_labels)  # Subdivide the hue space linearly
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        class_colors[label] = rgb
    return class_colors

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linked Circular Coordinates")

        self.gl_widget = GLWidget(self)
        self.pc_widget = ParallelCoordinatesWidget(self)
        self.table_widget = QTableWidget()
        self.table_widget.cellClicked.connect(self.on_table_cell_clicked)

        self.initUI()

    def initUI(self):
        load_button = QPushButton('Load File')
        load_button.clicked.connect(self.load_file)

        self.class_order_combobox = QComboBox(self)
        self.feature_order_combobox = QComboBox(self)

        self.update_plot_button = QPushButton('Update Plot')
        self.update_plot_button.clicked.connect(self.update_plot)

        control_layout = QHBoxLayout()
        control_layout.addWidget(load_button)
        control_layout.addWidget(QLabel('Class Order'))
        control_layout.addWidget(self.class_order_combobox)
        control_layout.addWidget(QLabel('Feature Order'))
        control_layout.addWidget(self.feature_order_combobox)
        control_layout.addWidget(self.update_plot_button)

        control_widget = QWidget()
        control_widget.setLayout(control_layout)

        layout = QVBoxLayout()
        layout.addWidget(control_widget)

        table_layout = QHBoxLayout()
        table_layout.addWidget(self.table_widget)
        table_widget_container = QWidget()
        table_widget_container.setLayout(table_layout)
        table_widget_container.setMinimumHeight(self.height() // 3)
        layout.addWidget(table_widget_container)
        
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.gl_widget)
        plot_layout.addWidget(self.pc_widget)

        layout.addLayout(plot_layout)

        legend_layout = QHBoxLayout()
        self.legend_widget = QWidget()
        self.legend_widget.setLayout(legend_layout)
        layout.addWidget(self.legend_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        table_min_height = self.height() // 3
        self.table_widget.setMinimumHeight(table_min_height)
        self.gl_widget.setMinimumHeight(self.height() - table_min_height)
        self.pc_widget.setMinimumHeight(self.height() - table_min_height)
        self.pc_widget.setMinimumWidth(self.width() // 2)  # Ensure the Parallel Coordinates Plot fits
        super(MainWindow, self).resizeEvent(event)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Text Files (*.txt)")
        if file_path:
            try:
                data, labels, feature_names, scaler, class_names, original_df = load_and_normalize_file(file_path)
                self.data = data
                self.labels = labels
                self.feature_names = feature_names
                self.scaler = scaler
                self.class_names = class_names
                self.original_df = original_df
                self.class_colors = get_class_colors(labels, class_names)
                self.update_controls()
                self.update_plot()
                self.update_table()
                self.update_legend()
            except ValueError as e:
                msg = QMessageBox()
                msg.setText(str(e))
                msg.exec()

    def update_controls(self):
        class_permutations = [','.join(p) for p in itertools.permutations(self.class_names)]
        self.class_order_combobox.clear()
        self.class_order_combobox.addItems(class_permutations)
        self.class_order_combobox.setCurrentIndex(0)
        
        feature_permutations = [','.join(p) for p in itertools.permutations(self.feature_names)]
        self.feature_order_combobox.clear()
        self.feature_order_combobox.addItems(feature_permutations)
        self.feature_order_combobox.setCurrentIndex(0)

    def update_plot(self):
        selected_class_order = self.class_order_combobox.currentText().split(',')
        selected_feature_order = self.feature_order_combobox.currentText().split(',')
        class_order = [self.class_names.index(c.strip()) for c in selected_class_order]
        feature_order = [self.feature_names.index(f.strip()) for f in selected_feature_order]
        reordered_data = self.data[:, feature_order]
        self.gl_widget.update_data(reordered_data, self.labels, self.feature_names, self.scaler, class_order, feature_order, self.class_colors, None)
        self.pc_widget.update_data(reordered_data, self.labels, [self.feature_names[i] for i in feature_order], self.class_colors, None)
        self.update_table()

    def update_table(self):
        selected_class_order = self.class_order_combobox.currentText().split(',')
        selected_feature_order = self.feature_order_combobox.currentText().split(',')
        feature_order = [self.feature_names.index(f.strip()) for f in selected_feature_order]
        
        reordered_df = self.original_df.copy()
        reordered_df = reordered_df[[self.feature_names[i] for i in feature_order] + ['class']]
        # Map the labels back to class names
        label_to_name = {index: name for index, name in enumerate(self.class_names)}
        reordered_df['class'] = self.labels.map(label_to_name)

        self.table_widget.setRowCount(reordered_df.shape[0])
        self.table_widget.setColumnCount(reordered_df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(reordered_df.columns)

        for i in range(reordered_df.shape[0]):
            for j in range(reordered_df.shape[1]):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(reordered_df.iat[i, j])))

    def update_legend(self):
        layout = self.legend_widget.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        
        for class_label, color in self.class_colors.items():
            label = QLabel(f"{self.class_names[class_label]}")
            label.setStyleSheet(f"background-color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}); color: white; padding: 2px; margin: 2px;")
            layout.addWidget(label)

    def on_table_cell_clicked(self, row, column):
        self.gl_widget.update_data(self.data, self.labels, self.feature_names, self.scaler, self.class_order_combobox.currentText().split(','), self.feature_order_combobox.currentText().split(','), self.class_colors, row)
        self.pc_widget.update_data(self.data, self.labels, self.feature_names, self.class_colors, row)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
