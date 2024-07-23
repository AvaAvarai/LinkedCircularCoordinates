import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itertools

def load_and_normalize_iris():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, labels, list(iris.feature_names), scaler, list(iris.target_names)

def calculate_label_positions(num_features, radius):
    positions = []
    for i in range(num_features):
        angle = 2 * np.pi * i / num_features - np.pi / 2  # Start from the top and move clockwise
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append((x, y))
    return positions

def plot_iris_on_circles(data, labels, feature_names, scaler, class_order, feature_order, ax):
    ax.clear()
    
    num_classes = len(np.unique(labels))
    num_features = data.shape[1]
    hsv_colors = [mcolors.hsv_to_rgb((i / num_classes, 1, 1)) for i in range(num_classes)]
    
    angles = np.linspace(0, 2 * np.pi, num_features + 1, endpoint=True)
    radii = np.linspace(1, num_classes, num_classes)
    
    # Draw circles and sectors
    for radius in radii:
        circle = plt.Circle((0, 0), radius, color='black', fill=False, linestyle='dashed')
        ax.add_artist(circle)
        for angle in angles:
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            ax.plot([0, x], [0, y], color='black', linestyle='dashed', linewidth=0.5)
    
    # Calculate label positions dynamically
    label_positions = calculate_label_positions(num_features, radii[-1] * 1.35)
    
    for i, feature_idx in enumerate(feature_order):
        # Adjust the starting position to ensure the first feature is at the top
        adjusted_index = i  # This is the fix, ensuring clockwise order
        sector_start = angles[adjusted_index]
        sector_end = angles[adjusted_index + 1]
        
        # Calculate original range from normalized range
        original_min = scaler.data_min_[feature_idx]
        original_max = scaler.data_max_[feature_idx]
        normalized_range = f"[{0:.2f} - {1:.2f}]"
        original_range = f"[{original_min:.2f} - {original_max:.2f}]"
        
        # Determine label position
        x, y = label_positions[adjusted_index]
        
        ax.text(x, y, f"{feature_names[feature_idx]}\nNorm: {normalized_range}\nOrig: {original_range}", 
                ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Scatter points and create legend handles
    legend_handles = []
    for class_index in class_order:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hsv_colors[class_index], markersize=10, label=class_names[class_index]))

    for j in range(len(data)):
        class_label = labels[j]
        radius = radii[class_order.index(class_label)]
        for i, feature_idx in enumerate(feature_order):
            adjusted_index = i  # This is the fix, ensuring clockwise order
            sector_start = angles[adjusted_index]
            sector_end = angles[adjusted_index + 1]
            data_angle = np.interp(data[j, feature_idx], [0, 1], [sector_start, sector_end])
            x, y = radius * np.cos(data_angle), radius * np.sin(data_angle)
            ax.scatter(x, y, color=hsv_colors[class_label], alpha=0.33)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_parallel_coordinates(data, labels, feature_names, class_order, feature_order, ax2):
    ax2.clear()
    num_classes = len(np.unique(labels))
    hsv_colors = [mcolors.hsv_to_rgb((i / num_classes, 1, 1)) for i in range(num_classes)]
    
    # Create DataFrame for parallel coordinates
    df = pd.DataFrame(data, columns=feature_names)
    df['Class'] = labels
    
    # Reorder DataFrame based on selected feature order
    reordered_columns = [feature_names[i] for i in feature_order] + ['Class']
    df = df[reordered_columns]
    
    parallel_coordinates(df, 'Class', color=hsv_colors, ax=ax2, linewidth=1)
    
    ax2.set_xticklabels([feature_names[i] for i in feature_order], rotation=15, ha='right')
    ax2.legend().set_visible(False)

def update_plot(*args):
    selected_class_order = class_order_combobox.get()
    selected_feature_order = feature_order_combobox.get()
    class_order = [class_names.index(c.strip()) for c in selected_class_order.split(',')]
    feature_order = [feature_names.index(f.strip()) for f in selected_feature_order.split(',')]
    plot_iris_on_circles(data, labels, feature_names, scaler, class_order, feature_order, ax)
    plot_parallel_coordinates(data, labels, feature_names, class_order, feature_order, ax2)
    canvas.draw()

data, labels, feature_names, scaler, class_names = load_and_normalize_iris()

root = tk.Tk()
root.title("Scatterplot Control Panel")
root.geometry("1450x800")

# Main frame to hold plot and controls
mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Plot frame
plot_frame = ttk.Frame(mainframe, padding="10 10 10 10")
plot_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})
fig.suptitle("SCC Scatterplot Multi-Axes vs Parallel Coordinates", y=0.99, x=0.45)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
plot_iris_on_circles(data, labels, feature_names, scaler, list(range(len(class_names))), list(range(len(feature_names))), ax)
plot_parallel_coordinates(data, labels, feature_names, list(range(len(class_names))), list(range(len(feature_names))), ax2)

# Create a single legend for both plots
legend_handles = []
hsv_colors = [mcolors.hsv_to_rgb((i / len(class_names), 1, 1)) for i in range(len(class_names))]
for i, class_name in enumerate(class_names):
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hsv_colors[i], markersize=10, label=class_name))
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.45, 0.875), ncol=len(class_names), title="Classes")

# Frame for controls
control_frame = ttk.Frame(mainframe, padding="0 0 0 0")
control_frame.grid(column=0, row=1, sticky=(tk.W, tk.E))

# Class order combobox
ttk.Label(control_frame, text="Class Order").grid(column=0, row=0, sticky=tk.W)
class_permutations = [','.join(p) for p in itertools.permutations(class_names)]
class_order_combobox = ttk.Combobox(control_frame, values=class_permutations, width=40)
class_order_combobox.grid(column=1, row=0, sticky=(tk.W, tk.E))
class_order_combobox.set(class_permutations[0])  # Default class order

# Feature order combobox
ttk.Label(control_frame, text="Feature Order").grid(column=0, row=1, sticky=tk.W)
feature_permutations = [','.join(p) for p in itertools.permutations(feature_names)]
feature_order_combobox = ttk.Combobox(control_frame, values=feature_permutations, width=40)
feature_order_combobox.grid(column=1, row=1, sticky=(tk.W, tk.E))
feature_order_combobox.set(feature_permutations[0])  # Default feature order

ttk.Button(control_frame, text="Update Plot", command=update_plot).grid(column=1, row=2, sticky=tk.E)

# Adjust grid configurations for resizing
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.rowconfigure(1, weight=0)

plot_frame.columnconfigure(0, weight=1)
plot_frame.rowconfigure(0, weight=1)

control_frame.columnconfigure(1, weight=1)

root.mainloop()
