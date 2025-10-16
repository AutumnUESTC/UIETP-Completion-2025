import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np



def create_legend(label_name_to_color, out_dir):
    # Set font properties to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24

    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(8, len(label_name_to_color) * 1))
    ax.axis('off')

    # Create legend handles
    handles = []
    for label, color in label_name_to_color.items():
        # Set color with transparency (alpha)
        print(label,color)
        color_with_alpha = tuple(np.array(color) / 255) + (0.8,)
        handles.append(plt.Line2D([0], [0], color=color_with_alpha, lw=20, label=label))

    # Draw legend
    ax.legend(handles=handles, loc='center', frameon=False, ncol=2)

    # Save legend image
    legend_dir = osp.join(out_dir, "Legend")
    os.makedirs(legend_dir, exist_ok=True)
    legend_file = osp.join(legend_dir, "legend.png")
    fig.savefig(legend_file, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Legend saved to: {legend_file}")

def main(out_dir):
    # Define label names and corresponding colors
    label_name_to_color = {
        'Background': (0, 0, 0),
        'IcedTower': (128, 0, 0),
        'IcedCamera': (0, 128, 0),
        'IcedCameraFog': (0, 0, 128),
        'IcedCameraSnow': (128, 128, 0),
        'CoveredCamera': (128, 0, 128)
    }

    create_legend(label_name_to_color, out_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save processed files", default="E:/yby/lab/Dataset")
    # args = parser.parse_args()

    main("E:/yby/lab/Dataset")
