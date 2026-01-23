import matplotlib.pyplot as plt
import numpy as np


def draw_analog_clock(hh, mm, save_path=None):
    """
    Draw a minimalist analog clock showing the time hh:mm

    Parameters:
    hh (int or str): Hours (0-23), can be in format '0X' like '09'
    mm (int or str): Minutes (0-59), can be in format '0X' like '05'
    save_path (str, optional): Path to save the image. If None, just displays the clock.
    """
    # Convert to integers if strings are provided
    hh = int(hh)
    mm = int(mm)

    # Convert to 12-hour format for drawing
    hh = hh % 12

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')

    # Draw clock circle
    circle = plt.Circle((0, 0), 1, color='white', ec='black', linewidth=3)
    ax.add_patch(circle)

    # Calculate angles (0 degrees is at 12 o'clock, clockwise)
    # Subtract from 90 degrees to start from top, negate to go clockwise
    minute_angle = np.pi / 2 - (mm * np.pi / 30)
    hour_angle = np.pi / 2 - ((hh + mm / 60) * np.pi / 6)

    # Draw minute hand (longer, thicker)
    minute_length = 0.75
    ax.plot([0, minute_length * np.cos(minute_angle)],
            [0, minute_length * np.sin(minute_angle)],
            'k-', linewidth=8)

    # Draw hour hand (shorter, thicker)
    hour_length = 0.45
    ax.plot([0, hour_length * np.cos(hour_angle)],
            [0, hour_length * np.sin(hour_angle)],
            'k-', linewidth=10)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Clock saved to {save_path}")
    else:
        plt.show()

    plt.close()