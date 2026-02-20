import cv2
import numpy as np

def draw_enhanced_text(frame, text, position, font_scale=1.0, color=(255, 255, 255), 
                      thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      background_color=(0, 0, 0), background_alpha=0.7,
                      outline_color=(0, 0, 0), outline_thickness=3):
    """
    Draw text with enhanced visibility including background, outline, and shadow effects.
    
    Args:
        frame: The image frame to draw on
        text: Text to display
        position: (x, y) position for the text
        font_scale: Font size multiplier
        color: Text color (B, G, R)
        thickness: Text thickness
        font: Font type
        background_color: Background color (B, G, R)
        background_alpha: Background transparency (0.0 to 1.0)
        outline_color: Outline color (B, G, R)
        outline_thickness: Outline thickness
    """
    x, y = position
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle
    padding = 10
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # Ensure coordinates are within frame bounds
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(frame.shape[1], bg_x2)
    bg_y2 = min(frame.shape[0], bg_y2)
    
    # Create background overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
    cv2.addWeighted(overlay, background_alpha, frame, 1 - background_alpha, 0, frame)
    
    # Draw text outline (shadow effect)
    shadow_offset = 2
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset), 
                font, font_scale, outline_color, outline_thickness)
    
    # Draw main text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def draw_violation_text(frame, text, position, violation_type="general"):
    """
    Draw violation text with specific styling based on violation type.
    
    Args:
        frame: The image frame to draw on
        text: Text to display
        position: (x, y) position for the text
        violation_type: Type of violation ("helmet", "glove", "occupancy", "checking")
    """
    # Define colors and styles for different violation types
    styles = {
        "helmet": {
            "color": (0, 0, 255),  # Red
            "background_color": (0, 0, 100),  # Dark red background
            "font_scale": 1.2,
            "thickness": 3
        },
        "glove": {
            "color": (0, 0, 255),  # Red
            "background_color": (0, 0, 100),  # Dark red background
            "font_scale": 1.2,
            "thickness": 3
        },
        "occupancy": {
            "color": (0, 0, 255),  # Red
            "background_color": (0, 0, 100),  # Dark red background
            "font_scale": 1.1,
            "thickness": 3
        },
        "checking": {
            "color": (0, 165, 255),  # Orange
            "background_color": (0, 80, 120),  # Dark orange background
            "font_scale": 1.0,
            "thickness": 2
        },
        "general": {
            "color": (255, 255, 255),  # White
            "background_color": (0, 0, 0),  # Black background
            "font_scale": 1.0,
            "thickness": 2
        }
    }
    
    style = styles.get(violation_type, styles["general"])
    
    draw_enhanced_text(
        frame=frame,
        text=text,
        position=position,
        font_scale=style["font_scale"],
        color=style["color"],
        thickness=style["thickness"],
        background_color=style["background_color"],
        background_alpha=0.8,
        outline_color=(0, 0, 0),
        outline_thickness=4
    )

def draw_status_text(frame, text, position, status_type="normal"):
    """
    Draw status text with appropriate styling.
    
    Args:
        frame: The image frame to draw on
        text: Text to display
        position: (x, y) position for the text
        status_type: Type of status ("normal", "warning", "info")
    """
    styles = {
        "normal": {
            "color": (0, 255, 0),  # Green
            "background_color": (0, 100, 0),  # Dark green background
            "font_scale": 1.0,
            "thickness": 2
        },
        "warning": {
            "color": (0, 165, 255),  # Orange
            "background_color": (0, 80, 120),  # Dark orange background
            "font_scale": 1.1,
            "thickness": 2
        },
        "info": {
            "color": (255, 255, 255),  # White
            "background_color": (0, 0, 0),  # Black background
            "font_scale": 0.9,
            "thickness": 2
        }
    }
    
    style = styles.get(status_type, styles["normal"])
    
    draw_enhanced_text(
        frame=frame,
        text=text,
        position=position,
        font_scale=style["font_scale"],
        color=style["color"],
        thickness=style["thickness"],
        background_color=style["background_color"],
        background_alpha=0.7,
        outline_color=(0, 0, 0),
        outline_thickness=3
    ) 