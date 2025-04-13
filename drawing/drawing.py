from utils import get_center_of_bbox, get_bbox_width
import numpy as np 
import cv2 

def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)
    triangle_points = np.array([[x, y], [x-10, y-20], [x+10, y-20]])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    return frame


def draw_annotation(frame, bbox, color, team=None, track_id=None, text=None, has_ball=False):
    x1, y1, x2, y2 = map(int, bbox)
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    
    cv2.ellipse(frame,
            center=(x_center, int(y2)),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4)

    cv2.rectangle(frame, (x2 - 40, y2), (x2 + 25, y2 + 20), color, -1)
    
    if team is not None and track_id is not None:
        cv2.putText(frame, f"T{team+1} ID:{track_id}", (x2 - 35, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"{text}", (x2 - 35, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if has_ball:
        draw_triangle(frame, bbox, (0, 255, 0))
    return frame


def add_transparent_rectangle(frame, x1, y1, x2, y2, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
