from utils.geometry import get_center_of_bbox,get_bbox_width
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

def draw_zone_control(frame, zones, zone_control, team_colors):
    """
    Visualize zone control by overlaying semi-transparent colors on the field.
    """
    overlay = frame.copy()
    
    for zone_idx, (x1, y1, x2, y2) in enumerate(zones):
        dominant_team = zone_control[zone_idx]["dominant"]
        team1_count = zone_control[zone_idx][0]
        team2_count = zone_control[zone_idx][1]
        
        # Draw zone outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Add zone number
        cv2.putText(frame, f"Zone {zone_idx+1}", (x1+10, y1+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Determine overlay color based on dominant team
        if dominant_team is not None:
            color = team_colors[dominant_team]
            # Calculate opacity based on dominance
            total_players = team1_count + team2_count
            if total_players > 0:
                dominance_ratio = max(team1_count, team2_count) / total_players
                opacity = 0.2 + (0.3 * dominance_ratio)  # Between 0.2 and 0.5
            else:
                opacity = 0.2
                
            # Draw filled rectangle with team color
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Add player count
            text = f"{team1_count}:{team2_count}"
            cv2.putText(overlay, text, (x1+10, y1+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    return frame

def add_zone_control_legend(frame, width, height, team_colors):
    
    """
    Add a legend explaining the zone control visualization
    """
    # Create a semi-transparent background for the legend
    x2= width - 200
    y2 = 70
    frame = add_transparent_rectangle(frame, x2-75, y2-60, x2+150, y2+20, (0, 0, 0), 0.7)
    
    # Add title
    cv2.putText(frame, "Zone Control Legend:", (x2 -70, y2 - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add team colors
    cv2.rectangle(frame, ( x2-70, y2-20), (x2-60, y2-10), team_colors[0], -1)
    cv2.putText(frame, "Team 1", (x2-55, y2-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.rectangle(frame, (x2-70, y2), ( x2-60,y2+10), team_colors[1], -1)
    cv2.putText(frame, "Team 2", (x2-55, y2+10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


def draw_possession_bar(frame, percentages):
    """
Draw a possession bar on the frame.
    """
    height, width, _ = frame.shape
    bar_width = int(width * 0.8)
    bar_height = 20
    start_x = int((width - bar_width) / 2)
    start_y = 20

    team1_width = int((percentages["team1"] / 100) * bar_width)
    team2_width = bar_width - team1_width

    # Team1 
    cv2.rectangle(frame, (start_x, start_y),
                  (start_x + team1_width, start_y + bar_height), (0, 0, 255), -1)

    # Team2 
    cv2.rectangle(frame, (start_x + team1_width, start_y),
                  (start_x + team1_width + team2_width, start_y + bar_height), (0, 255, 0), -1)
# percentages
    cv2.putText(frame, f"{percentages['team1']}%", (start_x, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{percentages['team2']}%", (start_x + bar_width - 50, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return frame

# Function to draw pass lines
def draw_pass_lines(frame, players, available_passes, team_color,ball_holder):
    """
    Draw lines representing available passes on the frame.
    Green lines for clear passes, red lines for blocked passes.
    """
    holder_idx = None
    for i, track_id in enumerate(players.tracker_id):
        if track_id == ball_holder:
            holder_idx = i
            break
    
    if holder_idx is None:
        return frame
    
    holder_position = get_center_of_bbox(players.xyxy[holder_idx])
    
    for receiver_id, distance, is_clear in available_passes:
        for i, track_id in enumerate(players.tracker_id):
            if track_id == receiver_id:
                receiver_position = get_center_of_bbox(players.xyxy[i])
                color = (0, 255, 0) if is_clear else (0, 0, 255)  # Green for clear, red for blocked
                thickness = 2 if distance < 150 else 1  # Thicker lines for closer players
                
                # Draw the pass line
                cv2.line(frame, 
                    (int(holder_position[0]), int(holder_position[1])),
                    (int(receiver_position[0]), int(receiver_position[1])),
                    color, thickness)
                
                # Draw a small circle at the end to indicate the receiver
                cv2.circle(frame, 
                    (int(receiver_position[0]), int(receiver_position[1])),
                    5, color, -1)
    
    return frame

