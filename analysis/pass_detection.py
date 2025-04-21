from utils.geometry import *
import numpy as np
def find_available_passes(frame, players, ball_holder, team_assignments):
    """
    Find all possible pass recipients for the player with the ball.
    Returns a list of (receiver_id, distance, is_clear) tuples.
    """
    if ball_holder is None or ball_holder not in team_assignments:
        return []
    
    holder_team = team_assignments[ball_holder]
    holder_idx = None
    for i, track_id in enumerate(players.tracker_id):
        if track_id == ball_holder:
            holder_idx = i
            break
    
    if holder_idx is None:
        return []
    
    holder_position = get_center_of_bbox(players.xyxy[holder_idx])
    available_passes = []
    
    # Find teammates
    for i, track_id in enumerate(players.tracker_id):
        if track_id != ball_holder and track_id in team_assignments and team_assignments[track_id] == holder_team:
            receiver_position = get_center_of_bbox(players.xyxy[i])
            distance = np.linalg.norm(np.array(holder_position) - np.array(receiver_position))
            
            # Check if pass is clear (no opponents in the way)
            is_clear = True
            for j, other_id in enumerate(players.tracker_id):
                if other_id in team_assignments and team_assignments[other_id] != holder_team:
                    opponent_position = get_center_of_bbox(players.xyxy[j])
                    if is_point_on_line_segment(holder_position, receiver_position, opponent_position, tolerance=30):
                        is_clear = False
                        break
            
            available_passes.append((track_id, distance, is_clear))
    
    return available_passes

# Add this helper function to check if a point is between two other points
def is_point_on_line_segment(p1, p2, p3, tolerance=20):
    """
    Check if point p3 is on the line segment from p1 to p2 within a tolerance.
    """
    # Convert to numpy arrays for easier operations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Calculate distance from p3 to the line segment from p1 to p2
    line_length = np.linalg.norm(p2 - p1)
    if line_length == 0:
        return np.linalg.norm(p3 - p1) <= tolerance
    
    # Project p3 onto the line
    t = np.dot(p3 - p1, p2 - p1) / (line_length * line_length)
    
    # Check if projection is on the segment
    if 0 <= t <= 1:
        projection = p1 + t * (p2 - p1)
        distance = np.linalg.norm(p3 - projection)
        return distance <= tolerance
    
    return False

