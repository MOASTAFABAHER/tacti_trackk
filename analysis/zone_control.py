
import numpy as np
from utils.geometry import get_center_of_bbox
# Add these functions to analyze zone control

def divide_field_into_zones(width, height, zones_x=3, zones_y=2):
    """
    Divide the field into a grid of zones.
    Returns a list of (x1, y1, x2, y2) coordinates for each zone.
    """
    zone_width = width // zones_x
    zone_height = height // zones_y
    zones = []
    
    for y in range(zones_y):
        for x in range(zones_x):
            x1 = x * zone_width
            y1 = y * zone_height
            x2 = (x + 1) * zone_width
            y2 = (y + 1) * zone_height
            zones.append((x1, y1, x2, y2))
    
    return zones

def get_player_zone(player_center, zones):
    """
    Determine which zone a player is in based on their position.
    Returns the zone index.
    """
    x, y = player_center
    for i, (x1, y1, x2, y2) in enumerate(zones):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

def analyze_zone_control(players, team_assignments, zones):
    """
    Analyze which team controls each zone.
    Returns a dictionary mapping zone index to (team1_count, team2_count, dominant_team).
    """
    zone_control = {}
    for i in range(len(zones)):
        zone_control[i] = {0: 0, 1: 0, "dominant": None}
    
    for i, track_id in enumerate(players.tracker_id):
        if track_id in team_assignments:
            player_center = get_center_of_bbox(players.xyxy[i])
            zone = get_player_zone(player_center, zones)
            if zone is not None:
                team = team_assignments[track_id]
                zone_control[zone][team] += 1
    
    # Determine dominant team for each zone
    for zone in zone_control:
        team1_count = zone_control[zone][0]
        team2_count = zone_control[zone][1]
        
        if team1_count > team2_count:
            zone_control[zone]["dominant"] = 0
        elif team2_count > team1_count:
            zone_control[zone]["dominant"] = 1
        else:
            zone_control[zone]["dominant"] = None  # Tie
    
    return zone_control
