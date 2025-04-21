from collections import defaultdict
import cv2


model_path = 'models/best.pt'



zones = None
zone_control = None
reference_colors = None



# additional variables
reference_colors = None
team_assignments = {}
color_history = defaultdict(list)
ball_positions_history = []


# Team colors
TEAM_COLORS = {
    0: (0, 0, 255),  # Team 1 - Red
    1: (255, 0, 0),  # Team 2 - Blue
    "goalkeeper": (0, 0, 255),
    "referee": (0, 255, 255),
    "ball": (255, 255, 255)
}