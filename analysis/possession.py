# possession.py

import numpy as np
from collections import Counter
import cv2

def get_closest_player_to_ball(players, ball_xy, team_assignments):
    """
    Deterimine the closest player to the ball 
    """
    min_distance = float("inf")
    closest_player = None

    for i, player in enumerate(players):
        if team_assignments[i] == "referee":
            continue

        player_center = player  # [x, y]
        distance = np.linalg.norm(np.array(player_center) - np.array(ball_xy))

        if distance < min_distance:
            min_distance = distance
            closest_player = i

    return closest_player


def update_possession_log(possession_log, current_holder, team_assignments):
    """
    update the possession log
    """
    if current_holder is not None:
        team = team_assignments[current_holder]
        possession_log.append(team)


def calculate_possession_percentages(possession_log):
    """
    calculate the possession percentages
    """
    total = len(possession_log)
    if total == 0:
        return {"team1": 0, "team2": 0}

    counts = Counter(possession_log)
    team1_percent = (counts.get("team1", 0) / total) * 100
    team2_percent = (counts.get("team2", 0) / total) * 100

    return {
        "team1": round(team1_percent, 2),
        "team2": round(team2_percent, 2)
    }


