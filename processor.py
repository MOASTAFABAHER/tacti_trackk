# processor.py
import cv2
import numpy as np
import os
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans
from collections import defaultdict

from utils import *
from classifiers import *
from drawing import *
from download_file import *
def process_video(input_video_path, output_video_path):
    download_model_if_not_exists(file_id='147f8MufseFwU_byEWAe7ejjMUNTw9Jbz', path='models/pitch_model.pt')
    download_model_if_not_exists(file_id="1HaIOi1V9PYGcpM4Iflz6QuQ8RfdsgwaR", path="models/best.pt")

    
    # if 'models/pitch_model.pt' not in os.listdir('models'):
    #     download_file(url_pitch, 'models/pitch_model.pt')
    
    model = YOLO('models/best.pt')
    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    team_assignments = {}
    TEAM_COLORS = {
        0: (0, 0, 255),
        1: (255, 0, 0),
        "goalkeeper": (0, 0, 255),
        "referee": (0, 244, 244),
        "ball": (255, 255, 255)
    }

    reference_colors = None
    color_history = defaultdict(list)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or counter == 60:
            break
        counter+=1

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        ball = detections[detections.class_id == 0]
        goalkeepers = detections[detections.class_id == 1]
        players = detections[detections.class_id == 2]
        referees = detections[detections.class_id == 3]

        annotated = frame.copy()
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[grass_color]]), cv2.COLOR_BGR2HSV)[0][0]

        jersey_colors = [get_jersey_color(frame[int(y1):int(y2), int(x1):int(x2)], grass_hsv)
                         for x1, y1, x2, y2 in players.xyxy]

        if len(jersey_colors) >= 2 and not reference_colors:
            kmeans = KMeans(n_clusters=2).fit(jersey_colors)
            reference_colors = {
                0: kmeans.cluster_centers_[0],
                1: kmeans.cluster_centers_[1],
                "goalkeeper": (0, 255, 255),
                "referee": (0, 165, 255),
                "ball": (0, 255, 0)
            }

        teams = []
        for color in jersey_colors:
            if reference_colors:
                distances = [np.linalg.norm(color - ref) for ref in reference_colors.values()]
                team = min(range(2), key=lambda x: distances[x])
                teams.append(team)
            else:
                teams.append(0)

        players = tracker.update_with_detections(players)
        for i, (track_id, confidence) in enumerate(zip(players.tracker_id, players.confidence)):
            if track_id not in color_history:
                color_history[track_id] = []
            if i < len(teams) and confidence > 0.7:
                color_history[track_id].append(teams[i])
                if len(color_history[track_id]) >= 5:
                    team = max(set(color_history[track_id]), key=color_history[track_id].count)
                    team_assignments[track_id] = team

        for bbox, track_id in zip(players.xyxy, players.tracker_id):
            team = team_assignments.get(track_id, 0)
            annotated = draw_annotation(annotated, bbox=bbox, color=TEAM_COLORS[team],
                                        track_id=track_id, team=team, has_ball=False)

        for referee in referees.xyxy:
            annotated = draw_annotation(annotated, bbox=referee, color=TEAM_COLORS["referee"], text='Referee')

        out.write(annotated)
        # counter += 1

    cap.release()
    out.release()
