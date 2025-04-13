from utils import *
import cv2 
import numpy as np 
from ultralytics import YOLO
import supervision as sv 
from classifiers import *
from sklearn.cluster import KMeans
from collections import defaultdict
import os 

from drawing import *
def main():
    
    # define the input and output video paths
    object_detection_model_path='models/best.pt'
    input_video_path='input_videos/2.mp4'
    output_video_path='output_videos/output.mp4'    
    
    #intialize counter to stop the while to view sample of video 
    counter=0
    #read video 
    cap=cv2.VideoCapture(input_video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(output_video_path,fourcc,fps,(width,height))
    
    #load model
    model=YOLO(object_detection_model_path)
    
    #intialize tracker
    tracker=sv.ByteTrack()
    team_assignments = {}  # {track_id: team}
    # Team colors
    TEAM_COLORS = {
    0: (0, 0, 255),  # Team 1 - Red
    1: (255, 0, 0),  # Team 2 - Blue
    "goalkeeper": (0, 0, 255),
    "referee": (0, 244, 244),
    "ball": (255, 255, 255)
}

    
    #intialize reference colors
    reference_colors=None
    color_history = defaultdict(list)  # {track_id: [colors]} for stability

    
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        if counter==60:
            break
        # counter+=1
        
        
        #make prediction of the model 
        results=model(frame)[0]
        #convert the results to detections as sv format 
        detections = sv.Detections.from_ultralytics(results)
        
        ball=detections[detections.class_id==0]
        goalkeepers=detections[detections.class_id==1]
        players=detections[detections.class_id==2]
        referees=detections[detections.class_id==3]
        annotated=frame.copy()
        grass_color=get_grass_color(frame)
        grass_hsv=cv2.cvtColor(np.uint8([[grass_color]]),cv2.COLOR_BGR2HSV)[0][0]
        
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
    
    # Tracking
        players = tracker.update_with_detections(players)
        # Update team assignments with stabilization
        for i, (track_id, confidence) in enumerate(zip(players.tracker_id, players.confidence)):
            if track_id not in color_history:
                color_history[track_id] = []
        
            if i < len(teams) and confidence > 0.7:
                color_history[track_id].append(teams[i])
            
            # Require 5 consistent frames before assigning
                if len(color_history[track_id]) >= 5:
                    # Use most frequent team assignment
                    team = max(set(color_history[track_id]), key=color_history[track_id].count)
                    team_assignments[track_id] = team
                    
        #Visualization
        for bbox, track_id in zip(players.xyxy, players.tracker_id):
            team = team_assignments.get(track_id, 0)
            annotated = draw_annotation(annotated, bbox=bbox, color=TEAM_COLORS[team],
                                  track_id=track_id, team=team, has_ball=False)
        # for bbox in goalkeepers.xyxy:
        #     annotated = draw_annotation(annotated, bbox=bbox, color=TEAM_COLORS["goalkeeper"],
        #                     text='Goalkeeper')
        for referee in referees.xyxy:
            annotated = draw_annotation(annotated, bbox=referee, color=TEAM_COLORS["referee"],
                                text='Referee')
        

        out.write(annotated)
    cap.release()
    out.release()
    os.system(f'start {output_video_path} ')
if __name__=="__main__":
    
    main()
    
        
        
    

        
        

        