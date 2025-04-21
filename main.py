from utils.video import load_video, save_video
from analysis.pass_detection import *
from visualization.drawing import *
from analysis.zone_control import *
from config import *
from ultralytics import YOLO 
import supervision as sv
from sklearn.cluster import KMeans
from classifiers import *
import os 
from interpolations import *

def main():
    cap, width, height, fps = load_video("input_videos/1.mp4")
    yolo_model = YOLO(model_path)
    tracker = sv.ByteTrack()
    reference_colors = None
    total_frames=0
    # possession variables
    team_possession = {0: 0, 1: 0}
    current_possession = None
    ball_holder = None
    ball_hold_frames = 0
    possession_threshold = 5
    total_frames = 0
    max_history_length = 10

    counter=0
    # Create a unique output video path using timestamp
    out = save_video(output_video_path, width, height, fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        if counter==300:
            break   
        # counter+=1
        # Step 1: detect players and ball
        results = yolo_model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        players = detections[detections.class_id == 2]
        goalkeepers = detections[detections.class_id == 1]
        referees = detections[detections.class_id == 3]
        ball = detections[detections.class_id == 0]   

        # Step 2: classification based on jersey color
        if len(players) > 0:
            grass_color = get_grass_color(frame)
            grass_hsv = cv2.cvtColor(np.uint8([[grass_color]]), cv2.COLOR_BGR2HSV)[0, 0]
            jersey_colors = [
                get_jersey_color(frame[int(y1):int(y2), int(x1):int(x2)], grass_hsv)
                for x1, y1, x2, y2 in players.xyxy
            ]

            # detect the team colors
            if len(jersey_colors) >= 2 and not reference_colors:
                kmeans = KMeans(n_clusters=2).fit(jersey_colors)
                reference_colors = {
                    0: kmeans.cluster_centers_[0],
                    1: kmeans.cluster_centers_[1],
                }

            # Assign players to teams
            teams = []
            for color in jersey_colors:
                if reference_colors:
                    distances = [np.linalg.norm(color - ref) for ref in reference_colors.values()]
                    team = distances.index(min(distances))
                    teams.append(team)

        # Step 3: tracking 
        players = tracker.update_with_detections(players)

        for i, track_id in enumerate(players.tracker_id):
            if i < len(teams):
                color_history[track_id].append(teams[i])
                if len(color_history[track_id]) >= 5:
                    assigned_team = max(set(color_history[track_id]), key=color_history[track_id].count)
                    team_assignments[track_id] = assigned_team

        # Step 4: ball position interpolation 
        ball_bbox = None
        if len(ball.xyxy) > 0:
            ball_bbox = tuple(map(int, ball.xyxy[0]))
            ball_positions_history.append(ball_bbox)
        else:
            ball_positions_history.append(None)

        if len(ball_positions_history) > max_history_length:
            ball_positions_history.pop(0)

        if None in ball_positions_history:
            interpolated = interpolate_ball_positions(ball_positions_history)
            if interpolated[-1]:
                ball_bbox = interpolated[-1]

        # Step 5: calculate possession  and zones 
        zones = divide_field_into_zones(width, height, zones_x=3, zones_y=2)

        if len(players.tracker_id) > 0:
            zone_control = analyze_zone_control(players, team_assignments, zones)
        if zone_control:
            draw_zone_control(annotated, zones, zone_control, TEAM_COLORS)
            add_zone_control_legend(annotated, width, height, TEAM_COLORS)
        # clalculate possession  
        #Step 5  calculate possession 
        closest_player = None
        min_distance = float('inf')
        if ball_bbox:
            ball_center = get_center_of_bbox(ball_bbox)
            for i, track_id in enumerate(players.tracker_id):
                player_center = get_center_of_bbox(players.xyxy[i])
                dist = np.linalg.norm(np.array(ball_center) - np.array(player_center))
                if dist < min_distance and dist < 40:
                    min_distance = dist
                    closest_player = track_id

            if closest_player:
                if closest_player == ball_holder:
                    ball_hold_frames += 1
                else:
                    ball_holder = closest_player
                    ball_hold_frames = 1

                if ball_hold_frames >= possession_threshold:
                    new_possession = team_assignments.get(ball_holder, None)
                    if new_possession is not None:
                        current_possession = new_possession
                        team_possession[current_possession] += 1
                        
        total_frames += 1
        available_passes = []
        if ball_holder and ball_holder in team_assignments:
            available_passes = find_available_passes(frame, players, ball_holder, team_assignments)
   
            
            
            
        # players drawing 
        for bbox,track_id in zip(players.xyxy, players.tracker_id):
            team=team_assignments.get(track_id, 0)
            has_ball = (track_id==ball_holder and ball_hold_frames>=possession_threshold)
            draw_annotation(annotated, bbox, TEAM_COLORS[team], team, track_id, has_ball=has_ball)
        
        # ball drawing
        if ball_bbox:
            draw_triangle(annotated, ball_bbox, TEAM_COLORS['ball'])
        for referee in referees.xyxy:
            draw_annotation(annotated, referee, TEAM_COLORS['referee'], team=None, track_id=None, text="Referee", has_ball=False)
        
        
        # possession drawing
            # Draw possession
        if total_frames > 0:
            team_0_ratio = team_possession[0] / total_frames
            team_1_ratio = team_possession[1] / total_frames
            total_ratio = team_0_ratio + team_1_ratio
            team_0_possession = (team_0_ratio / total_ratio) * 100 if total_ratio > 0 else 50
            team_1_possession = 100 - team_0_possession

            possession_text = f"Team {current_possession + 1} has possession" if current_possession is not None else "No possession"
            annotated = add_transparent_rectangle(annotated, width - 400, height - 130, width - 10, height - 10, (192, 192, 192), 0.7)
            cv2.putText(annotated, f"Team 1: {team_0_possession:.1f}%", (width - 350, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, TEAM_COLORS[0], 2)
            cv2.putText(annotated, f"Team 2: {team_1_possession:.1f}%", (width - 350, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, TEAM_COLORS[1], 2)
            cv2.putText(annotated, possession_text, (width - 350, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
        if ball_holder and ball_holder in team_assignments:
            team = team_assignments[ball_holder]
            annotated = draw_pass_lines(annotated, players, available_passes, TEAM_COLORS[team],ball_holder)
    
        # Add pass statistics to the display (optional)
        if ball_holder and available_passes:
            clear_passes = sum(1 for _, _, is_clear in available_passes if is_clear)
            cv2.putText(annotated, 
                        f"Clear passes: {clear_passes}/{len(available_passes)}", 
                        (50, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 255, 255), 2)    
                

        out.write(annotated)

    cap.release()
    out.release()
    os.system(f'start {output_video_path}')

if __name__ == "__main__":
    main()
