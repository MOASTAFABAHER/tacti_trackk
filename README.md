# ⚽ TactiTrak – AI-Powered Football Match Analyzer

**TactiTrak** is a powerful AI-driven football analysis system designed to extract tactical insights from match footage. Built with advanced computer vision techniques, the system detects and tracks players, ball, referees, and goalkeepers, while providing zone-based dominance analysis and pass opportunity detection.

---

## 🚀 Features

- **Player & Object Detection**
  - Uses YOLO and ByteTrack to detect and track players, referees, goalkeepers, and the ball.

- **Team Classification**
  - Classifies players into teams using clustering and jersey color detection techniques.

![Alt text](https://github.com/MOASTAFABAHER/tacti_trackk/blob/583f2fa106dce41dd826292437ba80c55dfd1d90/team_classifer.png)

- **Zone Control**
  - Divides the pitch into six tactical zones.
  - Calculates which team dominates each zone based on player distribution.
![Alt text](https://github.com/MOASTAFABAHER/tacti_trackk/blob/19fa855fa197eb1cc085aeefabc882ac163b9d4a/zone_controll_mockup.png)
- **Pass Line Detection**
  - Identifies all valid passing options when a player is in possession of the ball.
  - Visualizes pass lines to support decision-making analysis.
![Alt text](https://github.com/MOASTAFABAHER/tacti_trackk/blob/7ebf625538ded761bc2000ff41e90fba6b3e6732/pass_detection.png)
- **Ball Possession Tracking**
  - Tracks and estimates ball possession over time by each team.
![Alt text](https://github.com/MOASTAFABAHER/tacti_trackk/blob/583f2fa106dce41dd826292437ba80c55dfd1d90/team_classifer.png)
---

## 💻 Platforms

TactiTrak is available on:
- **Desktop application** (for in-depth analysis and video processing)
- **Mobile application** (for quick tactical reviews and field-side usage)

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries & Frameworks:** OpenCV, Supervision, YOLOv8, ByteTrack, NumPy, Matplotlib
- **AI Models:** YOLOv8 (for detection), KMeans (for team classification)
- **Deployment:** Desktop (Tkinter / PyQt or similar), Mobile (Flutter or React Native wrapper)

---
