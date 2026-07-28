import numpy as np
from typing import List, Dict, Optional, Tuple
import supervision as sv
from sports.common.view import ViewTransformer

class PossessionTracker:
    def __init__(self, smoothing_window=5, confidence_threshold=0.6):
        self.possession_history = []
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.current_possession = None
        self.possession_start_frame = 0
        

    def update(self, possession_result: dict, frame_number: int) -> dict:
        """Update possession with temporal smoothing."""
        self.possession_history.append(possession_result)
        
        # Keep only recent history
        if len(self.possession_history) > self.smoothing_window:
            self.possession_history.pop(0)
        
        # Analyze recent history
        recent_teams = [p["team"] for p in self.possession_history[-self.smoothing_window:]]
        recent_confidences = [p["confidence"] for p in self.possession_history[-self.smoothing_window:]]
        
        # Count team possessions (excluding None)
        team_counts = {0: 0, 1: 0}
        for team in recent_teams:
            if team is not None:
                team_counts[team] += 1
        
        # Determine if we have a stable opposing team
        max_count = max(team_counts.values()) if any(team_counts.values()) else 0
        stable_teams = [team for team, count in team_counts.items() if count == max_count]
        
        # Only change possession if we have an opposing team with 3+ consistent frames
        if (len(stable_teams) == 1 and max_count >= 3 and 
            self.current_possession is not None and 
            stable_teams[0] != self.current_possession):
            
            stable_team = stable_teams[0]
            avg_confidence = np.mean([c for p, c in zip(recent_teams, recent_confidences) 
                                    if p == stable_team])
            
            if avg_confidence >= self.confidence_threshold:
                self.current_possession = stable_team
                self.possession_start_frame = frame_number
                
                return {
                    "team": stable_team,
                    "confidence": avg_confidence,
                    "duration": frame_number - self.possession_start_frame,
                    "stable": True
                }
        
        # Handle initial possession when current_possession is None
        elif self.current_possession is None and len(stable_teams) == 1 and max_count >= 3:
            stable_team = stable_teams[0]
            avg_confidence = np.mean([c for p, c in zip(recent_teams, recent_confidences) 
                                    if p == stable_team])
            
            if avg_confidence >= self.confidence_threshold:
                self.current_possession = stable_team
                self.possession_start_frame = frame_number
                
                return {
                    "team": stable_team,
                    "confidence": avg_confidence,
                    "duration": 0,
                    "stable": True
                }
        
        # Keep current possession (ignore None results)
        return {
            "team": self.current_possession,
            "confidence": 0.5,
            "duration": frame_number - self.possession_start_frame if self.current_possession else 0,
            "stable": False
        }
    
    def calculate_ball_possession_distance(
        self,
        ball_detections: sv.Detections,
        players: sv.Detections,
        players_team_id: np.ndarray, 
        goalkeepers: sv.Detections,
        goalkeepers_team_id: np.ndarray,
        keypoints: sv.KeyPoints,
        config,
        all_tracker_ids: np.ndarray = None,
        threshold: float = 100.0
    ) -> Optional[Tuple[int, int, float]]:
        """
        Calculate ball possession based on distance to closest player or goalkeeper.
        
        Returns:
            Tuple of (team_id, closest_player_tracker_id, distance) or None
        """
        # Check if we have valid keypoints for transformation
        if len(keypoints.xy[0]) == 0:
            return None
        
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if not mask.any():
            return None
        
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(config.vertices)[mask].astype(np.float32)
        )
        
        # Check for ball detection
        if len(ball_detections) == 0:
            return None
            
        ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
        ball_xy = transformer.transform_points(points=ball_xy)

        # Combine players and goalkeepers
        all_players_xy = []
        all_team_ids = []

        # Add players
        if len(players) > 0:
            players_xy = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            players_xy = transformer.transform_points(points=players_xy)
            all_players_xy.extend(players_xy)
            all_team_ids.extend(players_team_id)

        # Add goalkeepers 
        if len(goalkeepers) > 0:
            goalkeepers_xy = goalkeepers.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            goalkeepers_xy = transformer.transform_points(points=goalkeepers_xy)
            all_players_xy.extend(goalkeepers_xy)
            all_team_ids.extend(goalkeepers_team_id)

        if not all_players_xy:
            return None

        # Convert to numpy arrays
        all_players_xy = np.array(all_players_xy)
        all_team_ids = np.array(all_team_ids)

        if len(all_team_ids) == 0:
            return None

        if len(all_players_xy) != len(all_team_ids):
            return None
        
        # Calculate distances to ball
        distances = np.linalg.norm(all_players_xy - ball_xy, axis=1)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Assign possession if within threshold
        if closest_distance <= threshold:
            tracker_id = all_tracker_ids[closest_idx] if all_tracker_ids is not None else closest_idx
            return all_team_ids[closest_idx], tracker_id, closest_distance

        return None