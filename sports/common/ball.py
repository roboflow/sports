from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors based on possession.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
        possession_colors (dict): Colors for different possession states.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.possession_buffer = deque(maxlen=buffer_size)  # Store possession states
        self.radius = radius
        self.thickness = thickness
        
        # Define possession colors
        self.possession_colors = {
            0: sv.Color.from_hex('#FF1493'),  # Team 0 - Pink
            1: sv.Color.from_hex('#00BFFF'),  # Team 1 - Blue
            None: sv.Color.from_hex('#FFFFFF')  # No possession - White
        }

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections, possession_team: int = None) -> np.ndarray:
        """
        Annotates the frame with circles based on detections and possession state.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.
            possession_team (int, optional): The team that has possession (0, 1, or None).

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        self.possession_buffer.append(possession_team)
        
        for i, (xy_positions, possession) in enumerate(zip(self.buffer, self.possession_buffer)):
            # Use possession color if available, otherwise use default color palette
            if possession is not None:
                color = self.possession_colors[possession]
                # Add transparency effect for older positions
                alpha = 0.3 + 0.7 * (i / max(1, len(self.buffer) - 1))
            else:
                color = self.color_palette.by_idx(i)
                alpha = 1.0
            
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            
            for center in xy_positions:
                # Draw main circle with possession color
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
                
                # Add possession indicator (smaller inner circle for current position)
                if i == len(self.buffer) - 1 and possession is not None:
                    # Inner circle to emphasize current possession
                    inner_radius = max(2, interpolated_radius // 3)
                    inner_color = sv.Color.from_hex('#FFFF00')  # Yellow center for emphasis
                    frame = cv2.circle(
                        img=frame,
                        center=tuple(center),
                        radius=inner_radius,
                        color=inner_color.as_bgr(),
                        thickness=-1  # Filled circle
                    )
                    
                    # Add possession text near the ball
                    possession_text = f"T{possession}" if possession is not None else "?"
                    text_position = (center[0] + interpolated_radius + 5, center[1] - 5)
                    cv2.putText(
                        frame, possession_text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
        
        return frame

    def get_possession_trail_color(self, possession_sequence: list) -> sv.Color:
        """
        Determine trail color based on recent possession history.
        
        Args:
            possession_sequence (list): Recent possession states
            
        Returns:
            sv.Color: Color representing the dominant possession
        """
        if not possession_sequence:
            return self.possession_colors[None]
        
        # Count possession occurrences
        possession_counts = {0: 0, 1: 0, None: 0}
        for p in possession_sequence:
            possession_counts[p] += 1
        
        # Return color of most frequent possession
        dominant_possession = max(possession_counts, key=possession_counts.get)
        return self.possession_colors[dominant_possession]


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]

    def predict(self, BALL_CLASS_ID) -> np.ndarray:
        """
        Predicts the average position of the ball based on the buffer.

        Returns:
            np.ndarray: The average position of the ball.
            If the buffer is empty, returns None.
        """
        if len(self.buffer) > 1:
            last_ball_position = self.buffer[-1]
            last_ball_position_2 = self.buffer[-2]

            # Check if both positions have detections
            if len(last_ball_position) > 0 and len(last_ball_position_2) > 0:
                last_pos = last_ball_position[0]  # Most recent position [x, y]
                prev_pos = last_ball_position_2[0]  # Previous position [x, y]

                # Calculate velocity vector (movement per frame)
                velocity = last_pos - prev_pos  # [dx, dy]

                # Project current position based on velocity
                current_ball_position = last_pos + velocity  # Predict next position

                # Create a predicted detection
                ball_detections = sv.Detections(
                    xyxy=np.array([[current_ball_position[0] - 6, current_ball_position[1] - 6,
                                   current_ball_position[0] + 6, current_ball_position[1] + 6]]),
                    class_id=np.array([BALL_CLASS_ID]),
                    confidence=np.array([0.5]),  # Lower confidence for predicted
                    tracker_id=np.array([0])
                )
                return ball_detections
        elif len(self.buffer) == 1:
            last_ball_position = self.buffer[-1]
            if len(last_ball_position) > 0:
                return sv.Detections(
                    xyxy=np.array([[last_ball_position[0][0] - 6, last_ball_position[0][1] - 6,
                                   last_ball_position[0][0] + 6, last_ball_position[0][1] + 6]]),
                    class_id=np.array([BALL_CLASS_ID]),
                    confidence=np.array([0.5]),  # Lower confidence for predicted
                    tracker_id=np.array([0])
                )
        return None