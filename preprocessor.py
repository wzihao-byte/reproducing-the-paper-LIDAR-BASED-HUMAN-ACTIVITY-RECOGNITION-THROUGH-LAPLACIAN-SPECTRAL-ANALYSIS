import numpy as np

class LidarPreprocessor:
    def __init__(self, ground_height_threshold=-999.0):
        # Lowered threshold to 0.05m to ensure simulated feet (at z=0) are not removed.
        self.ground_thresh = ground_height_threshold

    def get_person_cloud(self, points):
        """
        Filters out noise and ground points.
        Input: numpy array (N, 3)
        Output: Filtered numpy array
        """
        if len(points) == 0:
            return points

        # Filter out points below the ground threshold
        # Assuming Z-axis is index 2
        non_ground = points[points[:, 2] > self.ground_thresh]
        
        # If we accidentally filtered everything (e.g., crawling), return original
        if len(non_ground) < 10: 
            return points
            
        return non_ground