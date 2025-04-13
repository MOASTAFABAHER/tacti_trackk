import numpy as np
def interpolate_ball_positions(ball_positions) :
    positions=np.array([pos if pos is not None else [np.nan, np.nan, np.nan, np.nan] for pos in ball_positions])
    # find frames where ball was detected 
    valid_mask=~np.isnan(positions[:,0])
    
    #interpolate missing positions 
    x = np.arange(len(positions))  # Array that contains the index of each frame
    interpolated = np.zeros_like(positions)    
# Interpolate each coordinate (x1,y1,x2,y2)
    for i in range(4): ## 0 1 2 3 
        if np.sum(valid_mask) > 1:  # Need at least 2 points to interpolate
            interpolated[:, i] = np.interp(x, x[valid_mask], positions[valid_mask, i])
        else:  
            interpolated[:, i] = positions[:, i]
    
    # Fill remaining NaN values with last known position
    for i in range(4):
        valid_idx = np.where(~np.isnan(interpolated[:, i]))[0]
        if len(valid_idx) > 0:
            last_valid = 0
            for j in range(len(interpolated)):
                if np.isnan(interpolated[j, i]):
                    interpolated[j, i] = interpolated[last_valid, i]
                else:
                    last_valid = j
    
    return [pos.tolist() if not np.isnan(pos[0]) else None for pos in interpolated]
        