import cv2
import numpy as np
def get_grass_color(frame):
    # Select the bottom half of the frame since it likely contains the grass
    grass_region=frame[frame.shape[0]//2:,:]    
    
    #convert the grass region to hsv color 
    grass_hsv=cv2.cvtColor(grass_region,cv2.COLOR_BGR2HSV)
    
    #calculate the mean color of the grass region
    grass_mean_color=np.mean(grass_hsv,axis=(0,1))
    
    #return the mean color of the grass region
    return grass_mean_color

def get_jersey_color(player_roi,grass_hsv):
    
    # Convert the player's region of interest (ROI) from BGR to HSV color space
    hsv=cv2.cvtColor(player_roi,cv2.COLOR_BGR2HSV)
    
    # Define the range for the grass color based on the Hue of the grass (±10 degrees) 
    # and fixed values for Saturation and Value (50 to 255)
    lower_grass=np.array([grass_hsv[0]-10,50,50])
    upper_grass=np.array([grass_hsv[0]+10,255,255])
    
    # Create a mask that includes only pixels within the defined grass color range
    mask=cv2.inRange(hsv,lower_grass,upper_grass)
    
    # Invert the mask so that non-grass pixels are included (this will exclude the grass color)
    mask=cv2.bitwise_not(mask)
    
    # Extract the pixels that are not in the grass color range (i.e., the jersey color)
    pixels=hsv[mask==255]
    
    # If there are valid pixels, compute the average color of the jersey
    # Otherwise, return black (no color found)
    return np.mean(pixels, axis=0) if len(pixels) > 0 else np.array([0, 0, 0])

    
