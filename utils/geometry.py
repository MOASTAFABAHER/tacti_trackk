def get_bbox_width(bbox):
    #that is returns the width of bbox by subtracting x1 and x2 
    return bbox[2] - bbox[0]

def get_center_of_bbox(bbox):
    # Dived bbox to xyxy
    x1,y1,x2,y2=bbox
    #returns the center of bbox but as an integer value 
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
    

