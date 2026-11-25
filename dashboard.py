import cv2
import numpy as np


def create_dashboard(front, overview, rover, annotated, elapsed, reached, remaining, target_name):
    tile_width, tile_height = 480, 360
    gap, header_height, title_height = 8, 84, 30
    width = 2 * tile_width + 3 * gap
    height = header_height + 2 * (tile_height + title_height) + 3 * gap
    dashboard = np.full((height, width, 3), (25, 29, 34), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    hours, seconds = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(seconds, 60)
    cv2.putText(dashboard, "SENTINEL", (gap, 27), font, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(dashboard, f"Time : {hours:02}:{minutes:02}:{seconds:02}", (190, 27), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(dashboard, f"Reached : {reached}", (420, 27), font, 0.6, (110, 220, 110), 1, cv2.LINE_AA)
    cv2.putText(dashboard, f"Remaining : {remaining}", (730, 27), font, 0.6, (110, 190, 255), 1, cv2.LINE_AA)
    cv2.putText(dashboard, f"Target : {target_name if target_name is not None else 'searching'}", (gap, 62), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(dashboard, "Q : Quit", (690, 62), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    views = (("FRONT CAMERA", front), ("OVERVIEW", overview), ("ROVER VIEW", rover), ("DETECTIONS", annotated))
    
    for index, (title, frame) in enumerate(views):
        x = gap + (index % 2) * (tile_width + gap)
        y = header_height + gap + (index // 2) * (tile_height + title_height + gap)
        dashboard[y:y + title_height, x:x + tile_width] = (48, 54, 60)
        cv2.putText(dashboard, title, (x + 8, y + 21), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        dashboard[y + title_height:y + title_height + tile_height, x:x + tile_width] = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

    return dashboard
