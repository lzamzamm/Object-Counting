import cv2
import numpy as np
from ultralytics import YOLO


def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


input_video_path = "uin.mp4"
output_video_path = "output_video.mp4"

cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "Error membuka file video"

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(
    output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO("yolov8s.pt")

roi = [(600, 700), (1100, 700), (1100, 750), (600, 750)]

counted_ids = {
    "motorcycle": set(),
    "car": set()
}

class_colors = {
    "motorcycle": (255, 0, 0),
    "car": (0, 255, 0)
}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box[:4])
            track_id = int(track_id)
            class_id = int(boxes.cls[boxes.id.tolist().index(track_id)])
            class_name = model.names[class_id]

            if class_name in ["motorcycle", "car"]:
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

                if point_in_polygon(center_point, roi) and track_id not in counted_ids[class_name]:
                    counted_ids[class_name].add(track_id)

                color = class_colors[class_name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name}-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.polylines(frame, [np.array(roi, np.int32)], True,
                  (0, 0, 255), 2)

    motorcycle_count = len(counted_ids["motorcycle"])
    car_count = len(counted_ids["car"])

    cv2.putText(frame, f"Motorcycles: {motorcycle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors["motorcycle"], 2)
    cv2.putText(frame, f"Cars: {car_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors["car"], 2)

    # total_count = motorcycle_count + car_count
    # cv2.putText(frame, f"Total: {total_count}", (10, 110),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Merah untuk total
    cv2.putText(frame, f"Luthfantry Zamzam Muhammad (21106050024)", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Merah untuk total

    cv2.imshow("Object Counting", frame)

    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
