import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Check if a coordinate is within a specified range
def is_valid_coordinate(coord, img_width, img_height, xi, xf, yi, yf):
    x, y = coord
    return xi-5 <= x <= xf+5 and yi-5 <= y <= yf+5

# Check if a coordinate is close to a specified corner coordinate
def is_close_to_corner(coord, corner_coord):
    diff_x = abs(coord[0] - corner_coord[0])
    diff_y = abs(coord[1] - corner_coord[1])
    return diff_x <= 45 and diff_y <= 45

# Process video frames, perform YOLOv8 inference, and extract corner coordinates
def display_coordinates_near_corners(video_path, output_file, modelpath):
    # Load the YOLOv8 model
    model = YOLO(modelpath)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_number = 1

    # Open the text file for writing
    with open('output.txt', 'w') as text_file:
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                original_frame = frame.copy()

                # Run YOLOv8 inference on the frame
                results = model.predict(frame, conf=0.87)
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(int)

                        cv2.rectangle(original_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

                        xi = r[0]
                        yi = r[1]
                        xf = r[2]
                        yf = r[3]

                        green_box_coordinates = [(xi, yf), (xi, yi), (xf, yi), (xf, yf)]
                        cv2.polylines(original_frame, [np.array(green_box_coordinates)], True, (0, 255, 0), 2)
                        text_color = (0, 255, 0)
                        font_scale = 0.5
                        font_thickness = 1

                        # Define new positions for the text
                        positions = [
                            (xi, yi - 20),
                            (xi, yf + 20),
                            (xf, yf + 20),
                            (xf, yi - 10)
                        ]

                        for (x, y), position in zip(green_box_coordinates, positions):
                            text = f"{x}, {y}"
                            cv2.putText(original_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_black = np.array([0, 0, 0], dtype=np.uint8)
                        upper_black = np.array([180, 255, 25], dtype=np.uint8)
                        mask = cv2.inRange(hsv, lower_black, upper_black)

                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        detected_coordinates = {i: [] for i in range(1, len(green_box_coordinates) + 1)}

                        for contour in contours:
                            approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
                            corner_coordinates = [tuple(coord[0]) for coord in approx]
                            valid_coordinates = [coord for coord in corner_coordinates if is_valid_coordinate(coord, frame.shape[1], frame.shape[0], xi, xf, yi, yf)]

                            filtered_coordinates = []
                            for coord in valid_coordinates:
                                add_coord = True
                                for f_coord in filtered_coordinates:
                                    if abs(coord[0] - f_coord[0]) <= 1 and abs(coord[1] - f_coord[1]) <= 1:
                                        add_coord = False
                                        break
                                if add_coord:
                                    filtered_coordinates.append(coord)

                            if filtered_coordinates:
                                for coord in filtered_coordinates:
                                    for corner_idx, corner_coord in enumerate(green_box_coordinates, start=1):
                                        if is_valid_coordinate(coord, frame.shape[1], frame.shape[0], xi, xf, yi, yf) and is_close_to_corner(coord, corner_coord):
                                            if not detected_coordinates[corner_idx]:
                                                detected_coordinates[corner_idx] = coord
                                            else:
                                                current_dist = abs(detected_coordinates[corner_idx][0] - corner_coord[0]) + abs(detected_coordinates[corner_idx][1] - corner_coord[1])
                                                new_dist = abs(coord[0] - corner_coord[0]) + abs(coord[1] - corner_coord[1])
                                                if new_dist < current_dist:
                                                    detected_coordinates[corner_idx] = coord

                        # Print to command prompt
                        print(f"Frame {frame_number}")
                        for idx, coord in enumerate(green_box_coordinates, start=1):
                            print(f"Corner coordinate {idx}: {coord[0]}, {coord[1]}")
                        
                        for corner_idx, coord in detected_coordinates.items():
                            print(f"Near corner {corner_idx}: {coord[0]},{coord[1]}" if coord else f"Near corner {corner_idx}: None")

                        # Write to the text file
                        text_file.write(f"Frame {frame_number}\n")
                        for idx, coord in enumerate(green_box_coordinates, start=1):
                            text_file.write(f"Corner coordinate {idx}: {coord[0]}, {coord[1]}\n")
                        
                        for corner_idx, coord in detected_coordinates.items():
                            text_file.write(f"Near corner {corner_idx}: {coord[0]}, {coord[1]}\n" if coord else f"Near corner {corner_idx}: None\n")

                        for corner_idx, coord in detected_coordinates.items():
                            if coord:
                                text_position = (coord[0] + 10, coord[1] + 10)
                                cv2.putText(original_frame, f"{coord[0]}, {coord[1]}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                cv2.circle(original_frame, coord, 3, (0, 255, 255), -1)

                        # Run YOLOv8 tracking on the frame, persisting tracks between frames
                        tracking_results = model.track(frame, conf=0.87, persist=True)
                        
                        # Store the track history
                        track_history = defaultdict(lambda: [])
                        
                        # Get the boxes and track IDs
                        boxestrack = tracking_results[0].boxes.xywh.cpu()
                        track_ids = tracking_results[0].boxes.id.int().cpu().tolist()

                        # Visualize the results on the frame
                        annotated_frame = tracking_results[0].plot()

                        # Plot the tracks
                        for box, track_id in zip(boxestrack, track_ids):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)

                            # Draw the tracking lines
                            points_ = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points_], isClosed=False, color=(230, 230, 230), thickness=10)

                        # Combine the original frame and annotated frame into one display frame
                        combined_frame = cv2.addWeighted(original_frame, 0.7, annotated_frame, 0.3, 0)

                        # Write the combined frame to the output video file
                        out.write(combined_frame)

                        # Display the combined frame
                        cv2.imshow("X-ray Box Tracking Detection", combined_frame)

                        frame_number += 1

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            else:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
display_coordinates_near_corners('output_video.avi', 'outputcomputervision.mp4', 'best2.pt')
