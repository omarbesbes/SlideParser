import os
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

def box_iou(a, b):
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)
	inter_w = max(0, inter_x2 - inter_x1)
	inter_h = max(0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h
	a_area = (ax2 - ax1) * (ay2 - ay1)
	b_area = (bx2 - bx1) * (by2 - by1)
	union_area = a_area + b_area - inter_area
	return inter_area / union_area if union_area > 0 else 0

def custom_merge_boxes(boxes, class_ids, iou_thresh=0.5):
	"""
	boxes: Nx4 array-like (xyxy)
	class_ids: N array-like
	Returns merged_boxes as np.ndarray (just geometry, no class)
	"""
	boxes = np.array(boxes)
	class_ids = np.array(class_ids)
	N = len(boxes)
	used = np.zeros(N, dtype=bool)
	merged = []
	for i in range(N):
		if used[i]:
			continue
		group = [boxes[i]]
		used[i] = True
		for j in range(i+1, N):
			if used[j]:
				continue
			# (a) If both are (6,6) or (8,8), merging this pair only if big overlap
			if class_ids[i] == class_ids[j] and class_ids[i] in (6,8) and box_iou(boxes[i], boxes[j]) > 0.8:
				group.append(boxes[j])
				used[j] = True
				continue
			# (b) Otherwise, merge if IOU high
			if box_iou(boxes[i], boxes[j]) > iou_thresh:
				group.append(boxes[j])
				used[j] = True
		group = np.array(group)
		x1 = np.min(group[:,0])
		y1 = np.min(group[:,1])
		x2 = np.max(group[:,2])
		y2 = np.max(group[:,3])
		merged.append([x1, y1, x2, y2])
	return np.array(merged)

# Set paths and model
input_dir = "./tmp2/"
output_dir = "./tmp2_annotated2/"
os.makedirs(output_dir, exist_ok=True)        # Ensure output dir exists

model_path = "./yolov11x_best.pt" # Fine tuned yolov11
model = YOLO(model_path)

# Define allowed image extensions (add more if needed)
allowed_exts = {'.png', '.jpg', '.jpeg', '.bmp'}

# Define custom color palette for each class once
class_colors = [
    sv.Color(255, 0, 0),      # Red for "Caption"
    sv.Color(0, 255, 0),      # Green for "Footnote"
    sv.Color(0, 0, 255),      # Blue for "Formula"
    sv.Color(255, 255, 0),    # Yellow for "List-item"
    sv.Color(255, 0, 255),    # Magenta for "Page-footer"
    sv.Color(0, 255, 255),    # Cyan for "Page-header"
    sv.Color(128, 0, 128),    # Purple for "Picture"
    sv.Color(128, 128, 0),    # Olive for "Section-header"
    sv.Color(128, 128, 128),  # Gray for "Table"
    sv.Color(0, 128, 128),    # Teal for "Text"
    sv.Color(128, 0, 0)       # Maroon for "Title"
]

box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette(class_colors),
    thickness=5
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette(class_colors),
    text_color=sv.Color(255, 255, 255),
    text_scale=2
)

def annotate_folder():
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_exts:
            continue
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read and run model
        image = cv2.imread(input_path)
        results = model(input_path, conf=0.2, iou=0.8)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Each detection has: xyxy (boxes), confidences, class_id, tracker_id etc.
        # Let's merge per-class, preserving original structure (and confidence/class label of first in group).
        
        merged_boxes = custom_merge_boxes(
            boxes=detections.xyxy,
            class_ids=detections.class_id,
            iou_thresh=0.5    # Or your preferred threshold
        )
        
        # Provide dummy class_id so the annotator works
        dummy_class_ids = np.zeros(len(merged_boxes), dtype=int)
        dummy_conf = np.ones(len(merged_boxes))
        merged_detections = sv.Detections(
            xyxy=merged_boxes,
            confidence=dummy_conf,
            class_id=dummy_class_ids
        )
        
        # Annotate bounding boxes and labels
        annotated_image = box_annotator.annotate(
            scene=image,
            detections=merged_detections
        )
        
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated {filename} -> {output_path}")

def annotate_slide(image):
    results = model(image, conf=0.2, iou=0.8)[0]
    detections = sv.Detections.from_ultralytics(results)
    # Each detection has: xyxy (boxes), confidences, class_id, tracker_id etc.
    # Let's merge per-class, preserving original structure (and confidence/class label of first in group).
    
    merged_boxes = custom_merge_boxes(
        boxes=detections.xyxy,
        class_ids=detections.class_id,
        iou_thresh=0.5    # Or your preferred threshold
    )
    
    # Provide dummy class_id so the annotator works
    dummy_class_ids = np.zeros(len(merged_boxes), dtype=int)
    dummy_conf = np.ones(len(merged_boxes))
    merged_detections = sv.Detections(
        xyxy=merged_boxes,
        confidence=dummy_conf,
        class_id=dummy_class_ids
    )
    
    # Annotate bounding boxes and labels
    # annotated_image = box_annotator.annotate(
    #     scene=image,
    #     detections=merged_detections
    #)
    
    return merged_detections