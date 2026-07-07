from ultralytics import YOLO


class PersonDetector:
    """Wraps a YOLO model and only returns boxes for the 'person' class."""

    PERSON_CLASS_ID = 0  # COCO person class index

    def __init__(self, model_name, conf_threshold, imgsz):
        print("Loading model:", model_name)
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def detect(self, frame):
        """Run inference on a frame and return a list of person bounding boxes (xyxy)."""
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        boxes = []
        if len(results) > 0:
            r = results[0]
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    boxes.append(xyxy)
        return boxes
