import cv2


class OverlayWidgets:
    @staticmethod
    def draw_bounding_box(image, box, label, color, offset_x=0):
        x1, y1, x2, y2 = map(int, box)
        x1 += offset_x
        x2 += offset_x
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    @staticmethod
    def draw_gauge(image, distance):
        gauge_width = 12
        gauge_height = 100
        gauge_x = image.shape[1] - 15
        gauge_y = image.shape[0] - 120
        cv2.rectangle(image, (gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height), (0, 0, 0), -1)
        fill_height = int((distance / 100) * gauge_height)
        cv2.rectangle(image, (gauge_x, gauge_y + gauge_height - fill_height), (gauge_x + gauge_width, gauge_y + gauge_height), (0, 255, 0), -1)
        cv2.putText(image, str(int(distance)), (gauge_x, gauge_y + gauge_height - fill_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
        return image
