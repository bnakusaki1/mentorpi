"""QR-code tracking and preview annotation."""

import math

import cv2


class QRTracker:
    """Track the largest QR code, estimate its bearing/range, and draw a box.

    LiDAR range is preferred when supplied. Otherwise, range is estimated
    from the known physical width of the printed QR code.
    """

    def __init__(self, camera_hfov_deg=90.0, qr_size_m=0.15, lidar_reader=None):
        if not 0 < camera_hfov_deg < 180:
            raise ValueError("camera_hfov_deg must be between 0 and 180")
        if qr_size_m <= 0:
            raise ValueError("qr_size_m must be greater than zero")
        self.detector = cv2.QRCodeDetector()
        self.hfov_rad = math.radians(camera_hfov_deg)
        self.qr_size_m = qr_size_m
        self.lidar = lidar_reader

    def _quadrilaterals(self, frame):
        multi = getattr(self.detector, "detectAndDecodeMulti", None)
        if multi is not None:
            try:
                result = multi(frame)
                if len(result) >= 3 and result[2] is not None and len(result[2]):
                    return result[2]
            except (cv2.error, TypeError):
                pass

        try:
            _payload, points, _straight = self.detector.detectAndDecode(frame)
        except (cv2.error, TypeError):
            points = None
        if points is not None:
            return points

        # Geometry can sometimes be found even when OpenCV cannot decode the text.
        try:
            found, points = self.detector.detect(frame)
            return points if found else None
        except (cv2.error, AttributeError, TypeError):
            return None

    @staticmethod
    def _empty_result():
        return {
            "visible": False,
            "bearing_rad": None,
            "distance_m": None,
            "distance_source": None,
            "bbox": None,
            "corners": None,
        }

    def process_frame(self, original_frame):
        """Return target data and an annotated copy; never alters input frame."""
        annotated = original_frame.copy()
        h, w = original_frame.shape[:2]
        quads = self._quadrilaterals(original_frame)
        if quads is None:
            return self._empty_result(), annotated

        candidates = []
        for quad in quads:
            points = quad.reshape(-1, 2).astype(float)
            if len(points) < 4:
                continue
            side_px = sum(
                math.dist(points[i], points[(i + 1) % 4]) for i in range(4)
            ) / 4.0
            if side_px <= 0:
                continue
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            candidates.append((side_px, points, x0, y0, x1, y1))

        if not candidates:
            return self._empty_result(), annotated

        side_px, points, x0, y0, x1, y1 = max(candidates, key=lambda item: item[0])
        focal_px = w / (2.0 * math.tan(self.hfov_rad / 2.0))
        center_x = float(points[:, 0].mean())
        bearing_rad = math.atan((center_x - w / 2.0) / focal_px)

        distance_m = None
        distance_source = None
        if self.lidar is not None:
            try:
                distance_m = self.lidar.get_distance_at(bearing_rad)
            except Exception:
                distance_m = None
            if distance_m is not None:
                distance_source = "lidar"
        if distance_m is None:
            distance_m = self.qr_size_m * focal_px / side_px
            distance_source = "qr_size"

        corners = points.astype("int32").reshape((-1, 1, 2))
        cv2.polylines(annotated, [corners], True, (40, 220, 100), 3)
        x, y = int(x0), max(24, int(y0) - 10)
        cv2.putText(
            annotated,
            f"QR {distance_m:.2f}m  {math.degrees(bearing_rad):+.0f}deg",
            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 100), 2,
            cv2.LINE_AA,
        )

        return {
            "visible": True,
            "bearing_rad": bearing_rad,
            "distance_m": distance_m,
            "distance_source": distance_source,
            "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            "corners": points,
        }, annotated
