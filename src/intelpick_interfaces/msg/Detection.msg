# One object seen by the camera.
string label          # sort key: color name ("red") or YOLO class, optionally "red_cube"
float32 confidence    # 0..1

# Image position (pixels, centroid of the object)
float32 u
float32 v
float32 angle         # rotation of the min-area rectangle, radians
float32 area_px

# Position on the table in the arm base frame (metres, +X forward, +Y left).
# NaN when no camera->table calibration is loaded.
float32 x
float32 y
