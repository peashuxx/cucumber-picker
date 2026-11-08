import pyrealsense2 as rs
import numpy as np
import cv2

def setup_camera():
    pipeline = rs.pipeline()
    config = rs.config()

    # 彩色流与深度流使用相同分辨率。
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # 将深度重投影到彩色图坐标。
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, profile

def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        return None, None, None

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image, aligned_depth_frame

if __name__ == "__main__":
    print("Initializing D435i Camera...")
    pipeline, align, profile = setup_camera()
    print("Camera initialized successfully.")

    try:
        while True:
            color_image, depth_image, _ = get_aligned_frames(pipeline, align)
            if color_image is not None:
                # 伪彩色仅用于显示，不参与深度计算。
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                images = np.hstack((color_image, depth_colormap))

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
