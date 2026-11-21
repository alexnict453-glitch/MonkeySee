import cv2
if hasattr(cv2, "setNumThreads"):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

import pipeline.ntables
from pipeline.visionmanager import VisionManager, pin_current_thread_little_cores
import argparse
import platform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dev", action="store_true", help="Use development configuration")
    args = parser.parse_args()

    config_file = args.config
    if config_file is None:
        if args.dev or platform.system() in ["Windows", "Darwin"]:
            config_file = "config_dev.json"
        else:
            config_file = "config.json"

    print(f"Using config: {config_file}")
    vision_manager = VisionManager(config_file)
    pin_current_thread_little_cores()
    import pipeline.htmlserver
    servers = []
    camera_ids = list(vision_manager.get_all_pipelines().keys())
    for camera_id in camera_ids:
        server = pipeline.htmlserver.DashboardServer(vision_manager, camera_id)
        servers.append(server)

    if 0 not in camera_ids:
        servers.append(pipeline.htmlserver.CameraIndexServer(vision_manager))
        print("Camera hub available at port 5800")
    else:
        print("Camera 0 occupies port 5800; open each camera dashboard directly")

    try:
        vision_manager.execute()
    except KeyboardInterrupt:
        vision_manager.stop_all()