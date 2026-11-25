
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
from update_rover import (update_rover, show_ray_of_cam, show_target_rays, throw_ray_from_cam_to_target, cast_payload_ray, select_detected_enemy, remove_hit_enemy, is_active_enemy)
import cv2
from dashboard import create_dashboard
from yolo_detector import YOLODetector

SCENE_PATH = "scenes/scene.xml"
CAMERA_NAME = "front_cam"
OVERVIEW_CAMERA_NAME = "overview_cam"
ROVER_CAMERA_NAME = "rover_cam"
TARGET_FPS = 60
WINDOW_NAME = "SENTINEL"

INFERENCE_INTERVAL = 0.2
SHOT_DISPLAY_TIME = 0.2
CONFIDENCE_THRESHOLD = 0.75

def main(scene_path=SCENE_PATH):
    model = mujoco.MjModel.from_xml_path(scene_path)
    detector = YOLODetector(confidence=0.1)
    data = mujoco.MjData(model)
    ray_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    enemy_ids = [geom_id for geom_id in range(model.ngeom) if is_active_enemy(model, geom_id)]

    start_time = time.perf_counter()
    last_inference = -INFERENCE_INTERVAL

    def inspect_frame(frame):
        detections = detector.detect(frame)
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            print(item["class_name"], "x =", (x1 + x2) / 2, "y =", (y1 + y2) / 2, "confidence =", item["confidence"])
        return detector.annotate(frame, detections), detections

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 984, 888)
        with ThreadPoolExecutor(max_workers=1) as executor:
            inference = None
            annotated = None
            rays = []
            target_id = None
            shot_time = None
            previous_frame = start_time
            while True:
                frame_start = time.perf_counter()
                t = frame_start - start_time
                dt = frame_start - previous_frame
                previous_frame = frame_start

                aimed_target = target_id
                target = data.geom_xpos[target_id].copy() if target_id is not None else None
                aligned = update_rover(model, data, t, dt, target=target)

                renderer.update_scene(data, camera=CAMERA_NAME)
                rgb = renderer.render()

                if inference is not None and inference.done():
                    annotated, detections = inference.result()
                    ray_data.qpos[:] = inference_qpos
                    mujoco.mj_forward(model, ray_data)
                    hit_ids = []
                    for item in detections:
                        x1, y1, x2, y2 = item["bbox"]

                        if item["confidence"] > CONFIDENCE_THRESHOLD:
                            _, _, hit_id = throw_ray_from_cam_to_target(model, ray_data, CAMERA_NAME, renderer.width, renderer.height, (x1 + x2) / 2, (y1 + y2) / 2)
                            hit_ids.append(hit_id)
                    target_id = select_detected_enemy(model, target_id, hit_ids)
                    inference = None

                if target_id is not None and target_id == aimed_target and aligned:
                    origin, end, hit_id = cast_payload_ray(model, data)
                    remove_hit_enemy(model, hit_id)
                    rays = [(origin, end)]
                    shot_time = t
                    target_id = None
                if shot_time is not None and t - shot_time >= SHOT_DISPLAY_TIME:
                    rays = []
                    shot_time = None

                renderer.update_scene(data, camera=OVERVIEW_CAMERA_NAME)

                show_ray_of_cam(model, data, renderer, CAMERA_NAME)
                show_target_rays(renderer, rays)
                overview = renderer.render()

                renderer.update_scene(data, camera=ROVER_CAMERA_NAME)
                show_target_rays(renderer, rays)
                rover_view = renderer.render()

                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if inference is None and t - last_inference >= INFERENCE_INTERVAL:
                    inference_qpos = data.qpos.copy()
                    inference = executor.submit(inspect_frame, frame.copy())
                    last_inference = t

                remaining = sum(is_active_enemy(model, geom_id) for geom_id in enemy_ids)
                target_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, target_id) if target_id is not None else None)
                dashboard = create_dashboard(
                    frame, cv2.cvtColor(overview, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(rover_view, cv2.COLOR_RGB2BGR),
                    annotated if annotated is not None else frame,
                    t, len(enemy_ids) - remaining, remaining, target_name,
                )
                cv2.imshow(WINDOW_NAME, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    cv2.imwrite("test_scene.png", frame)
                    print("Image enregistrée : test_scene.png")

                if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                elapsed = time.perf_counter() - frame_start
                time.sleep(max(0, 1 / TARGET_FPS - elapsed))

            if inference is not None:
                inference.result()

    finally:
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation du rover SENTINEL")
    parser.add_argument("--scene", default=SCENE_PATH, help="Fichier XML de la scene (defaut : scene.xml)")
    args = parser.parse_args()
    main(args.scene)
