import math
import mujoco
import numpy as np

RAY_COUNT = 15
RAY_LENGTH = 10.0
REMOVED_GEOM_GROUP = 5
RAY_GEOM_GROUPS = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)
PAYLOAD_SPEED = math.radians(50)
PAYLOAD_TOLERANCE = math.radians(0.5)

RADIUS = 5.5
SPEED = 2.0
WHEEL_RADIUS = 0.15
WHEELBASE = 0.60
HEIGHT = 0.32

def set_joint(model, data, name, value):
    joint_id = model.joint(name).id
    data.qpos[model.jnt_qposadr[joint_id]] = value

def orient_camera(model, data, target=(0.0, 0.0)):
    mount = data.body("camera_mount")
    pos = mount.xpos

    dx = target[0] - pos[0]
    dy = target[1] - pos[1]

    target_yaw = math.atan2(dy, dx)

    payload_rotation = data.body("payload_tilt").xmat.reshape(3, 3)
    payload_yaw = math.atan2(payload_rotation[1, 0], payload_rotation[0, 0])

    relative_yaw = math.atan2(math.sin(target_yaw - payload_yaw), math.cos(target_yaw - payload_yaw))

    set_joint(model, data, "camera_yaw", np.clip(relative_yaw, math.radians(-170), math.radians(170)))
    mujoco.mj_forward(model, data)


def show_ray_of_cam(model, data, renderer, camera_name):
    cam_id = model.camera(camera_name).id
    pos = data.cam_xpos[cam_id].copy()
    rotation = data.cam_xmat[cam_id].reshape(3, 3)

    fovy = math.radians(model.cam_fovy[cam_id])
    aspect = renderer.width / renderer.height
    fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)

    forward = -rotation[:, 2]
    yaw = math.atan2(forward[1], forward[0])

    for angle in np.linspace(-fovx / 2, fovx / 2, RAY_COUNT):
        direction = np.array([math.cos(yaw + angle), math.sin(yaw + angle), 0.0])
        end = pos + direction * RAY_LENGTH

        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            break

        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 0.015, pos, end)

        geom.rgba[:] = [0.2, 0.9, 0.5, 1.0]
        renderer.scene.ngeom += 1


def throw_ray_from_cam_to_target(model, data, camera_name, width, height, x_pos_target, y_pos_target):
    cam_id = model.camera(camera_name).id
    origin = data.cam_xpos[cam_id].copy()
    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    focal_length = height / (2 * math.tan(math.radians(model.cam_fovy[cam_id]) / 2))
    direction = rotation @ np.array([(x_pos_target - width / 2) / focal_length, (height / 2 - y_pos_target) / focal_length, -1.0,])
    direction /= np.linalg.norm(direction)
    camera_body = model.cam_bodyid[cam_id]
    body_exclude = camera_body if camera_body != 0 else -1
    return cast_ray(model, data, origin, direction, body_exclude)


def cast_ray(model, data, origin, direction, body_exclude):
    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction, RAY_GEOM_GROUPS, True, body_exclude, geom_id)
    length = min(distance, RAY_LENGTH) if distance >= 0 else RAY_LENGTH
    hit_id = int(geom_id[0]) if 0 <= distance <= RAY_LENGTH else -1

    return origin, origin + direction * length, hit_id


def cast_payload_ray(model, data):
    site = data.site("payload_muzzle")
    origin = site.xpos.copy()
    direction = site.xmat.reshape(3, 3)[:, 0].copy()
    return cast_ray(model, data, origin, direction, model.body("payload_tilt").id)


def is_active_enemy(model, geom_id):
    if (geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE or model.geom_group[geom_id] == REMOVED_GEOM_GROUP):
        return False
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name is not None and name.startswith("enemy")


def select_detected_enemy(model, current_target, detected_ids):
    enemies = [geom_id for geom_id in detected_ids if is_active_enemy(model, geom_id)]
    if current_target in enemies:
        return current_target
    return enemies[0] if enemies else None


def remove_hit_enemy(model, geom_id):
    if not is_active_enemy(model, geom_id):
        return

    model.geom_group[geom_id] = REMOVED_GEOM_GROUP
    model.geom_contype[geom_id] = 0
    model.geom_conaffinity[geom_id] = 0


def show_target_rays(renderer, rays):
    for origin, end in rays:
        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            break
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.array([0.9, 0.2, 0.5, 1.0], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, origin, end)
        renderer.scene.ngeom += 1


def update_orient_laser(model, data, target, dt):
    if target is None:
        return False

    pivot = data.body("payload_tilt").xpos
    rover_rotation = data.body("rover").xmat.reshape(3, 3)
    dx, dy, dz = rover_rotation.T @ (target - pivot)
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))

    aligned = True
    for name, desired in (("payload_yaw", yaw), ("payload_pitch", pitch)):
        joint_id = model.joint(name).id
        lower, upper = model.jnt_range[joint_id]
        goal = np.clip(desired, lower, upper)
        current = data.qpos[model.jnt_qposadr[joint_id]]
        angle = current + np.clip(goal - current, -PAYLOAD_SPEED * dt, PAYLOAD_SPEED * dt)
        set_joint(model, data, name, angle)
        aligned = aligned and abs(desired - angle) <= PAYLOAD_TOLERANCE

    mujoco.mj_forward(model, data)
    return aligned


def update_rover(model, data, t, dt, target=None):
    theta = SPEED * t / RADIUS

    x = RADIUS * math.cos(theta)
    y = RADIUS * math.sin(theta)

    yaw = theta + math.pi / 2

    body_id = model.body("rover").id
    joint_id = model.body_jntadr[body_id]
    adr = model.jnt_qposadr[joint_id]

    data.qpos[adr:adr + 3] = [x, y, HEIGHT]
    data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    wheel_angle = SPEED * t / WHEEL_RADIUS

    for name in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        set_joint(model, data, name, wheel_angle)

    steering = math.atan(WHEELBASE / RADIUS)

    set_joint(model, data, "steer_fl", steering)
    set_joint(model, data, "steer_fr", steering)

    data.ctrl[:] = 0
    data.qvel[:] = 0

    mujoco.mj_forward(model, data)
    aligned = update_orient_laser(model, data, target, dt)
    orient_camera(model, data, target=(0.0, 0.0))
    return aligned
