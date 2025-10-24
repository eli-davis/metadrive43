# B"H

BOOL_RENDER = True

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_bridge.py

import os
import time
import sys
import math
import numpy as np

from termcolor import cprint as print_in_color

sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2/metadrive43/openpilot99_setup")
import get_char

#from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod

#from openpilot.tools.sim.bridge.common import SimulatorBridge

#sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2/metadrive43/openpilot99_setup")
#from metadrive_world import MetaDriveWorld

W, H = 1928, 1208

# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_process.py

from collections import namedtuple
from panda3d.core import Vec3
#from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

#from openpilot.common.realtime import Ratekeeper

vec3 = namedtuple("vec3", ["x", "y", "z"])

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0,0)

metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle"])

def apply_metadrive_patches(bool_continuous_loop=True):

    # ___________________________________________ #
    # ___________________________________________ #

    # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
    def add_image_sensor_patched(self, name: str, cls, args):
        if self.global_config["image_on_cuda"]:
            sensor = cls(*args, self, cuda=True)
        else:
            sensor = cls(*args, self, cuda=False)
        assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
        self.sensors[name] = sensor

    EngineCore.add_image_sensor = add_image_sensor_patched

    # ___________________________________________ #
    # ___________________________________________ #

    # we aren't going to use the built-in observation stack, so disable it to save time
    def observe_patched(self, *args, **kwargs):
        return self.state

    ImageObservation.observe = observe_patched

    # ___________________________________________ #
    # ___________________________________________ #

    # disable destination, we want to loop continuously
    def arrive_destination_patch(self, *args, **kwargs):
        return False

    if bool_continuous_loop:
        MetaDriveEnv._is_arrive_destination = arrive_destination_patch

    # ___________________________________________ #
    # ___________________________________________ #


# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #


def straight_block(length):
    return {
        "id": "S",
        "pre_block_socket_index": 0,
        "length": length
    }

def curve_block(length, angle=45, direction=0):
    return {
        "id": "C",
        "pre_block_socket_index": 0,
        "length": length,
        "radius": length,
        "angle": angle,
        "dir": direction
    }

def create_map(track_size=60):
    curve_len = track_size * 2
    return dict(
        type=MapGenerateMethod.PG_MAP_FILE,
        lane_num=2,
        lane_width=4.5,
        config=[
            None,
            straight_block(track_size),
            curve_block(curve_len, 90),
            straight_block(track_size),
            curve_block(curve_len, 90),
            straight_block(track_size),
            curve_block(curve_len, 90),
            straight_block(track_size),
            curve_block(curve_len, 90),
        ]
    )

# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_common.py

from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import Texture, GraphicsOutput


class CopyRamRGBCamera(RGBCamera):
    """Camera which copies its content into RAM during the render process, for faster image grabbing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_texture = Texture()
        self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

    def get_rgb_array_cpu(self):
        origin_img = self.cpu_texture
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
        img = img[:,:,:3] # RGBA to RGB
        # img = np.swapaxes(img, 1, 0)
        img = img[::-1] # Flip on vertical axis
        return img


class RGBCameraWide(CopyRamRGBCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lens = self.get_lens()
        lens.setFov(120)
        lens.setNear(0.1)

class RGBCameraRoad(CopyRamRGBCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lens = self.get_lens()
        lens.setFov(40)
        lens.setNear(0.1)

# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_world.py

import ctypes
import functools


class MetadriveGym():

    def __init__(self):

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # setup cameras

        dual_camera=True

        self.main_road_image = np.zeros((H, W, 3), dtype=np.uint8)
        self.wide_road_image = np.zeros((H, W, 3), dtype=np.uint8)

        #self.main_camera_array = Array(ctypes.c_uint8, W*H*3)
        #self.main_road_image = np.frombuffer(self.main_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

        #self.wide_camera_array = Array(ctypes.c_uint8, W*H*3)
        #self.wide_road_image = np.frombuffer(self.wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

        sensors = dict()
        sensors["rgb_road"] = (RGBCameraRoad, W, H)
        sensors["rgb_wide"] = (RGBCameraWide, W, H)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # setup config

        config = dict()

        config['use_render'] = BOOL_RENDER

        config['vehicle_config'] = dict()
        config['vehicle_config']['enable_reverse'] = False
        #config['vehicle_config']['render_vehicle'] = BOOL_RENDER
        config['vehicle_config']['image_source'] = "rgb_road"

        config['sensors'] = sensors

        config['image_on_cuda'] = _cuda_enable
        config['image_observation'] = True
        config['interface_panel'] = []
        config['out_of_route_done'] = False
        config['on_continuous_line_done'] = False

        config['crash_vehicle_done'] = False
        config['crash_object_done']  = False

        # traffic is incredibly expensive
        config['traffic_density'] = 0.0

        config['map_config'] = create_map()
        config['decision_repeat'] = 1

        # 50ms
        config['physics_world_step_size'] = 0.05

        config['preload_models'] = False
        config['show_logo'] = False

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # start metadrive

        apply_metadrive_patches()

        self.env = MetaDriveEnv(config)

        self.lane_idx_prev = self.reset()

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        print(f"simulator delta_t in env.step() = {self.env.engine.global_config['physics_world_step_size']}")

        for i in range(0, 40):
            _, _, terminated, _, _ = self.env.step([0.0, 0.0])

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def get_current_lane_info(self):
        _, lane_info, on_lane = self.env.vehicle.navigation._get_current_lane(self.env.vehicle)
        lane_idx = lane_info[2] if lane_info is not None else None
        return lane_idx, on_lane

    def reset(self):
        self.env.reset()
        self.env.vehicle.config["max_speed_km_h"] = 1000
        lane_idx_prev, _ = self.get_current_lane_info()
        return lane_idx_prev

    def get_cam_as_rgb(self, camera_str):
        cam = self.env.engine.sensors[camera_str]
        cam.get_cam().reparentTo(self.env.vehicle.origin)
        cam.get_cam().setPos(C3_POSITION)
        cam.get_cam().setHpr(C3_HPR)
        img = cam.perceive(to_float=False)
        if not isinstance(img, np.ndarray):
            # convert cupy array to numpy
            img = img.get()
        return img


    # just one step? or N frames

    def step(self, steer_angle, throttle_out, brake_out, bool_reset):

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        vc = [0.0, 0.0]

        steer_ratio = 15
        steer_metadrive = steer_angle * 1 / (self.env.vehicle.MAX_STEERING * steer_ratio)
        steer_metadrive = np.clip(steer_metadrive, -1, 1)

        vc[0] = steer_metadrive

        if throttle_out:
            vc[1] = throttle_out
        else:
            vc[1] = -brake_out

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if bool_reset:
            self.lane_idx_prev = self.reset()

        _, _, terminated, _, _ = self.env.step(vc)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        lane_idx_curr, on_lane = self.get_current_lane_info()
        bool_out_of_lane = lane_idx_curr != self.lane_idx_prev or not on_lane
        self.lane_idx_prev = lane_idx_curr

        # get_cam_as_rgb() returns a NEW array
        # main_road_image[...] = get_cam_as_rgb --> copies that data INTO the pre-allocated buffer
        self.main_road_image[...] = self.get_cam_as_rgb("rgb_road")
        self.wide_road_image[...] = self.get_cam_as_rgb("rgb_wide")

        vehicle_state = metadrive_vehicle_state(
            velocity=vec3(x=float(self.env.vehicle.velocity[0]), y=float(self.env.vehicle.velocity[1]), z=0),
            position=self.env.vehicle.position,
            bearing=float(math.degrees(self.env.vehicle.heading_theta)),
            steering_angle=self.env.vehicle.steering * self.env.vehicle.MAX_STEERING
        )

        return vehicle_state, self.main_road_image, self.wide_road_image, bool_out_of_lane


# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #




def run_metadrive():

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    metadrive_gym = MetadriveGym()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    bool_manual = True

    while True:
        # Read manual controls
        action = get_char.get_char()

        throttle_manual = 0.0
        brake_manual = 0.0
        steer_manual = 0.0

        # based on https://github.com/commaai/openpilot/blob/v0.9.9/tools/sim/lib/keyboard_ctrl.py
        if action == "UP":
            throttle_manual = 1.0

        elif action == "DOWN":
            brake_manual = 1.0

        elif action == "LEFT":
            steer_manual = -0.15

        elif action == "RIGHT":
            steer_manual = 0.15

        else:
            pass

        '''
        # CarState from these values?
        # see simulated_car.py

        self.simulator_state.user_brake = brake_manual
        self.simulator_state.user_gas = throttle_manual
        self.simulator_state.user_torque = steer_manual * -10000
        '''

        if bool_manual:
            steer_out = steer_manual * -40
            throttle_out =  throttle_manual
            brake_out = brake_manual

        # todo: auto
        #else:
        #    throttle_out = np.clip(self.simulated_car.sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        #    brake_out = np.clip(-self.simulated_car.sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        #    steer_out = self.simulated_car.sm['carControl'].actuators.steeringAngleDeg

        vehicle_state, main_road_image, wide_road_image, bool_out_of_lane = metadrive_gym.step(steer_out, throttle_out, brake_out, bool_reset=False)

        print("STEP")

        print(f"vehicle_state={vehicle_state}")
        print(f"main_road_image shape={main_road_image.shape} dtype={main_road_image.dtype} avg={np.mean(main_road_image)}")
        print(f"wide_road_image shape={wide_road_image.shape} dtype={wide_road_image.dtype} avg={np.mean(wide_road_image)}")

        print_in_color(f"bool_out_of_lane={bool_out_of_lane}", "yellow")


if __name__ == "__main__":
    run_metadrive()
