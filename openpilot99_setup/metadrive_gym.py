# B"H


BOOL_RENDER = True

BOOL_WRITE_SHARED_MEM = True

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_bridge.py

import os
import time
import sys
import math
import numpy as np

from termcolor import cprint as print_in_color

import cv2

#sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2/metadrive43/openpilot99_setup")
#import get_char
import keyboard

# __________________________________________________________________________ #
# __________________________________________________________________________ #

sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2")
import camera_db

camera_db.setup_metadrive()

# __________________________________________________________________________ #
# __________________________________________________________________________ #

sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2/shared_mem")
from camera_shared_memory_array import CameraViewerSharedMemoryArray_RGBA
from model_output_shared_memory_array import ModelOutputSharedMemoryArray

# __________________________________________________________________________ #
# __________________________________________________________________________ #


sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2/user_interface")
from replay_drive import EasyInference
from ui_overlay import UI_Overlay
import ui_helpers

# __________________________________________________________________________ #
# __________________________________________________________________________ #

# Import for the ground-truth path object
sys.path.insert(0, "/home/deepview/SSD/pathfinder/software2/openpilot99")
import run_planner

# __________________________________________________________________________ #
# __________________________________________________________________________ #

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.lane.circular_lane import CircularLane

W, H = 1928, 1208

# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_process.py

from collections import namedtuple
from panda3d.core import Vec3

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation


vec3 = namedtuple("vec3", ["x", "y", "z"])

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0,0)

metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle", "heading_theta"])

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
            straight_block(track_size*4),

            curve_block(curve_len, 90),

            straight_block(track_size),

            curve_block(curve_len, 90),

            straight_block(track_size),
            straight_block(track_size),
            straight_block(track_size),
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

#
# actually bgr!!!!
#
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

        # img is upside down, so flip it
        img = img[::-1]

        # panda3d stores BGR
        rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #cv2.imwrite("img0.bmp", rgb)
        return rgb


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

        # (trying to get the virtual camera not to move)
        if BOOL_RENDER:
            self.env.engine.disable_mouse() # Disables mouse control
            #self.env.engine.camera.reparentTo(self.env.vehicle.origin)
            #self.env.engine.camera.setPos(C3_POSITION)
            #self.env.engine.camera.setHpr(C3_HPR)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        print(f"simulator delta_t in env.step() = {self.env.engine.global_config['physics_world_step_size']}")

        for i in range(0, 40):
            _, _, terminated, _, info = self.env.step([0.0, 0.0])

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

        #steer_ratio = 8
        steer_metadrive = steer_angle * 1 / (self.env.vehicle.MAX_STEERING ) # * steer_ratio)
        steer_metadrive = np.clip(steer_metadrive, -1, 1)

        vc[0] = steer_metadrive

        if throttle_out:
            vc[1] = throttle_out
        else:
            vc[1] = -brake_out

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if bool_reset:
            self.lane_idx_prev = self.reset()

        _, _, terminated, _, info = self.env.step(vc)

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
            steering_angle=self.env.vehicle.steering * self.env.vehicle.MAX_STEERING,
            heading_theta=float(self.env.vehicle.heading_theta)
        )

        return vehicle_state, self.main_road_image, self.wide_road_image, bool_out_of_lane


# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #
# __________________________________________________________________________ #



'''
def run_metadrive():

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    metadrive_gym = MetadriveGym()


    shared_mem_camera_rgba = CameraViewerSharedMemoryArray_RGBA(bool_create=True, service_name="metadrive_gym")
    shared_mem_model = ModelOutputSharedMemoryArray(bool_create=True, service_name="metadrive_gym")

    inference_helper = EasyInference()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #bool_manual = False
    frame_i = 0

    # === ACCELERATION ===
    # Initialize variables to calculate acceleration (aEgo)
    prev_vEgo = 0.0
    delta_t = 0.05  # This must match 'physics_world_step_size' from your config
    # ====================

    CarControl_output = None

    while True:

        throttle_manual = 0.0
        brake_manual = 0.0
        steer_manual = 0.0

        # --- REPLACED LOGIC ---
        # Change from if/elif to separate 'if' statements

        # (1) this allows combined moves such as up+left
        # (2) also now running continuous vs awaiting key presses

        if keyboard.is_pressed("up"):
            throttle_manual = 1.0

        if keyboard.is_pressed("down"):
            brake_manual = 1.0

        if keyboard.is_pressed("left"):
            # replace -0.15 * -40
            steer_manual = 6

        if keyboard.is_pressed("right"):
            # replace 0.15 * -40
            steer_manual = -6

        # Add a way to quit
        if keyboard.is_pressed("x"):
            print("exitting...")
            break

        # --- END REPLACED LOGIC ---

        # Read manual controls
        #action = get_char.get_char()
        #
        # based on https://github.com/commaai/openpilot/blob/v0.9.9/tools/sim/lib/keyboard_ctrl.py
        #if action == "UP":
        #    throttle_manual = 1.0
        #
        #elif action == "DOWN":
        #    brake_manual = 1.0
        #
        #elif action == "LEFT":
        #    steer_manual = -0.15
        #
        #elif action == "RIGHT":
        #    steer_manual = 0.15
        #
        #else:
        #    pass

        # CarState from these values?
        # see simulated_car.py

        #self.simulator_state.user_brake = brake_manual
        #self.simulator_state.user_gas = throttle_manual
        #self.simulator_state.user_torque = steer_manual * -10000

        # _______________________________________________________
        # _______________________________________________________

        #if bool_manual:
        steer_out = steer_manual
        throttle_out =  throttle_manual
        brake_out = brake_manual
        #else:
        if True:
            # Use the CarControl_output from the PREVIOUS frame
            if CarControl_output is None:
                # First frame, do nothing
                #steer_out = 0.0
                #throttle_out = 0.0
                #brake_out = 0.0
                pass
            else:
                print(f"OpenPilot accel command: {CarControl_output.actuators.accel:.2f} m/s²")

                # Get actuators from the previous frame's OpenPilot calculation
                #steer_out = CarControl_output.actuators.steeringAngleDeg
                accel = CarControl_output.actuators.accel

                # Convert OpenPilot accel to throttle/brake
                #if throttle_manual:
                #throttle_out = throttle_manual
                #else:
                #    throttle_out = np.clip(accel / 1.6, 0.0, 1.0)
                #brake_out = np.clip(-accel / 4.0, 0.0, 1.0)

        vehicle_state, main_road_image, wide_road_image, bool_out_of_lane = metadrive_gym.step(steer_out, throttle_out, brake_out, bool_reset=False)

        # _______________________________________________________
        # _______________________________________________________

        #cv2.imwrite("img1.bmp", main_road_image)

        #print("STEP")

        #print(f"vehicle_state={vehicle_state}")
        #print(f"main_road_image shape={main_road_image.shape} dtype={main_road_image.dtype} avg={np.mean(main_road_image)}")
        #print(f"wide_road_image shape={wide_road_image.shape} dtype={wide_road_image.dtype} avg={np.mean(wide_road_image)}")

        #print_in_color(f"bool_out_of_lane={bool_out_of_lane}", "yellow")

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        vEgo = math.sqrt(vehicle_state.velocity.x**2 + vehicle_state.velocity.y**2)

        # === START FIX ===
        # Calculate current acceleration
        aEgo = (vEgo - prev_vEgo) / delta_t
        prev_vEgo = vEgo
        # === END FIX ===

        # set cruise speed to 20 m/s (45 mph)
        car_state_dict = { "vEgo": vEgo,
                           "aEgo": aEgo,
                           "steeringAngleDeg": vehicle_state.steering_angle,
                           "vCruise": 100.0, #km/h?
                           "standstill": bool(vEgo < 1.0),  # <--- ADD THIS LINE (True if speed is near zero)
                           "brakePressed": bool(brake_manual >= 0.1),  # <--- ADD THIS LINE
                           "cruiseState": {"enabled": True, "standstill": bool(vEgo < 1.0), },  # <--- ADD THIS LINE
        }

        # --- Create a copy for overlay drawing ---
        #display_image = main_road_image.copy() # Overlays are drawn on this

        # 4. Call EasyInference OpenPilot Cameras Method
        inf_start = time.time()
        model_output_array = None # Default

        CarControl_output, model_output_array = inference_helper.run_inference_openpilot_cameras(main_road_image, wide_road_image, frame_i, vEgo, car_state_dict)

        # ++++++++++++++++++++++++++++++++++++++++++++++
        # Check if the result is None (which happens during buffer warmup)
        if CarControl_output is None:
            print_in_color(f"STEP {frame_i}: Skipping frame, model buffers are warming up.", "cyan")
            frame_i += 1
            continue
        # ++++++++++++++++++++++++++++++++++++++++++++++

        #cv2.imwrite("img2.bmp", main_road_image)

        # Convert final display image (with overlays) to RGBA
        rgba_frame = cv2.cvtColor(main_road_image, cv2.COLOR_RGB2RGBA)
        shared_mem_camera_rgba.write(rgba_frame, frame_i)
        shared_mem_model.write(model_output_array.astype(np.float16), frame_i)

        print(f"STEP {frame_i}: Vel={vEgo*2.23:.1f} MPH, Steer={vehicle_state.steering_angle:.1f} deg")

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        frame_i += 1

        #os._exit(0)
'''

# __________________________________________________________________________ #
# __________________________________________________________________________ #


def run_metadrive2():
    """
    Runs the Metadrive simulation with manual keyboard controls
    and overlays the ground-truth centerline and navigation data
    from the simulator itself, ignoring all OpenPilot model predictions.
    """

    print_in_color("Starting run_metadrive2() for control debugging...", "green")
    print_in_color("Controls: [Up] Gas, [Down] Brake, [Left] Steer Left, [Right] Steer Right, [x] Exit", "cyan")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 1. Initialize Gym, UI, and Shared Memory
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    metadrive_gym = MetadriveGym()

    # We only need the camera shared memory to send images to display_video.py
    shared_mem_camera_rgba = CameraViewerSharedMemoryArray_RGBA(bool_create=True, service_name="metadrive_gym")

    # We need the UI overlay to draw the path and text
    ui_overlay = UI_Overlay()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2. Initialize Loop Variables
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    frame_i = 0

    # For calculating acceleration (aEgo)
    prev_vEgo = 0.0
    # Get delta_t from the simulator config to be precise
    delta_t = metadrive_gym.env.engine.global_config['physics_world_step_size']

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3. Run Simulation Loop
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    while True:

        # --- Read Keyboard Input ---
        throttle_manual = 0.0
        brake_manual = 0.0
        steer_manual = 0.0 # This is in degrees (for our controls)

        if keyboard.is_pressed("up"):
            throttle_manual = 1.0

        if keyboard.is_pressed("down"):
            brake_manual = 1.0

        if keyboard.is_pressed("left"):
            steer_manual = 6  # 6 degrees left

        if keyboard.is_pressed("right"):
            steer_manual = -6 # 6 degrees right

        if keyboard.is_pressed("x"):
            print_in_color("Exitting...", "green")
            break

        # --- Step the Simulator ---
        # We get vehicle_state, images, and the new ground-truth 'info' dict
        vehicle_state, main_road_image, wide_road_image, bool_out_of_lane = \
            metadrive_gym.step(steer_manual, throttle_manual, brake_manual, bool_reset=False)

        # --- Prepare CarState Data (for UI display) ---
        vEgo = math.sqrt(vehicle_state.velocity.x**2 + vehicle_state.velocity.y**2)
        aEgo = (vEgo - prev_vEgo) / delta_t
        prev_vEgo = vEgo

        car_state_dict = {
            "vEgo": vEgo,
            "aEgo": aEgo,
            "steeringAngleDeg": vehicle_state.steering_angle,
            "vCruise": 0.0, # We're not using OP cruise
            "standstill": bool(vEgo < 1.0),
            "brakePressed": bool(brake_manual >= 0.1),
            "cruiseState": {"enabled": False, "standstill": bool(vEgo < 1.0), },
        }

        # ____________________________________________________________________ #
        # ____________________________________________________________________ #

        # --- Create Ground Truth Centerline Path (Using get_points()) ---

        # Get the vehicle's current lane object
        agent_lane = metadrive_gym.env.vehicle.lane

        # Get the centerline points (these are in world coordinates)
        # We call the get_points() method, sampling every 5 meters.

        agent_centerline_points = None


        # FIX: Check lane type and get points based on the correct API
        if isinstance(agent_lane, CircularLane):
            # For curves, we must manually sample points along the arc
            agent_centerline_points = []
            num_segments = 20
            for i in range(num_segments + 1):
                phase = agent_lane.start_phase + (agent_lane.end_phase - agent_lane.start_phase) * (i / num_segments)
                x = agent_lane.center[0] + agent_lane.radius * math.cos(phase)
                y = agent_lane.center[1] + agent_lane.radius * math.sin(phase)
                agent_centerline_points.append((x, y))
        elif isinstance(agent_lane, StraightLane):
            # FIX: For straights, we must manually sample points
            agent_centerline_points = []
            num_segments = 20 # Same as the curve
            start = agent_lane.start
            end = agent_lane.end
            for i in range(num_segments + 1):
                t = i / num_segments
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                agent_centerline_points.append((x, y))


        if agent_centerline_points is not None and len(agent_centerline_points) > 1:
            car_x_world = vehicle_state.position[0]
            car_y_world = vehicle_state.position[1]
            heading_rad = vehicle_state.heading_theta # Now available!

            sin_h = math.sin(heading_rad)
            cos_h = math.cos(heading_rad)

            path_x_fwd = []
            path_y_left = []

            # Convert world points to vehicle's local (OpenPilot) coordinate system
            for point_world in agent_centerline_points:
                dx = point_world[0] - car_x_world
                dy = point_world[1] - car_y_world

                # World (X-right, Y-fwd) to OpenPilot (X-fwd, Y-left)
                x_fwd = dx * cos_h + dy * sin_h
                y_left = dx * sin_h - dy * cos_h

                # FIX: Only add points that are IN FRONT of the car
                if x_fwd > 0:
                    path_x_fwd.append(x_fwd)
                    path_y_left.append(y_left)

            # Use run_planner.simulate_object (which is just a dict wrapper)
            # to create a path object that ui_helpers.plot_model can understand
            gt_path_dict = {
                "x": np.array(path_x_fwd),
                "y": np.array(path_y_left),
                "z": np.zeros(len(path_x_fwd))
            }
            gt_path_obj = run_planner.simulate_object(gt_path_dict)

        # ____________________________________________________________________ #
        # ____________________________________________________________________ #

        # --- Draw Overlays ---

        # 1. Draw the ground truth path
        if gt_path_obj is not None:
            print_in_color(f"[metadrive_gym.py] path overlay", "green")

            # We call plot_model directly from ui_helpers
            ui_helpers.plot_model(
                main_road_image, 
                ui_overlay.camera_calibration, 
                plan_positions=gt_path_obj,  # This will be drawn in CYAN
                mpc_plan_positions=None      # We pass None for the red MPC path
            )
        else:
            print_in_color(f"[metadrive_gym.py] path not present", "red")

        # 2. Draw the ground truth text values
        # We'll "hack" the CarControl_dict to pass our ground truth values
        # to the UI text display.
        CarControl_dict = {
            "enabled": False, 
            "actuators": { 
                "steeringAngleDeg": 0.0,
                "accel": 0.0,
            } 
        }
        # Use desire_str to label our hacked values
        #desire_str = f"GT SteerErr: {heading_err_deg:6.2f} deg | GT LatOff: {lat_offset:6.2f} m"

        ui_overlay.overlay_control_values(
            main_road_image, 
            frame_i, 
            car_state_dict, 
            CarControl_dict, 
            desire_str="", 
            left_blinker=False, 
            right_blinker=False
        )

        # --- Write to Shared Memory for Display ---
        if BOOL_WRITE_SHARED_MEM:
            rgba_frame = cv2.cvtColor(main_road_image, cv2.COLOR_RGB2RGBA)
            shared_mem_camera_rgba.write(rgba_frame, frame_i)

        # --- Print to Console ---
        print(f"STEP {frame_i}: Vel={vEgo*2.23:.1f} MPH, SteerIn={steer_manual:.1f} deg")

        frame_i += 1

if __name__ == "__main__":
    run_metadrive2()
