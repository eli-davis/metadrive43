# B"H

# based on openpilot 0.9.9
# --> openpilot/tools/sim/bridge/metadrive/metadrive_bridge.py

import os
import time
import sys
import math
import numpy as np

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

import math
import time
import numpy as np

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

def apply_metadrive_patches(arrive_dest_done=True):
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
        sensor = cls(*args, self, cuda=True)
    else:
        sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  if not arrive_dest_done:
    MetaDriveEnv._is_arrive_destination = arrive_destination_patch


def metadrive_process(dual_camera: bool,
                      config: dict,
                      camera_array,
                      wide_camera_array):
    #image_lock,
    #controls_recv: Connection,
    #simulation_state_send: Connection,
    #vehicle_state_send: Connection,
    #exit_event,
    #op_engaged,
    #test_duration,
    #test_run):

    arrive_dest_done = config.pop("arrive_dest_done", True)
    apply_metadrive_patches(arrive_dest_done)

    road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    if dual_camera:
        assert wide_camera_array is not None
        wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    env = MetaDriveEnv(config)

    # ________________________________ #
    # ________________________________ #

    def get_current_lane_info(vehicle):
        _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
        lane_idx = lane_info[2] if lane_info is not None else None
        return lane_idx, on_lane

    def reset():
        env.reset()
        env.vehicle.config["max_speed_km_h"] = 1000
        lane_idx_prev, _ = get_current_lane_info(env.vehicle)

        simulation_state = metadrive_simulation_state(
            running=True,
            done=False,
            done_info=None,
        )
        simulation_state_send.send(simulation_state)

        return lane_idx_prev

    # ________________________________ #
    # ________________________________ #

    lane_idx_prev = reset()
    start_time = None

    # ________________________________ #
    # ________________________________ #

    def get_cam_as_rgb(cam):
        cam = env.engine.sensors[cam]
        cam.get_cam().reparentTo(env.vehicle.origin)
        cam.get_cam().setPos(C3_POSITION)
        cam.get_cam().setHpr(C3_HPR)
        img = cam.perceive(to_float=False)
        if not isinstance(img, np.ndarray):
            # convert cupy array to numpy
            img = img.get()
        return img

    # ________________________________ #
    # ________________________________ #

    steer_ratio = 8
    vc = [0,0]


    while True:

        vehicle_state = metadrive_vehicle_state(
            velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
            position=env.vehicle.position,
            bearing=float(math.degrees(env.vehicle.heading_theta)),
            steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING
        )

        # TODO: steer_angle, gas, should_reset = controls_recv.recv()

        steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
        steer_metadrive = np.clip(steer_metadrive, -1, 1)

        vc = [steer_metadrive, gas]

        if should_reset:
            lane_idx_prev = reset()
            start_time = None


        _, _, terminated, _, _ = env.step(vc)

        lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
        out_of_lane = lane_idx_curr != lane_idx_prev or not on_lane
        lane_idx_prev = lane_idx_curr

        road_image[...] = get_cam_as_rgb("rgb_road")

        if dual_camera:
            wide_road_image[...] = get_cam_as_rgb("rgb_wide")



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
# --> openpilot/tools/sim/bridge/metadrive/metadrive_world.py

import ctypes
import functools
#import multiprocessing
import numpy as np
import time

#from multiprocessing import Pipe, Array

#from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
#from openpilot.tools.sim.bridge.metadrive.metadrive_process import (metadrive_process, metadrive_simulation_state,
#                                                                   metadrive_vehicle_state)
#from openpilot.tools.sim.lib.common import SimulatorState, World
#from openpilot.tools.sim.lib.camerad import W, H


class MetaDriveWorld():
  def __init__(self):

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # setup cameras

    dual_camera=True

    self.road_image = np.zeros((H, W, 3), dtype=np.uint8)
    self.wide_road_image = np.zeros((H, W, 3), dtype=np.uint8)

    self.camera_array = Array(ctypes.c_uint8, W*H*3)
    self.road_image = np.frombuffer(self.camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    self.wide_camera_array = None

    if dual_camera:
      self.wide_camera_array = Array(ctypes.c_uint8, W*H*3)
      self.wide_road_image = np.frombuffer(self.wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    sensors = dict()

    sensors["rgb_road"] = (RGBCameraRoad, W, H)

    if dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    config = dict(
      use_render=False,

      vehicle_config=dict(
        enable_reverse=False,
        render_vehicle=False,
        image_source="rgb_road",
      ),

      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=False,
      on_continuous_line_done=False,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=False,
      traffic_density=0.0, # traffic is incredibly expensive
      map_config=create_map(),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False,
      show_logo=False,
      anisotropic_filtering=False
    )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    metadrive_process(dual_camera, config, self.camera_array, self.wide_camera_array)
    #self.image_lock,
    #self.controls_recv,
    #self.simulation_state_send,
    #self.vehicle_state_send,
    #self.exit_event,
    #self.op_engaged,
    #test_duration,
    #self.test_run))

    print("----------------------------------------------------------")
    print("---- Spawning Metadrive world, this might take awhile ----")
    print("----------------------------------------------------------")

    # wait for a state message to ensure metadrive is launched
    #self.vehicle_last_pos = self.vehicle_state_recv.recv().position

    self.steer_ratio = 15
    self.vc = [0.0,0.0]
    self.reset_time = 0
    self.should_reset = False


  def apply_controls(self, steer_angle, throttle_out, brake_out):
    if (time.monotonic() - self.reset_time) > 2:
      self.vc[0] = steer_angle

      if throttle_out:
        self.vc[1] = throttle_out
      else:
        self.vc[1] = -brake_out
    else:
      self.vc[0] = 0
      self.vc[1] = 0

    self.controls_send.send([*self.vc, self.should_reset])
    self.should_reset = False

  def read_state(self):
    list_of_states = []
    while self.simulation_state_recv.poll(0):
      list_of_states += self.simulation_state_recv.recv()
    return list_of_states


  def read_sensors(self, state: SimulatorState):
    while self.vehicle_state_recv.poll(0):
      md_vehicle: metadrive_vehicle_state = self.vehicle_state_recv.recv()
      curr_pos = md_vehicle.position

      state.velocity = md_vehicle.velocity
      state.bearing = md_vehicle.bearing
      state.steering_angle = md_vehicle.steering_angle
      state.gps.from_xy(curr_pos)
      state.valid = True

      is_engaged = state.is_engaged
      if is_engaged and self.first_engage is None:
        self.first_engage = time.monotonic()
        self.op_engaged.set()

      # check moving 5 seconds after engaged, doesn't move right away
      after_engaged_check = is_engaged and time.monotonic() - self.first_engage >= 5 and self.test_run

      x_dist = abs(curr_pos[0] - self.vehicle_last_pos[0])
      y_dist = abs(curr_pos[1] - self.vehicle_last_pos[1])
      dist_threshold = 1
      if x_dist >= dist_threshold or y_dist >= dist_threshold: # position not the same during staying still, > threshold is considered moving
        self.distance_moved += x_dist + y_dist

      time_check_threshold = 29
      current_time = time.monotonic()
      since_last_check = current_time - self.last_check_timestamp
      if since_last_check >= time_check_threshold:
        if after_engaged_check and self.distance_moved == 0:
          self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"vehicle_not_moving" : True}))
          self.exit_event.set()

        self.last_check_timestamp = current_time
        self.distance_moved = 0
        self.vehicle_last_pos = curr_pos

  def read_cameras(self):
    pass

  def tick(self):
    pass

  def reset(self):
    self.should_reset = True

  def close(self, reason: str):
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))
    self.exit_event.set()
    self.metadrive_process.join()


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

'''
class MetaDriveBridge():

  TICKS_PER_FRAME = 5

  def __init__(self):
    pass


  def send_camera_images(self, world: 'World'):
    world.image_lock.acquire()
    yuv = self.camerad.rgb_to_yuv(world.road_image)
    self.camerad.cam_send_yuv_road(yuv)

    if world.dual_camera:
      yuv = self.camerad.rgb_to_yuv(world.wide_road_image)
      self.camerad.cam_send_yuv_wide_road(yuv)


  def run(self):
    self.world = MetaDriveWorld()

    #self.simulated_car = SimulatedCar()
    #self.simulated_sensors = SimulatedSensors(self.dual_camera)


    #self.simulated_camera_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_sensors.send_camera_images, self.world),
    #                                                                    20, self._exit_event))

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      self.world.tick()

    while self._keep_alive:
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0

      self.simulator_state.cruise_button = 0
      self.simulator_state.left_blinker = False
      self.simulator_state.right_blinker = False

      throttle_manual = steer_manual = brake_manual = 0.

      # Read manual controls
      if not q.empty():
        message = q.get()
        if message.type == QueueMessageType.CONTROL_COMMAND:
          m = message.info.split('_')
          if m[0] == "steer":
            steer_manual = float(m[1])
          elif m[0] == "throttle":
            throttle_manual = float(m[1])
          elif m[0] == "brake":
            brake_manual = float(m[1])
          elif m[0] == "cruise":
            if m[1] == "down":
              self.simulator_state.cruise_button = CruiseButtons.DECEL_SET
            elif m[1] == "up":
              self.simulator_state.cruise_button = CruiseButtons.RES_ACCEL
            elif m[1] == "cancel":
              self.simulator_state.cruise_button = CruiseButtons.CANCEL
            elif m[1] == "main":
              self.simulator_state.cruise_button = CruiseButtons.MAIN
          elif m[0] == "blinker":
            if m[1] == "left":
              self.simulator_state.left_blinker = True
            elif m[1] == "right":
              self.simulator_state.right_blinker = True
          elif m[0] == "ignition":
            self.simulator_state.ignition = not self.simulator_state.ignition
          elif m[0] == "reset":
            self.world.reset()
          elif m[0] == "quit":
            break

      self.simulator_state.user_brake = brake_manual
      self.simulator_state.user_gas = throttle_manual
      self.simulator_state.user_torque = steer_manual * -10000

      steer_manual = steer_manual * -40

      # Update openpilot on current sensor state
      self.simulated_sensors.update(self.simulator_state, self.world)

      self.simulated_car.sm.update(0)
      self.simulator_state.is_engaged = self.simulated_car.sm['selfdriveState'].active

      if self.simulator_state.is_engaged:
        throttle_op = np.clip(self.simulated_car.sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = np.clip(-self.simulated_car.sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = self.simulated_car.sm['carControl'].actuators.steeringAngleDeg

        self.past_startup_engaged = True
      elif not self.past_startup_engaged and self.simulated_car.sm['selfdriveState'].engageable:
        self.simulator_state.cruise_button = CruiseButtons.DECEL_SET if self.startup_button_prev else CruiseButtons.MAIN # force engagement on startup
        self.startup_button_prev = not self.startup_button_prev

      throttle_out = throttle_op if self.simulator_state.is_engaged else throttle_manual
      brake_out = brake_op if self.simulator_state.is_engaged else brake_manual
      steer_out = steer_op if self.simulator_state.is_engaged else steer_manual

      self.world.apply_controls(steer_out, throttle_out, brake_out)
      self.world.read_state()
      self.world.read_sensors(self.simulator_state)

      if self.world.exit_event.is_set():
        self.shutdown()

      if self.rk.frame % self.TICKS_PER_FRAME == 0:
        self.world.tick()
        self.world.read_cameras()
'''




def run_metadrive():
    simulator_bridge = MetaDriveBridge()


if __name__ == "__main__":
    run_metadrive()
