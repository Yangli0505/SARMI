import copy
import os
from typing import TYPE_CHECKING, Callable, List, Optional
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action


class RiskFieldGraphics:
    @staticmethod
    def dynamic_vehicle_field(x, y, v, x_obs, y_obs, v_obs):
        """计算车辆动态风险场"""
        A = 5.0
        kv = 0.5
        alpha = 0.2
        L_obs = 5.0
        # 增大最小纵向范围
        min_sigma_v = 6  # 从2.0增加到4.0
        sigma_v = max(min_sigma_v, kv * np.abs(v_obs - v))
        # 减小横向范围
        sigma_y = max(2, min(3.0, sigma_v/3))  # 横向范围在1.5-3米之间，并减小与sigma_v的比例
        
        relv = np.where(v_obs > v, 1, -1)
        numerator = A * np.exp(-((x - x_obs)**2 / sigma_v**2) - ((y - y_obs)**2 / (sigma_y**2)))
        denominator = 1 + np.exp(relv * (x - x_obs - alpha * L_obs * relv))
        return numerator / denominator

    @classmethod
    def display_risk_field(cls, env, surface):
        """显示动态风险场"""
        if env.vehicle.position[0] > env.road.road_ends:
            return
            
        # 根据所有车辆的位置动态确定显示范围
        vehicle_positions = [v.position[0] for v in env.road.vehicles if v is not env.vehicle]
        if not vehicle_positions:  # 如果没有其他车辆，直接返回
            return
            
        # 确定显示范围，考虑所有车辆
        x_min = min(vehicle_positions) - 20
        x_max = max(vehicle_positions) + 20
        y_min, y_max = -10, 10
        
        # 增加网格分辨率
        x = np.linspace(x_min, x_max, 800)  # 从400增加到800
        y = np.linspace(y_min, y_max, 320)  # 从160增加到320
        X, Y = np.meshgrid(x, y)
        
        # 计算总风险场
        risk_field = np.zeros_like(X)
        
        # 计算所有其他车辆的风险场
        for vehicle in env.road.vehicles:
            if vehicle is not env.vehicle:  # 排除自车
                v_field = cls.dynamic_vehicle_field(
                    X, Y, 
                    env.vehicle.speed,
                    vehicle.position[0], vehicle.position[1],
                    vehicle.speed
                )
                risk_field += v_field
        
        # 检查数值
        if np.any(np.isnan(risk_field)) or np.any(np.isinf(risk_field)):
            risk_field = np.nan_to_num(risk_field, 0)
        
        # 增强风险场的对比度
        risk_field = np.power(risk_field, 0.7)  # 使用幂函数增强对比度
        
        # 将风险场转换为颜色
        cmap = cm.get_cmap('viridis')
        risk_field_normalized = risk_field / np.max(risk_field) if np.max(risk_field) > 0 else risk_field
        colors = cmap(risk_field_normalized)
        
        # 使用更高分辨率的surface
        risk_surface = pygame.Surface((800, 320), pygame.SRCALPHA)  # 增加surface的分辨率
        for i in range(800):
            for j in range(320):
                r = min(255, max(0, int(colors[j, i, 0] * 255 * 0.8)))
                g = min(255, max(0, int(colors[j, i, 1] * 255 * 0.8)))
                b = min(255, max(0, int(colors[j, i, 2] * 255 * 0.8)))
                alpha = min(255, max(0, int(255 * np.power(risk_field_normalized[j, i], 0.25))))
                risk_surface.set_at((i, j), (r, g, b, alpha))
        
        # 缩放和位置调整
        scaled_surface = pygame.transform.scale(risk_surface, 
            (int((x_max - x_min) * surface.scaling), int((y_max - y_min) * surface.scaling)))
        
        # 计算绘制位置
        pos = surface.pos2pix(x_min, y_min)
        surface.blit(scaled_surface, pos)

############################################################################################
class EnvViewer(object):

    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False

    def __init__(self, env: 'AbstractEnv', config: Optional[dict] = None) -> None:
        self.env = env
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]
        self.show_mpc_trajectory = self.config["show_mpc_trajectory"]
        self.show_other_vehicles_predict = self.config['show_other_vehicles_predict']
        self.show_risk_field = self.config['show_risk_field']
        
        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode()
        # instruction allows the drawing to be done on surfaces without
        # handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.config["screen_width"], self.config["screen_height"]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = self.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None

    def set_agent_display(self, agent_display: Callable) -> None:
        """
        Set a display callback provided by an agent

        So that they can render their behaviour on a dedicated agent surface, or even on the simulation surface.

        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if self.agent_display is None:
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen = pygame.display.set_mode((self.config["screen_width"],
                                                           2 * self.config["screen_height"]))
                else:
                    self.screen = pygame.display.set_mode((2 * self.config["screen_width"],
                                                           self.config["screen_height"]))
            self.agent_surface = pygame.Surface((self.config["screen_width"], self.config["screen_height"]))
        self.agent_display = agent_display

    def set_agent_action_sequence(self, actions: List['Action']) -> None:
        """
        Set the sequence of actions chosen by the agent, so that it can be displayed

        :param actions: list of action, following the env's action space specification
        """
        if isinstance(self.env.action_type, DiscreteMetaAction):
            actions = [self.env.action_type.actions[a] for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,
                                                                          1 / self.env.config["policy_frequency"],
                                                                          1 / 3 / self.env.config["policy_frequency"],
                                                                          1 / self.env.config["simulation_frequency"])

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
    
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        # 在绘制车辆之前显示风险场
        if self.show_risk_field:
            RiskFieldGraphics.display_risk_field(self.env, self.sim_surface)

        if self.show_mpc_trajectory and self.env.vehicle.position[0] < self.env.road.road_ends:
            mpc_trajectory = self.env.vehicle.show_mpc_trajectory()
            VehicleGraphics.display_trajectory(
                mpc_trajectory,
                self.sim_surface,
                offscreen = self.offscreen
            )
        
        if self.show_other_vehicles_predict:
            if self.env.vehicle.position[0] > self.env.road.road_ends:
                other_vehicle = self.env.road.vehicles
            else:
                other_vehicle = self.env.road.vehicles[:-1]
            predict_time = np.arange(0, 0.8, 0.1)
            for i in range (len(other_vehicle)):
                other_vehicle_trajectory = []
                v = copy.deepcopy(other_vehicle[i])
                position, _ = other_vehicle[i].predict_trajectory_constant_speed(predict_time)
                for j in range(len(position)):
                    v.position = position[j]
                    other_vehicle_trajectory.append(copy.deepcopy(v))
                VehicleGraphics.display_trajectory(
                    other_vehicle_trajectory,
                    self.sim_surface,
                    offscreen=self.offscreen
                )

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen,
            config=self.config)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1

    def get_image(self) -> np.ndarray:
        """
        The rendered image as a rgb array.

        OpenAI gym's channel convention is H x W x C
        """
        surface = self.screen if self.config["render_agent"] and not self.offscreen else self.sim_surface
        data = pygame.surfarray.array3d(surface)  # in W x H x C channel convention
        return np.moveaxis(data, 0, 1)

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()


class EventHandler(object):
    @classmethod
    def handle_event(cls, action_type: ActionType, event: pygame.event.EventType) -> None:
        """
        Map the pygame keyboard events to control decisions

        :param action_type: the ActionType that defines how the vehicle is controlled
        :param event: the pygame event
        """
        if isinstance(action_type, DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)
        elif action_type.__class__ == ContinuousAction:
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_discrete_action_event(cls, action_type: DiscreteMetaAction, event: pygame.event.EventType) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_LEFT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_DOWN and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_UP:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])

    @classmethod
    def handle_continuous_action_event(cls, action_type: ContinuousAction, event: pygame.event.EventType) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)


class ObservationGraphics(object):
    COLOR = (0, 0, 0)

    @classmethod
    def display(cls, obs, sim_surface):
        from highway_env.envs.common.observation import LidarObservation
        if isinstance(obs, LidarObservation):
            cls.display_grid(obs, sim_surface)

    @classmethod
    def display_grid(cls, lidar_observation, surface):
        psi = np.repeat(np.arange(-lidar_observation.angle/2,
                                  2 * np.pi - lidar_observation.angle/2,
                                  2 * np.pi / lidar_observation.grid.shape[0]), 2)
        psi = np.hstack((psi[1:], [psi[0]]))
        r = np.repeat(np.minimum(lidar_observation.grid[:, 0], lidar_observation.maximum_range), 2)
        points = [(surface.pos2pix(lidar_observation.origin[0] + r[i] * np.cos(psi[i]),
                                   lidar_observation.origin[1] + r[i] * np.sin(psi[i])))
                  for i in range(np.size(psi))]
        pygame.draw.lines(surface, ObservationGraphics.COLOR, True, points, 1)
