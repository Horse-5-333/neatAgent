from physics import *
from agent import *
import arcade
import time
import collections
import pickle
import os

BG_COLOR = (26, 29, 26)
TXT_COLOR = (247, 247, 249)
DARK_GRAY = (51, 82, 88)
LIGHT_GRAY = (165, 178, 182)
ACC1_COLOR = (219, 127, 103)
ACC2_COLOR = (143, 45, 86)
SUBTLE_GRID_COLOR = (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2],
                     50)  # RGBA with a low alpha so it stays in the background


# UI
class LiveLineChart:
    def __init__(self, x, y, width, height, title="Chart", min_y=0, max_y=60, max_points=100,
                 line_color=arcade.color.GREEN):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_y = min_y
        self.max_y = max_y
        self.max_points = max_points
        self.line_color = line_color
        self.data = collections.deque(maxlen=max_points)

        # --- NEW: Grid Tracking ---
        self.total_points_added = 0
        self.grid_x_spacing = max_points // 10  # Drop a vertical line every 10% of the chart

        self.title_text = arcade.Text(title, x + 5, y + height + 25, arcade.color.WHITE, 14, font_name="Jetbrains Mono")
        self.max_text = arcade.Text(f"{max_y}", x - 35, y + height - 10, arcade.color.GRAY, 14,
                                    font_name="Jetbrains Mono", bold="semibold")
        self.min_text = arcade.Text(f"{min_y}", x - 35, y, arcade.color.GRAY, 12, font_name="Jetbrains Mono")

    def add_point(self, value):
        self.data.append(value)
        self.total_points_added += 1  # --- NEW: Increment total points ---

    def draw(self):
        # Background
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height,
                                          (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 25))

        # --- NEW: THE GRID ---

        # 1. Horizontal (Static) Grid Lines
        grid_y_divisions = 6
        for j in range(1, grid_y_divisions):
            py = self.y + (j / grid_y_divisions) * self.height
            arcade.draw_line(self.x, py, self.x + self.width, py, SUBTLE_GRID_COLOR, 1)

        # 2. Vertical (Scrolling) Grid Lines
        if len(self.data) > 1:
            x_step = self.width / (self.max_points - 1)

            # Find the absolute index of the oldest point currently in the deque
            start_index = self.total_points_added - len(self.data)

            for i in range(len(self.data)):
                absolute_index = start_index + i

                # If this specific data point falls on a grid interval, draw a vertical line behind it
                if absolute_index % self.grid_x_spacing == 0:
                    px = self.x + (i * x_step)
                    arcade.draw_line(px, self.y, px, self.y + self.height, SUBTLE_GRID_COLOR, 1)

        # --- END GRID ---

        # y-axis
        arcade.draw_line(self.x, self.y, self.x, self.y + self.height, LIGHT_GRAY, 2)

        # x-axis (y=0)
        if self.min_y <= 0 <= self.max_y:
            x_height = (-self.min_y / (self.max_y - self.min_y)) * self.height
            arcade.draw_line(self.x, self.y + x_height, self.x + self.width, self.y + x_height, LIGHT_GRAY, 2)

        # Text labels
        self.title_text.draw()
        self.max_text.draw()
        self.min_text.draw()

        # Data Line
        if len(self.data) > 1:
            points = []
            x_step = self.width / (self.max_points - 1)
            y_range = self.max_y - self.min_y

            for i, val in enumerate(self.data):
                clamped_val = max(self.min_y, min(self.max_y, val))
                normalized_y = (clamped_val - self.min_y) / y_range if y_range != 0 else 0
                px = self.x + (i * x_step)
                py = self.y + (normalized_y * self.height)
                points.append((px, py))

            arcade.draw_line_strip(points, self.line_color, line_width=4)


# --- MAIN APPLICATION ---

class PhysicsSimulator(arcade.Window):
    def __init__(self, env, brain_filepath=None):
        super().__init__(int(SCREEN_WIDTH * PPM), int(SCREEN_HEIGHT * PPM), "Double Pendulum Simulator",
                         antialiasing=True)
        self.agent_score_in_period = None
        arcade.set_background_color(BG_COLOR)

        # AI setup
        self.env = env

        # Pickle Loading Logic
        if brain_filepath and os.path.exists(brain_filepath):
            with open(brain_filepath, "rb") as f:
                self.network = pickle.load(f)
            print(f"Successfully loaded brain from: {brain_filepath}")
        else:
            self.network = gen0_network()
            print("No saved brain provided or found. Falling back to random Gen 0 brain.")

        self.current_obs = self.env.observations()
        self.reward = None

        # HUD Trackers
        self.frame_count = 0
        self.total_score = 0.0

        # scene collections (obejcts, graphs)
        self.scene_masses = [self.env.cart, self.env.bob1, self.env.bob2]
        self.action_hist = collections.deque(maxlen=300)
        self.reward_hist_data = collections.deque(maxlen=300)

        # camera and rendering planes
        self.main_camera = arcade.Camera2D()
        self.text_camera = arcade.Camera2D()
        self.text_camera.zoom = 0.5
        self.text_camera.bottom_left = (0, 0)

        # --- NEW: HUD Cached Text Objects ---
        # Because text_camera zoom is 0.5, the coordinate system is effectively doubled
        viewport_top = (SCREEN_HEIGHT * PPM) * 2

        self.frame_text = arcade.Text(
            "FRAME: 0000",
            x=75,
            y=viewport_top - 60,
            color=TXT_COLOR,
            font_size=24,
            font_name="Jetbrains Mono",
            bold=True
        )

        self.score_text = arcade.Text(
            "SCORE:     0.00",
            x=75,
            y=viewport_top - 100,
            color=ACC1_COLOR,
            font_size=24,
            font_name="Jetbrains Mono",
            bold=True
        )

        self.action_hist_chart = LiveLineChart(
            x=75, y=75, width=600, height=300,
            title="Agent Force Applied", min_y=-2, max_y=2, max_points=300,
            line_color=(ACC1_COLOR[0], ACC1_COLOR[1], ACC1_COLOR[2], 200)
        )

        self.reward_hist_chart = LiveLineChart(
            x=75 / 2 + (SCREEN_WIDTH * PPM), y=75, width=600, height=300,
            title="Reward Received", min_y=0, max_y=6, max_points=300,
            line_color=(ACC2_COLOR[0], ACC2_COLOR[1], ACC2_COLOR[2], 200)
        )

    def on_update(self, delta_time):
        cycle_start_time = time.perf_counter()

        action_force = self.network.forward_pass(self.current_obs)
        self.action_hist.append(action_force)
        self.action_hist_chart.add_point(action_force)

        self.current_obs, self.reward = self.env.step(action_force, 1/60.0)

        # Update HUD Trackers
        self.frame_count += 1
        if self.reward is not None:
            self.total_score += self.reward

        # --- NEW: Update Cached Text Properties ---
        if self.frame_count <= 900:
            self.agent_score_in_period = self.total_score
            self.frame_text.text = f"FRAME: {self.frame_count :04d}"
            self.score_text.text = f"SCORE: {self.total_score:>4.0f}"
        else:
            self.frame_text.text = f"FRAME: 0900 + {self.frame_count-900:0d}"
            self.score_text.text = (f"SCORE: {self.agent_score_in_period:>4.0f} + "
                                    f"{self.total_score - self.agent_score_in_period:>.0f}")
        reward = self.reward
        if reward is not None:
            self.reward_hist_data.append(reward)
            self.reward_hist_chart.add_point(reward)

    def on_draw(self):
        self.clear()

        with self.main_camera.activate():
            draw_track(TRACK_HEIGHT, TRACK_LENGTH)

            # Draw Pole 1
            arcade.draw_line(self.env.cart.s.x * PPM, self.env.cart.s.y * PPM,
                             self.env.bob1.s.x * PPM, self.env.bob1.s.y * PPM,
                             LIGHT_GRAY, 4)

            # Draw Pole 2
            arcade.draw_line(self.env.bob1.s.x * PPM, self.env.bob1.s.y * PPM,
                             self.env.bob2.s.x * PPM, self.env.bob2.s.y * PPM,
                             LIGHT_GRAY, 4)

            # Draw Masses
            for pt_mass in self.scene_masses:
                pos_p = pt_mass.s * PPM
                radius_p = pt_mass.radius_m * PPM
                arcade.draw_circle_filled(pos_p.x, pos_p.y, radius_p, ACC1_COLOR)
                draw_vec(pt_mass.s, pt_mass.v, ACC2_COLOR)

        with self.text_camera.activate():
            self.action_hist_chart.draw()
            self.reward_hist_chart.draw()

            # --- NEW: Call .draw() on cached objects ---
            self.frame_text.draw()
            self.score_text.draw()


def draw_vec(tail, vector, color, thickness=2, scaling=True, arrowhead=True):
    # auto determine thickness: try to be 3px, unless len < cutoff, then scale down.
    # tip should try to be same size, unless vector is small
    # length: 0.25 to 2 meters

    if scaling:
        vector = vector * 0.1
        orig_len = vector.magnitude
        if orig_len < 0.25:
            vector = vector * (0.25 / orig_len)
        elif orig_len > 3:
            vector = vector * (3 / orig_len)

    tip = (tail + vector)
    arcade.draw_line(tail.x * PPM, tail.y * PPM,
                     tip.x * PPM, tip.y * PPM,
                     color, thickness)

    # arrowhead stuffs
    if arrowhead:
        flare1 = tip - (0.15 * vector.normalized()) + (0.15 * vector.perpendicular_normalized)
        flare2 = tip - (0.15 * vector.normalized()) + (-0.15 * vector.perpendicular_normalized)

        arcade.draw_line(tip.x * PPM, tip.y * PPM,
                         flare1.x * PPM, flare1.y * PPM,
                         color, thickness)

        arcade.draw_line(tip.x * PPM, tip.y * PPM,
                         flare2.x * PPM, flare2.y * PPM,
                         color, thickness)


def draw_track(height, length):
    start = Vec(SCREEN_WIDTH / 2 - length / 2, height)
    end = Vec(SCREEN_WIDTH / 2 + length / 2, height)
    arcade.draw_line(start.x * PPM, height * PPM,
                     end.x * PPM, height * PPM,
                     DARK_GRAY, 2)

    # meter ticks
    draw_vec(Vec(SCREEN_WIDTH / 2 - length / 2, height - 0.25),
             Vec(0, .5),
             DARK_GRAY, 2, False, False)
    draw_vec(Vec(SCREEN_WIDTH / 2 + length / 2, height - 0.25),
             Vec(0, .5),
             DARK_GRAY, 2, False, False)
    for i in range(1, length):
        draw_vec(Vec(SCREEN_WIDTH / 2 - length / 2 + i, height - 0.1),
                 Vec(0, .2),
                 DARK_GRAY, 2, False, False)


if __name__ == "__main__":
    env = DoublePendulumEnv()
    env.reset()

    glitched_brain_path = "saved_networks/champion_gen_20.pkl"

    viewer = PhysicsSimulator(env, brain_filepath=glitched_brain_path)
    arcade.run()