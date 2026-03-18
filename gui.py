import arcade
import time
import collections
import pickle
import os
import threading
import queue
import math
import concurrent.futures.process
import concurrent.futures
import concurrent.futures
import random
import colorsys
from functools import partial

from physics import Vec, DoublePendulumEnv, TRACK_HEIGHT, TRACK_LENGTH, SCREEN_WIDTH, SCREEN_HEIGHT, PPM
from agent import gen0_network, InnovationManager, Network, fast_forward_pass_flat

BG_COLOR = (26, 29, 26)
TXT_COLOR = (247, 247, 249)
DARK_GRAY = (51, 82, 88)
LIGHT_GRAY = (165, 178, 182)
ACC1_COLOR = (219, 127, 103)
ACC2_COLOR = (143, 45, 86)
ACC3_COLOR = (112, 160, 115) # Muted green accent requested
SUBTLE_GRID_COLOR = (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 50)

# Training Constants
POPULATION = 256
GENERATIONS = 1000
SIM_TIME = 20
DT = 1/60.0
ELITE_PERCENTILE = 0.1
ELITE_MUTATE = 0.8
CURRICULUM_STEP = 0.005
NEXT_STAGE_CUTOFF = 800
COMPATIBILITY_THRESHOLD = 5.0


from train import evaluate_single_network


def generate_species_color(species_id):
    # Use golden ratio conjugate to uniformly scatter hues across the color wheel
    golden_ratio_conjugate = 0.618033988749895
    h = (species_id * golden_ratio_conjugate) % 1.0
    # Medium saturation/value for distinct but not overly aggressive colors
    r, g, b = colorsys.hsv_to_rgb(h, 0.60, 0.75)
    return (int(r * 255), int(g * 255), int(b * 255), 255)


class NetworkVisualizer:
    def __init__(self, x, y, width, height, title="Network Architecture"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title_text = arcade.Text(title, x + 5, y + height + 5, arcade.color.WHITE, 12, font_name="Jetbrains Mono")
        self.network = None
        self.batch = None

    def update_network(self, new_network):
        self.network = new_network
        self.rebuild_shapes()

    def rebuild_shapes(self):
        if not self.network:
            return
        self.raw_lines = []
        self.raw_circles = []

        padding = 20
        usable_w = self.width - (padding * 2)
        usable_h = self.height - (padding * 2)

        node_positions = {}
        depth_counts = collections.defaultdict(int)
        depth_slots = collections.defaultdict(int)

        for n in self.network.neurons:
            depth_counts[n.depth] += 1

        for n in self.network.neurons:
            dx = self.x + padding + (n.depth * usable_w)
            slot = depth_slots[n.depth]
            total_at_depth = depth_counts[n.depth]
            dy = self.y + padding + ((slot / max(1, total_at_depth - 1)) * usable_h) if total_at_depth > 1 else self.y + self.height / 2
            node_positions[n.id] = (dx, dy)
            depth_slots[n.depth] += 1

        for s in self.network.connections:
            if not s.enabled: continue
            if s.input_id in node_positions and s.output_id in node_positions:
                start = node_positions[s.input_id]
                end = node_positions[s.output_id]
                color = tuple(list(ACC1_COLOR) + [150]) if s.weight >= 0 else tuple(list(ACC3_COLOR) + [150])
                thickness = 2
                self.raw_lines.append((start[0], start[1], end[0], end[1], color, thickness))
        for n_id, pos in node_positions.items():
            self.raw_circles.append((pos[0], pos[1]))

    def draw(self):
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height, (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 25))
        arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)
        self.title_text.draw()
        if hasattr(self, 'shape_list') and self.shape_list:
            self.shape_list.draw()
        elif hasattr(self, 'raw_lines'):
            for l in self.raw_lines:
                arcade.draw_line(l[0], l[1], l[2], l[3], l[4], l[5])
            for c in self.raw_circles:
                arcade.draw_circle_filled(c[0], c[1], 6, LIGHT_GRAY, num_segments=32)


class GenerationChart:
    def __init__(self, x, y, width, height, title="Generational Fitness", max_points=100):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points
        self.title_text = arcade.Text(title, x + 5, y + height + 5, arcade.color.WHITE, 12, font_name="Jetbrains Mono")
        
        self.max_fitness = collections.deque(maxlen=max_points)
        self.p75_fitness = collections.deque(maxlen=max_points)
        self.med_fitness = collections.deque(maxlen=max_points)
        self.p25_fitness = collections.deque(maxlen=max_points)
        self.min_fitness = collections.deque(maxlen=max_points)
        
        self.variance_events = collections.deque(maxlen=max_points)
        
        self.total_points_added = 0
        self.grid_x_spacing = max_points // 10
        self.min_y = 0
        self.max_y = 100
        self.max_text = arcade.Text(f"{self.max_y}", x - 5, y + height - 15, arcade.color.GRAY, 10, font_name="Jetbrains Mono", anchor_x="right")
        self.min_text = arcade.Text(f"{self.min_y}", x - 5, y + 5, arcade.color.GRAY, 10, font_name="Jetbrains Mono", anchor_x="right")

    def add_data(self, max_f, p75_f, med_f, p25_f, min_f, gen_num, var_increased):
        self.max_fitness.append(max_f)
        self.p75_fitness.append(p75_f)
        self.med_fitness.append(med_f)
        self.p25_fitness.append(p25_f)
        self.min_fitness.append(min_f)
        self.variance_events.append(var_increased)
        self.total_points_added += 1

    def draw(self):
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height, (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 25))
        arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)
        
        grid_y_divisions = 6
        for j in range(1, grid_y_divisions):
            py = self.y + (j / grid_y_divisions) * self.height
            arcade.draw_line(self.x, py, self.x + self.width, py, SUBTLE_GRID_COLOR, 1)

        if len(self.max_fitness) < 2: 
            self.title_text.draw()
            return

        x_step = self.width / (self.max_points - 1)
        start_index = self.total_points_added - len(self.max_fitness)
        for i in range(len(self.max_fitness)):
            if (start_index + i) % self.grid_x_spacing == 0:
                px = self.x + self.width - ((len(self.max_fitness) - 1 - i) * x_step)
                arcade.draw_line(px, self.y, px, self.y + self.height, SUBTLE_GRID_COLOR, 1)

        raw_max = max(max(self.max_fitness) * 1.10, 1.0)
        current_max = math.ceil(raw_max / 10.0) * 10.0
        self.max_y = current_max
        self.max_text.text = f"{self.max_y:.0f}"
        
        # Draw variance increased background bars
        for i in range(len(self.max_fitness)):
            if self.variance_events[i]:
                px = self.x + self.width - ((len(self.max_fitness) - 1 - i) * x_step)
                arcade.draw_lrbt_rectangle_filled(px - x_step/2, px + x_step/2, self.y, self.y + self.height, (ACC3_COLOR[0], ACC3_COLOR[1], ACC3_COLOR[2], 40))

        grid_y_divisions = 6
        for j in range(1, grid_y_divisions):
            py = self.y + (j / grid_y_divisions) * self.height
            arcade.draw_line(self.x, py, self.x + self.width, py, SUBTLE_GRID_COLOR, 1)

        for i in range(len(self.max_fitness)):
            if (start_index + i) % self.grid_x_spacing == 0:
                px = self.x + self.width - ((len(self.max_fitness) - 1 - i) * x_step)
                arcade.draw_line(px, self.y, px, self.y + self.height, SUBTLE_GRID_COLOR, 1)

        self.title_text.draw()
        self.max_text.draw()
        self.min_text.draw()
        
        pts_max, pts_p75, pts_med, pts_p25, pts_min = [], [], [], [], []
        for i in range(len(self.max_fitness)):
            px = self.x + self.width - ((len(self.max_fitness) - 1 - i) * x_step)
            pts_max.append((px, self.y + (self.max_fitness[i] / self.max_y) * self.height))
            pts_p75.append((px, self.y + (self.p75_fitness[i] / self.max_y) * self.height))
            pts_med.append((px, self.y + (self.med_fitness[i] / self.max_y) * self.height))
            pts_p25.append((px, self.y + (self.p25_fitness[i] / self.max_y) * self.height))
            pts_min.append((px, self.y + (self.min_fitness[i] / self.max_y) * self.height))

        # OPACITIES: min=25%, p25=50%, med=100%, p75=50%, max=25%
        # Colors: red for median, orange for the rest
        c_min = (255, 165, 0, 64)
        c_p25 = (255, 165, 0, 128)
        c_med = (ACC2_COLOR[0], ACC2_COLOR[1], ACC2_COLOR[2], 255)
        c_p75 = (255, 165, 0, 128)
        c_max = (255, 165, 0, 64)

        arcade.draw_line_strip(pts_min, c_min, 2)
        arcade.draw_line_strip(pts_p25, c_p25, 2)
        arcade.draw_line_strip(pts_p75, c_p75, 2)
        arcade.draw_line_strip(pts_max, c_max, 2)
        arcade.draw_line_strip(pts_med, c_med, 2)
        
        arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)


class SpeciesChart:
    def __init__(self, x, y, width, height, title="Species Population", max_points=100):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points
        self.title_text = arcade.Text(title, x + 5, y + height + 5, arcade.color.WHITE, 12, font_name="Jetbrains Mono")
        
        self.history = collections.deque(maxlen=max_points)
        self.shape_list = arcade.shape_list.ShapeElementList()
        self.needs_update = False

    def add_data(self, species_dict):
        self.history.append(species_dict)
        self.needs_update = True

    def draw(self):
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height, (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 25))
        self.title_text.draw()

        if len(self.history) < 2: 
            arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)
            return
            
        if self.needs_update:
            self._build_shapes()
            self.needs_update = False
            
        if self.shape_list:
            self.shape_list.draw()
            
        arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)

    def _build_shapes(self):
        self.shape_list = arcade.shape_list.ShapeElementList()
        x_step = self.width / (self.max_points - 1)
        
        all_ids = set()
        for dict_snap in self.history:
            all_ids.update(dict_snap.keys())

        sorted_ids = sorted(list(all_ids))

        stacked_y = []
        for i in range(len(self.history)):
            current_stack = {0: self.y} # id=0 represents the bottom starting coordinate for drawing
            running = 0
            total_pop = sum(self.history[i].values())
            if total_pop == 0: total_pop = 1
            for sid in sorted_ids:
                running += self.history[i].get(sid, 0)
                current_stack[sid] = self.y + (running / total_pop) * self.height
            stacked_y.append(current_stack)
            
        for i in range(len(self.history)):
            px = self.x + self.width - ((len(self.history) - 1 - i) * x_step)
            px_left = px
            px_right = px + x_step + 0.5 # Sub-pixel overlap destroys MSAA boundary lines while perfectly preserving width
            
            for sid in sorted_ids:
                color = generate_species_color(sid)
                prev_sid = sorted_ids[sorted_ids.index(sid)-1] if sorted_ids.index(sid) > 0 else 0
                
                y_bottom = stacked_y[i][prev_sid]
                y_top = stacked_y[i][sid]
                
                # Guarantee a minimum visual sliver of 2 pixels for any alive species
                if self.history[i].get(sid, 0) > 0:
                    y_top = max(y_top, y_bottom + 2.0)
                
                if y_top > y_bottom:
                    y_top_clamped = min(y_top, self.y + self.height)
                    y_bottom_clamped = min(y_bottom, self.y + self.height)
                    px_right_clamped = min(px_right, self.x + self.width)
                    
                    w = px_right_clamped - px_left
                    h = y_top_clamped - y_bottom_clamped
                    if w > 0 and h > 0:
                        cx = px_left + w / 2
                        cy = y_bottom_clamped + h / 2
                        rect = arcade.shape_list.create_rectangle_filled(cx, cy, w, h, color)
                        self.shape_list.append(rect)


class AgentInfoPanel:
    def __init__(self, x, y, width, height, title="Active Agent Details"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title_text = arcade.Text(title, x + 5, y + height + 5, arcade.color.WHITE, 12, font_name="Jetbrains Mono")
        
        self.gen_text = arcade.Text("Generation: -", x + 15, y + height - 25, LIGHT_GRAY, 12, font_name="Jetbrains Mono")
        self.species_text = arcade.Text("Species ID: -", x + 15, y + height - 50, arcade.color.WHITE, 12, font_name="Jetbrains Mono", bold=True)
        self.nodes_text = arcade.Text("Nodes: -", x + 15, y + height - 75, LIGHT_GRAY, 12, font_name="Jetbrains Mono")
        self.synapses_text = arcade.Text("Synapses: -", x + 15, y + height - 100, LIGHT_GRAY, 12, font_name="Jetbrains Mono")
        self.variance_text = arcade.Text("Env. Variance: -", x + 15, y + height - 125, LIGHT_GRAY, 12, font_name="Jetbrains Mono")
        self.live_species_text = arcade.Text("Alive Species: -", x + 15, y + height - 150, LIGHT_GRAY, 12, font_name="Jetbrains Mono")
        
        self.species_color = (0, 0, 0, 0)
        self.species_id = None

    def update_info(self, network, gen, variance, alive_species):
        self.gen_text.text = f"Generation: {gen}"
        self.species_id = getattr(network, 'species_id', None)
        if self.species_id is not None:
            self.species_text.text = f"Species: {self.species_id}"
            self.species_color = generate_species_color(self.species_id)
        
        self.nodes_text.text = f"Total Nodes: {len(network.neurons)}"
        active_synapses = sum(1 for s in network.connections if s.enabled)
        self.synapses_text.text = f"Active Synapses: {active_synapses}"
        
        self.variance_text.text = f"Env. Variance: {variance:.3f}"
        self.live_species_text.text = f"Alive Species: {alive_species}"

    def draw(self):
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height, (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 25))
        arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)
        self.title_text.draw()
        
        self.gen_text.draw()
        
        if self.species_id is not None:
            pad_x = 5
            pad_y = 2
            arcade.draw_lrbt_rectangle_filled(self.species_text.x - pad_x, 
                                              self.species_text.x + self.species_text.content_width + pad_x, 
                                              self.species_text.y - pad_y, 
                                              self.species_text.y + self.species_text.content_height + pad_y, 
                                              self.species_color)
            
        self.species_text.draw()
        self.nodes_text.draw()
        self.synapses_text.draw()
        self.variance_text.draw()
        self.live_species_text.draw()


class LiveLineChart:
    def __init__(self, x, y, width, height, title="Chart", min_y=0, max_y=60, max_points=100,
                 line_color=ACC1_COLOR, dynamic_y=False, is_reward=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_y = min_y
        self.max_y = max_y
        self.initial_min_y = min_y
        self.initial_max_y = max_y
        self.max_points = max_points
        self.line_color = line_color
        self.dynamic_y = dynamic_y
        self.is_reward = is_reward
        self.base_title = title
        self.data = collections.deque(maxlen=max_points)
        self.total_points_added = 0
        self.grid_x_spacing = max_points // 10

        self.title_text = arcade.Text(title, x + 5, y + height + 5, arcade.color.WHITE, 12, font_name="Jetbrains Mono")
        self.max_text = arcade.Text(f"{max_y}", x - 5, y + height - 15, arcade.color.GRAY, 10, font_name="Jetbrains Mono", anchor_x="right")
        self.min_text = arcade.Text(f"{min_y}", x - 5, y + 5, arcade.color.GRAY, 10, font_name="Jetbrains Mono", anchor_x="right")

    def add_point(self, value):
        self.data.append(value)
        self.total_points_added += 1
        
        if self.is_reward:
            self.title_text.text = f"Cumulative Reward: {value:.1f}"
            
        if self.dynamic_y and len(self.data) > 0:
            cmax = max(self.data)
            cmin = min(self.data)
            self.max_y = max(self.initial_max_y, cmax * 1.1)
            self.min_y = min(self.initial_min_y, cmin * 1.1 if cmin < 0 else max(-0.1, cmin * 0.9))
            self.max_text.text = f"{self.max_y:.1f}"
            self.min_text.text = f"{self.min_y:.1f}"

    def reset(self):
        self.data.clear()
        self.title_text.text = self.base_title
        if self.dynamic_y:
            self.max_y = self.initial_max_y
            self.min_y = self.initial_min_y
            self.max_text.text = f"{self.max_y:.1f}"
            self.min_text.text = f"{self.min_y:.1f}"

    def draw(self):
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height, (DARK_GRAY[0], DARK_GRAY[1], DARK_GRAY[2], 25))
        arcade.draw_lbwh_rectangle_outline(self.x, self.y, self.width, self.height, LIGHT_GRAY, 1)
        
        grid_y_divisions = 6
        for j in range(1, grid_y_divisions):
            py = self.y + (j / grid_y_divisions) * self.height
            arcade.draw_line(self.x, py, self.x + self.width, py, SUBTLE_GRID_COLOR, 1)

        if len(self.data) > 1:
            x_step = self.width / (self.max_points - 1)
            start_index = self.total_points_added - len(self.data)
            for i in range(len(self.data)):
                if (start_index + i) % self.grid_x_spacing == 0:
                    px = self.x + self.width - ((len(self.data) - 1 - i) * x_step)
                    arcade.draw_line(px, self.y, px, self.y + self.height, SUBTLE_GRID_COLOR, 1)

        if self.min_y <= 0 <= self.max_y:
            x_height = (-self.min_y / (self.max_y - self.min_y)) * self.height
            arcade.draw_line(self.x, self.y + x_height, self.x + self.width, self.y + x_height, LIGHT_GRAY, 1)

        self.title_text.draw()
        self.max_text.draw()
        self.min_text.draw()

        if len(self.data) > 1:
            points = []
            x_step = self.width / (self.max_points - 1)
            y_range = self.max_y - self.min_y
            for i, val in enumerate(self.data):
                clamped_val = max(self.min_y, min(self.max_y, val))
                normalized_y = (clamped_val - self.min_y) / y_range if y_range != 0 else 0
                px = self.x + self.width - ((len(self.data) - 1 - i) * x_step)
                py = self.y + (normalized_y * self.height)
                points.append((px, py))
            arcade.draw_line_strip(points, self.line_color, line_width=2)


class TrainingApp(arcade.Window):
    def __init__(self):
        # We start with 1200x900 to ensure charts don't overlap physics and text doesn't scrunch
        super().__init__(1200, 900, "Double Pendulum AI Training", antialiasing=True)
        arcade.set_background_color(BG_COLOR)

        self.sim_thread = None
        self.sim_queue = queue.Queue()
        self.training_running = False
        
        self.gen0_path = "saved_networks/champion_gen_0.pkl" 
        self.env = DoublePendulumEnv(start_var=0.05)
        self.current_obs = self.env.reset()
        self.active_network = None
        self.active_gen_display = 0
        
        self.frame_count = 0
        self.frames_per_cycle = 720 
        self.fps_queue = collections.deque(maxlen=60)
        
        self.main_camera = arcade.Camera2D()
        self.hud_camera = arcade.Camera2D()

        # Dynamic layout for panels
        margin = 35
        panel_w = 350
        panel_h = 160
        y_top = self.height - margin - panel_h - 40 # extra space for the headers
        
        self.gen_chart = GenerationChart(margin + 20, y_top, panel_w - 20, panel_h, title="Generational Fitness")
        self.species_chart = SpeciesChart(margin * 2 + panel_w + 20, y_top, panel_w - 20, panel_h, title="Species Demographics")
        self.net_vis = NetworkVisualizer(margin * 3 + panel_w * 2 + 20, y_top, panel_w - 20, panel_h)
        
        bottom_y = margin
        self.action_chart = LiveLineChart(margin + 20, bottom_y, panel_w - 20, panel_h, title="Agent Engine Force", min_y=-1, max_y=1, max_points=300, line_color=ACC1_COLOR)
        self.reward_chart = LiveLineChart(margin * 2 + panel_w + 20, bottom_y, panel_w - 20, panel_h, title="Real-time Reward", min_y=0, max_y=50, max_points=300, line_color=ACC2_COLOR, dynamic_y=True, is_reward=True)
        self.agent_info = AgentInfoPanel(margin * 3 + panel_w * 2 + 20, bottom_y, panel_w - 20, panel_h)

        self.status_text = arcade.Text("Status: STOPPED", margin, self.height - 35, ACC1_COLOR, 18, font_name="Jetbrains Mono", bold=True)
        self.controls_text = arcade.Text("[SPACE] Start/Stop Training", self.width - 250, self.height - 35, ACC3_COLOR, 14, font_name="Jetbrains Mono", anchor_x="right")
        self.fps_text = arcade.Text("FPS: 0", self.width - 50, self.height - 35, LIGHT_GRAY, 14, font_name="Jetbrains Mono", anchor_x="right")


    def toggle_pause(self):
        global PAUSE_FLAG
        if not self.training_running:
            self.training_running = True
            PAUSE_FLAG = False
            self.status_text.text = "Status: TRAINING (Booting...)"
            self.status_text.color = ACC3_COLOR
            
            seed_net = None
            if self.gen0_path and os.path.exists(self.gen0_path):
                with open(self.gen0_path, "rb") as f:
                    seed_net = pickle.load(f)[0]
            if seed_net:
                self.env = DoublePendulumEnv(start_var=0.05)
                self.current_obs = self.env.reset()

            global KILL_FLAG
            KILL_FLAG = False
            
            self.sim_thread = threading.Thread(target=runner_loop, args=(self.sim_queue, seed_net,), daemon=True)
            self.sim_thread.start()
        else:
            PAUSE_FLAG = not PAUSE_FLAG
            if PAUSE_FLAG:
                self.status_text.text = "Status: PAUSED"
                self.status_text.color = arcade.color.YELLOW
            else:
                self.status_text.text = "Status: TRAINING (Resumed)"
                self.status_text.color = ACC3_COLOR

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.SPACE:
            self.toggle_pause()

    def on_update(self, delta_time):
        if delta_time > 0:
            self.fps_queue.append(1.0 / delta_time)
            
        newest_update = None
        while not self.sim_queue.empty():
            newest_update = self.sim_queue.get()
            
            if isinstance(newest_update, str):
                if newest_update == "STOPPED":
                    self.status_text.text = "Status: STOPPED"
                continue

            max_f = newest_update['max_f']
            p75_f = newest_update.get('p75_f', max_f)
            med_f = newest_update['med_f']
            p25_f = newest_update.get('p25_f', med_f)
            min_f = newest_update.get('min_f', med_f)
            
            var_up = newest_update['var_inc']
            s_dict = newest_update['species_sizes']
            gen = newest_update['gen']
            current_variance = newest_update['variance']
            alive_count = newest_update['alive_species']
            
            self.status_text.text = f"Status: Training Generation {gen}"
            
            self.gen_chart.add_data(max_f, p75_f, med_f, p25_f, min_f, gen, var_up)
            self.species_chart.add_data(s_dict)
            self.active_network_buffer = newest_update['network']
            self.active_gen_buffer = gen
            self.active_var_buffer = current_variance
            self.active_alive_buffer = alive_count

        self.frame_count += 1
        
        if self.frame_count >= self.frames_per_cycle or self.active_network is None:
            self.frame_count = 0
            if hasattr(self, 'active_network_buffer') and self.active_network_buffer is not None:
                self.active_network = self.active_network_buffer
                self.active_gen_display = self.active_gen_buffer
                self.net_vis.update_network(self.active_network)
                self.agent_info.update_info(self.active_network, self.active_gen_display, self.active_var_buffer, self.active_alive_buffer)
            
            self.env = DoublePendulumEnv(start_var=0.0) 
            self.current_obs = self.env.reset()
            # Cancel the instant velocity that happens from 0.0 start_var pushing down the pendulum to prevent haunted swing
            self.env.bob1.v = Vec(0,0)
            self.env.bob2.v = Vec(0,0)
            self.action_chart.reset()
            self.reward_chart.reset()

        if self.active_network:
            force = self.active_network.forward_pass(self.current_obs)
            self.current_obs, reward, _ = self.env.step(force, 1/60.0)
            # Add action to LiveLineChart 
            self.action_chart.add_point(force)
            self.reward_chart.add_point(reward)


    def on_draw(self):
        self.clear()

        if len(self.fps_queue) > 0:
            avg_fps = sum(self.fps_queue) / len(self.fps_queue)
            self.fps_text.text = f"FPS: {avg_fps:.0f}"

        with self.main_camera.activate():
            x_offset = (self.width / 2) - (SCREEN_WIDTH * PPM / 2)
            y_offset = (self.height / 2) - (SCREEN_HEIGHT * PPM / 2) - 100 # Drop camera 100px so it doesn't overlap UI
            
            def adjusted_x(x_meters): return x_meters * PPM + x_offset
            def adjusted_y(y_meters): return y_meters * PPM + y_offset

            draw_track_adj(TRACK_HEIGHT, TRACK_LENGTH, x_offset, y_offset)
            
            cart_x = adjusted_x(self.env.cart.s.x)
            cart_y = adjusted_y(self.env.cart.s.y)
            arcade.draw_circle_filled(cart_x, cart_y, self.env.cart.radius_m * PPM, ACC1_COLOR)
            
            b1_x = adjusted_x(self.env.bob1.s.x)
            b1_y = adjusted_y(self.env.bob1.s.y)
            b2_x = adjusted_x(self.env.bob2.s.x)
            b2_y = adjusted_y(self.env.bob2.s.y)
            
            arcade.draw_line(cart_x, cart_y, b1_x, b1_y, LIGHT_GRAY, 4)
            arcade.draw_line(b1_x, b1_y, b2_x, b2_y, LIGHT_GRAY, 4)

            for b, bx, by in [(self.env.bob1, b1_x, b1_y), (self.env.bob2, b2_x, b2_y)]:
                arcade.draw_circle_filled(bx, by, b.radius_m * PPM, ACC1_COLOR)

        with self.hud_camera.activate():
            self.gen_chart.draw()
            self.species_chart.draw()
            self.net_vis.draw()
            
            self.action_chart.draw()
            self.reward_chart.draw()
            self.agent_info.draw()
            
            self.status_text.draw()
            self.controls_text.draw()
            self.fps_text.draw()


def draw_vec_adj(tail_x, tail_y, vec_x, vec_y, color, thickness=2, x_offset=0, y_offset=0):
    start_x = tail_x * PPM + x_offset
    start_y = tail_y * PPM + y_offset
    end_x = (tail_x + vec_x) * PPM + x_offset
    end_y = (tail_y + vec_y) * PPM + y_offset
    arcade.draw_line(start_x, start_y, end_x, end_y, color, thickness)

def draw_track_adj(height, length, x_offset, y_offset):
    start_x = (SCREEN_WIDTH / 2 - length / 2) * PPM + x_offset
    end_x = (SCREEN_WIDTH / 2 + length / 2) * PPM + x_offset
    cy = height * PPM + y_offset
    arcade.draw_line(start_x, cy, end_x, cy, DARK_GRAY, 2)

    # meter ticks from double.py
    tx_start = (SCREEN_WIDTH / 2 - length / 2)
    tx_end = (SCREEN_WIDTH / 2 + length / 2)
    draw_vec_adj(tx_start, height - 0.25, 0, 0.5, DARK_GRAY, 2, x_offset, y_offset)
    draw_vec_adj(tx_end, height - 0.25, 0, 0.5, DARK_GRAY, 2, x_offset, y_offset)
    
    for i in range(1, length):
        draw_vec_adj(tx_start + i, height - 0.1, 0, 0.2, DARK_GRAY, 2, x_offset, y_offset)


# =========================================================================================
# BACKGROUND TRAINING THREAD LOGIC
# =========================================================================================
KILL_FLAG = False
PAUSE_FLAG = False

def runner_loop(ui_queue, seed_network):
    global KILL_FLAG, PAUSE_FLAG
    inno_tracker = InnovationManager()
    
    population = []
    if seed_network:
        population = [seed_network.clone() for _ in range(POPULATION)]
        for p in population: p.mutate_weights()
    else:
        population = [gen0_network() for _ in range(POPULATION)]
        
    steps = int(SIM_TIME / DT)

    if not os.path.exists("saved_networks"):
        os.makedirs("saved_networks")

    current_variance = 0.05
    compatibility_threshold = COMPATIBILITY_THRESHOLD

    optimal_workers = max(1, os.cpu_count() - 2)
    next_species_id = 0
    species_reps = {}

    # Multiprocessing inside the Thread!
    with concurrent.futures.process.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        for generation in range(GENERATIONS + 1):
            if KILL_FLAG: break
            while PAUSE_FLAG:
                time.sleep(0.1)
                if KILL_FLAG: break
            if KILL_FLAG: break
            
            gen_seed = random.randint(0, 100000000)
            eval_func = partial(evaluate_single_network, run_steps=steps,
                                generation_seed=gen_seed,
                                start_var=current_variance)
            flat_pop = [bench.export_flat() for bench in population]
            
            eval_results = list(executor.map(eval_func, flat_pop, chunksize=16))

            for i in range(POPULATION):
                population[i].fitness, population[i].frames = eval_results[i]

            # Speciation Continuity Fix
            species_members = collections.defaultdict(list)
            
            for network in population:
                found_species = False
                for s_idx, rep in species_reps.items():
                    if network.distance_to(rep) < compatibility_threshold:
                        species_members[s_idx].append(network)
                        network.species_id = s_idx
                        found_species = True
                        break
                if not found_species:
                    s_idx = next_species_id
                    next_species_id += 1
                    species_reps[s_idx] = network
                    species_members[s_idx].append(network)
                    network.species_id = s_idx

            # Update species representatives for the next generation to be a random 
            # member of the newly formed species to prevent drifting
            new_species_reps = {}
            for s_idx, members in species_members.items():
                if members:
                    new_species_reps[s_idx] = random.choice(members)
            species_reps = new_species_reps

            # The new optimal target for 256 agents
            target_species = 20

            # Give it a slightly wider buffer (don't panic if it's 18 or 22)
            species_diff = abs(len(species_reps) - target_species)

            # Proportional step sizes
            step_size = 1.0 if species_diff > 3 else 0.5

            if len(species_reps) > target_species + 2:
                compatibility_threshold += step_size
            elif len(species_reps) < target_species - 2:
                compatibility_threshold -= step_size

            # Keep the floor around 2.0 to prevent the TV static!
            compatibility_threshold = min(max(0.5, compatibility_threshold), 20.0)

            # Calculate adjusted fitness
            for network in population:
                species_size = len(species_members[network.species_id])
                network.adjusted_fitness = network.fitness / species_size

            population.sort(key=lambda n: n.adjusted_fitness, reverse=True)
            best_raw = max(population, key=lambda n: n.fitness)
            good_performer_raw = sorted([n.frames for n in population], reverse=True)[int(0.10 * POPULATION)]

            var_inc = False
            if good_performer_raw > NEXT_STAGE_CUTOFF:
                current_variance = min(current_variance + CURRICULUM_STEP, 1.0)
                var_inc = True

            species_sizes = {s_idx: len(members) for s_idx, members in species_members.items()}

            # Sort by raw fitness for accurate quartile distribution
            raw_sorted = sorted(population, key=lambda n: n.fitness, reverse=True)
            
            # PUSH TO UI
            ui_queue.put({
                'max_f': raw_sorted[0].fitness,
                'p75_f': raw_sorted[int(0.25 * POPULATION)].fitness,
                'med_f': raw_sorted[int(0.50 * POPULATION)].fitness,
                'p25_f': raw_sorted[int(0.75 * POPULATION)].fitness,
                'min_f': raw_sorted[-1].fitness,
                'gen': generation,
                'var_inc': var_inc,
                'species_sizes': species_sizes,
                'network': best_raw.clone(),
                'variance': current_variance,
                'alive_species': len(species_members)
            })

            # Checkpoint Save
            if generation % 100 == 0:
                with open(f"saved_networks/champion_gen_{generation}.pkl", "wb") as f:
                    pickle.dump([best_raw], f)

            # Breed Next Gen
            # MUST SUM adjusted fitness. Standard NEAT uses proportional sum to dictate slot density.
            # Using simple average double-penalizes successful species sizes and causes ping-ponging!
            species_sum_adj_fitness = {}
            for s_idx, members in species_members.items():
                species_sum_adj_fitness[s_idx] = sum(m.adjusted_fitness for m in members)

            total_adj_fitness = sum(species_sum_adj_fitness.values())
            
            species_slots = {}
            fractional_parts = {}
            
            for s_idx, sum_fit in species_sum_adj_fitness.items():
                exact = (sum_fit / total_adj_fitness) * POPULATION if total_adj_fitness > 0 else POPULATION / len(species_members)
                species_slots[s_idx] = int(exact)
                fractional_parts[s_idx] = exact - int(exact)
                
            remaining = POPULATION - sum(species_slots.values())
            for s_idx, _ in sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)[:remaining]:
                species_slots[s_idx] += 1

            next_gen_elites = []
            next_gen_children = []
            
            for s_idx, slots in species_slots.items():
                if slots == 0: continue
                members = species_members[s_idx]
                members.sort(key=lambda n: n.adjusted_fitness, reverse=True)
                
                clones_to_make = min(2, len(members), slots)
                for i in range(clones_to_make):
                    next_gen_elites.append(members[i].clone())
                    
                slots_remaining = slots - clones_to_make
                if slots_remaining <= 0: continue

                mating_pool_size = max(1, len(members) // 2)
                mating_pool = members[:mating_pool_size]

                pool_fitness = [m.adjusted_fitness for m in mating_pool]
                min_fit = min(pool_fitness)
                weights = [(f - min_fit + 0.001) for f in pool_fitness] 
                
                while slots_remaining > 0:
                    if random.random() < 0.75 and len(mating_pool) > 1:
                        parent1, parent2 = random.choices(mating_pool, weights=weights, k=2)
                        attempts = 0
                        while parent1 is parent2 and attempts < 10:
                             parent2 = random.choices(mating_pool, weights=weights, k=1)[0]
                             attempts += 1
                        child = Network.crossover(parent1, parent2)
                    else:
                        parent = random.choices(mating_pool, weights=weights, k=1)[0]
                        child = parent.clone()
                        
                    next_gen_children.append(child)
                    slots_remaining -= 1

            next_gen = next_gen_elites + next_gen_children

            inno_tracker.start_new_generation()

            for network in next_gen_children:
                network.mutate_weights()
                if random.random() <= 0.05: network.mutate_add_neuron(inno_tracker)
                if random.random() <= 0.1:  network.mutate_add_synapse(inno_tracker)

            population = next_gen

    if KILL_FLAG:
        print("Thread elegantly halted via SPAceBAR KILL_FLAG.")
    ui_queue.put("STOPPED")


if __name__ == "__main__":
    app = TrainingApp()
    arcade.run()
