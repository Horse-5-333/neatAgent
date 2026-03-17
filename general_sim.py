import arcade
import collections
import time
import random

from arcade.color import WHITE
from pyglet.event import EVENT_HANDLE_STATE

# Import our decoupled physics engine
from physics import Vec, PtMass, Wall, G, resolve_collision

SCREEN_WIDTH = 16  # meters
SCREEN_HEIGHT = 9
PPM = 80  # pixels per meter

GLOBAL_FRICTION_MULTIPLIER = 0.05

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

        self.title_text = arcade.Text(title, x + 5, y + height - 25, arcade.color.WHITE, 14, font_name="Jetbrains Mono")
        self.max_text = arcade.Text(f"{max_y}", x - 35, y + height - 10, arcade.color.GRAY, 12,
                                    font_name="Jetbrains Mono")
        self.min_text = arcade.Text(f"{min_y}", x - 35, y, arcade.color.GRAY, 12, font_name="Jetbrains Mono")

    def add_point(self, value):
        self.data.append(value)

    def draw(self):
        arcade.draw_lbwh_rectangle_filled(self.x, self.y, self.width, self.height, (30, 30, 30, 200))
        arcade.draw_line(self.x, self.y, self.x + self.width, self.y, arcade.color.GRAY, 2)
        arcade.draw_line(self.x, self.y, self.x, self.y + self.height, arcade.color.GRAY, 2)

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
                px = self.x + (i * x_step)
                py = self.y + (normalized_y * self.height)
                points.append((px, py))

            arcade.draw_line_strip(points, self.line_color, line_width=2)


class PhysicsSimulator(arcade.Window):
    def __init__(self):
        super().__init__(int(SCREEN_WIDTH * PPM), int(SCREEN_HEIGHT * PPM), "Kinematics Simulator")
        arcade.set_background_color(arcade.color.BLACK)


        self.cart = PtMass(mass=10.0, s=Vec(8, 5))
        self.bob = PtMass(mass=2.0, s=Vec(8, 1))

        self.scene_masses = [self.cart, self.bob]
        # Walls configured with inward-pointing normals
        # self.scene_walls = [
        #     Wall(start_pt=Vec(1, 1), end_pt=Vec(15, 1)),  # bottom
        #     Wall(start_pt=Vec(1, 7), end_pt=Vec(1, 1)),  # left
        #     Wall(start_pt=Vec(15, 1), end_pt=Vec(15, 7))  # right
        # ]
        self.frame_times_ms = collections.deque(maxlen=600)

        self.main_camera = arcade.Camera2D()
        self.text_camera = arcade.Camera2D()
        self.text_camera.zoom = 0.5
        self.text_camera.bottom_left = (0, 0)

        self.perf_text = arcade.Text(
            text="frame time: 0.00 ms\nobjects: 0",
            x=20,
            y=(self.height * 2) - 50,
            color=arcade.color.WHITE,
            font_size=24,
            font_name="Jetbrains Mono",
            multiline=True,
            width=800
        )

        self.fps_chart = LiveLineChart(
            x=450,
            y=(self.height * 2) - 120,
            width=500,
            height=100,
            title="Frame Time (ms)",
            min_y=0,
            max_y=20,
            max_points=600,
            line_color=(255, 255, 255)
        )

        self.left_pressed = False
        self.right_pressed = False
        self.motor_force = 150.0

    # def on_mouse_press(self, x, y, button, modifiers):
    #     mouse_pos_m = Vec(x, y) / PPM
    #     self.scene_masses.append(PtMass(
    #         mass=random.uniform(0.1, 5),
    #         radius_m=random.uniform(0.05, .5),
    #         s=mouse_pos_m,
    #         v=Vec(random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0))
    #     ))
    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.A:
            self.left_pressed = True
        elif symbol == arcade.key.D:
            self.right_pressed = True

    def on_key_release(self, symbol, modifiers):
        if symbol == arcade.key.A:
            self.left_pressed = False
        elif symbol == arcade.key.D:
            self.right_pressed = False


    def on_update(self, delta_time):
        cycle_start_time = time.perf_counter()

        delta_time = min(delta_time, 1/60.0)

        sub_steps = 3
        sub_dt = delta_time / sub_steps

        for _ in range(sub_steps):
            # user input
            if self.left_pressed:
                self.cart.apply_force(Vec(-self.motor_force, 0))
            if self.right_pressed:
                self.cart.apply_force(Vec(self.motor_force, 0))

            # gravity on bob
            self.bob.apply_force(Vec(0, -G * self.bob.mass))

            # air drag on bob
            drag_coeff = 0.05 * GLOBAL_FRICTION_MULTIPLIER
            v_mag = self.bob.v.magnitude
            drag_force = self.bob.v * (-drag_coeff * v_mag)
            self.bob.apply_force(drag_force)

            # trac friction
            rolling_friction = 2.0 * GLOBAL_FRICTION_MULTIPLIER
            self.cart.apply_force(self.cart.v * -rolling_friction)

            # joint friction
            rod_vec = self.bob.s - self.cart.s
            rod_dir = rod_vec.normalized()

            # find tangent to rod
            tangent = Vec(-rod_dir.y, rod_dir.x)

            # Get velocity of bob relative to cart, projected onto the swing arc
            rel_v = self.bob.v - self.cart.v
            swing_speed = rel_v.dot(tangent)

            # Apply friction opposing the swing
            joint_friction_coeff = 0.8 * GLOBAL_FRICTION_MULTIPLIER
            joint_friction = tangent * (-joint_friction_coeff * swing_speed)

            self.bob.apply_force(joint_friction)
            self.cart.apply_force(joint_friction * -1)  # Newton's Third Law

            # 6. The Rigid Pole Constraint
            rest_length = 4.0
            current_length = rod_vec.magnitude

            # Positive if compressed, negative if stretched
            overlap = rest_length - current_length

            # Massive stiffness (k) keeps it acting like steel, not a bungee cord
            k = 10000
            spring_force_mag = k * overlap

            # Heavy damping (d) prevents the rod from vibrating like a guitar string
            d = 100
            v_normal = rel_v.dot(rod_dir)
            damping_force_mag = -d * v_normal

            # The total tension/compression force of the pole
            total_rod_force = rod_dir * (spring_force_mag + damping_force_mag)

            # The pole pulls/pushes both objects equally
            self.cart.apply_force(total_rod_force * -1)
            self.bob.apply_force(total_rod_force)

            # lock cart accel to rail
            self.cart.a.y = 0.0

            self.cart.update(sub_dt)
            self.bob.update(sub_dt)

            #lock to rail
            self.cart.s.y = 5.0
            self.cart.v.y = 0.0


        # TELEMETRY

        frame_time_ms = (time.perf_counter() - cycle_start_time) * 1000
        self.frame_times_ms.append(frame_time_ms)
        self.fps_chart.add_point(frame_time_ms)

        sorted_times = sorted(self.frame_times_ms)
        high_cutoff = sorted_times[(99 * len(sorted_times)) // 100]

        self.perf_text.text = f"frame time: {frame_time_ms:.2f} ms \n99% high: {high_cutoff:.2f} ms\nobjects: {len(self.scene_masses)}"

    def on_draw(self):
        self.clear()

        with self.main_camera.activate():
            for pt_mass in self.scene_masses:
                pos_p = pt_mass.s * PPM
                radius_p = pt_mass.radius_m * PPM
                arcade.draw_circle_filled(pos_p.x, pos_p.y, radius_p, arcade.color.WHITE)

            # for wall in self.scene_walls:
            #     x0 = wall.start.x * PPM
            #     x1 = wall.end.x * PPM
            #     y0 = wall.start.y * PPM
            #     y1 = wall.end.y * PPM
            #     arcade.draw_line(x0, y0, x1, y1, arcade.color.WHITE, 2)


            arcade.draw_line(self.cart.s.x * PPM,
                             self.cart.s.y * PPM,
                             self.bob.s.x * PPM,
                             self.bob.s.y * PPM,
                             (255, 255, 255),
                             5)
        with self.text_camera.activate():
            self.perf_text.draw()
            self.fps_chart.draw()


if __name__ == "__main__":
    window = PhysicsSimulator()
    arcade.run()