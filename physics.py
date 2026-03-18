import math
import random
from numba import njit
from numba.parfors.parfor_lowering import replace_var_with_array_in_block

SCREEN_WIDTH = 14  # meters
SCREEN_HEIGHT = 16
PPM = 50 # pixels per meter
EPSILON = 1e-6
TRACK_HEIGHT = 10
TRACK_LENGTH = 8
MAX_FORCE = 600 # N

GRAVITY = 9.81
FRICTION_MULT = 0.05

class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __mul__(self, scalar):
        return Vec(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vec(self.x / scalar, self.y / scalar)

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def normalized(self):
        mag = self.magnitude
        if mag < EPSILON:
            return Vec(0, 0)
        return Vec(self.x/mag, self.y/mag)

    @property
    def magnitude(self):
        return math.hypot(self.x, self.y)

    @property
    def magnitude_squared(self):
        return self.x ** 2 + self.y ** 2

    def project(self, onto):
        return (self.dot(onto) / (onto.magnitude ** 2)) * onto

    def project_unit(self, onto):
        return (self.dot(onto)) * onto

    @property
    def perpendicular_normalized(self):
        return Vec(-self.normalized().y, self.normalized().x)

class Mat2:
    def __init__(self, a00, a01, a10, a11):
        self.a00 = a00
        self.a01 = a01
        self.a10 = a10
        self.a11 = a11

    @classmethod
    def rotation(cls, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return cls(c, -s, s, c)

    @classmethod
    def scale(cls, s, sy=None):
        sy = sy if sy is not None else s
        return cls(s, 0, 0, sy)

    def __mul__(self, vec):
        return Vec(
            self.a00 * vec.x + self.a01 * vec.y,
            self.a10 * vec.x + self.a11 * vec.y
        )

class PtMass:
    def __init__(self, mass=1.0, s=None, v=None, a=None, radius_m=0.25):
        self.mass = mass
        self.s = s if s is not None else Vec(0, 0)
        self.v = v if v is not None else Vec(0, 0)
        self.a = a if a is not None else Vec(0, 0)
        self.radius_m = radius_m

    def apply_force(self, f):
        self.a += f / self.mass

    def update(self, dt):
        self.v.x += self.a.x * dt
        self.v.y += self.a.y * dt
        self.s.x += self.v.x * dt
        self.s.y += self.v.y * dt
        self.a.x = 0.0
        self.a.y = 0.0

class Wall:
    def __init__(self, start_pt, end_pt):
        self.start = start_pt
        self.end = end_pt
        self.span = end_pt - start_pt

        self.length = self.span.magnitude
        self.dir = self.span.normalized()
        self.norm = Vec(-self.span.y, self.span.x).normalized()


def resolve_collision(m1, m2):
    # 1. Quick distance check using squared length for performance
    collision_vec = m1.s - m2.s
    dist_squared = collision_vec.magnitude_squared
    min_dist = m1.radius_m + m2.radius_m

    # If overlapping and not occupying the exact same mathematical point
    if 0 < dist_squared < min_dist ** 2:
        dist = math.sqrt(dist_squared)
        overlap = min_dist - dist

        # The unit vector pointing from m2 to m1
        collision_normal = collision_vec / dist

        # --- STEP 1: POSITIONAL CORRECTION ---
        # Push them apart based on their mass ratio so heavy objects move less
        total_mass = m1.mass + m2.mass

        # Multiply by a slight damper (e.g., 0.8) to prevent jittering in tight piles
        correction_factor = 0.8
        m1.s += collision_normal * (overlap * (m2.mass / total_mass) * correction_factor)
        m2.s -= collision_normal * (overlap * (m1.mass / total_mass) * correction_factor)

        # --- STEP 2: IMPULSE RESOLUTION ---
        relative_velocity = m1.v - m2.v
        vel_along_normal = relative_velocity.dot(collision_normal)

        # If they are already moving apart, do not apply bounce
        if vel_along_normal > 0:
            return

        # Restitution (0.0 = clay, 1.0 = perfectly bouncy)
        restitution = 0.3

        # Calculate the scalar impulse (j) using 1D conservation of momentum
        j = -(1 + restitution) * vel_along_normal
        j /= (1 / m1.mass + 1 / m2.mass)

        # Apply the impulse vector directly to the velocities
        impulse_vec = collision_normal * j
        m1.v += impulse_vec / m1.mass
        m2.v -= impulse_vec / m2.mass


@njit
def fast_physics_step(action_force, dt, gravity, friction_multiplier,
                      cart_x, cart_v_x, cart_mass,
                      b1_x, b1_y, b1_vx, b1_vy, b1_mass, b1_rest,
                      b2_x, b2_y, b2_vx, b2_vy, b2_mass, b2_rest,
                      start_var):
    sub_steps = 3
    sub_dt = dt / sub_steps

    SCREEN_WIDTH_HALF = 7.0
    TRACK_LENGTH_HALF = 4.0
    TRACK_HEIGHT = 10.0
    MAX_FORCE = 600.0
    EPSILON = 1e-6

    for _ in range(sub_steps):
        cart_ax = (MAX_FORCE * action_force) / cart_mass
        b1_ax = 0.0
        b1_ay = -gravity
        b2_ax = 0.0
        b2_ay = -gravity

        if SCREEN_WIDTH_HALF - TRACK_LENGTH_HALF > cart_x:
            cart_x = SCREEN_WIDTH_HALF - TRACK_LENGTH_HALF
            cart_v_x = -1.0 * cart_v_x
        if SCREEN_WIDTH_HALF + TRACK_LENGTH_HALF < cart_x:
            cart_x = SCREEN_WIDTH_HALF + TRACK_LENGTH_HALF
            cart_v_x = -1.0 * cart_v_x

        drag_coeff = 0.05 * friction_multiplier

        b1_v_mag = math.hypot(b1_vx, b1_vy)
        b1_ax -= drag_coeff * b1_v_mag * b1_vx / b1_mass
        b1_ay -= drag_coeff * b1_v_mag * b1_vy / b1_mass

        b2_v_mag = math.hypot(b2_vx, b2_vy)
        b2_ax -= drag_coeff * b2_v_mag * b2_vx / b2_mass
        b2_ay -= drag_coeff * b2_v_mag * b2_vy / b2_mass

        rolling_friction = 2.0 * friction_multiplier
        cart_ax -= (rolling_friction * cart_v_x) / cart_mass

        # Joint Friction (Cart & Bob 1)
        rod_x = b1_x - cart_x
        rod_y = b1_y - TRACK_HEIGHT
        rod_mag = math.hypot(rod_x, rod_y)

        if rod_mag < EPSILON:
            rod_dir_x, rod_dir_y = 0.0, 1.0
        else:
            rod_dir_x, rod_dir_y = rod_x / rod_mag, rod_y / rod_mag
            
        tangent_x, tangent_y = -rod_dir_y, rod_dir_x

        rel_v_x = b1_vx - cart_v_x
        rel_v_y = b1_vy - 0.0
        swing_speed = rel_v_x * tangent_x + rel_v_y * tangent_y

        joint_friction_coeff = 0.8 * friction_multiplier
        joint_f_x = tangent_x * (-joint_friction_coeff * swing_speed)
        joint_f_y = tangent_y * (-joint_friction_coeff * swing_speed)

        b1_ax += joint_f_x / b1_mass
        b1_ay += joint_f_y / b1_mass
        cart_ax -= joint_f_x / cart_mass

        # Constraint 1: Cart to Bob
        overlap = b1_rest - rod_mag

        k = 10000.0
        d = 100.0

        spring_force_mag = k * overlap
        damping_force_mag = -d * (rel_v_x * rod_dir_x + rel_v_y * rod_dir_y)

        total_rod_force_x = rod_dir_x * (spring_force_mag + damping_force_mag)
        total_rod_force_y = rod_dir_y * (spring_force_mag + damping_force_mag)

        cart_ax -= total_rod_force_x / cart_mass
        b1_ax += total_rod_force_x / b1_mass
        b1_ay += total_rod_force_y / b1_mass

        # Constraint 2: Bob to Bob2
        rod2_x = b2_x - b1_x
        rod2_y = b2_y - b1_y
        rod2_mag = math.hypot(rod2_x, rod2_y)

        if rod2_mag < EPSILON:
            rod2_dir_x, rod2_dir_y = 0.0, 1.0
        else:
            rod2_dir_x, rod2_dir_y = rod2_x / rod2_mag, rod2_y / rod2_mag

        rel_v2_x = b2_vx - b1_vx
        rel_v2_y = b2_vy - b1_vy

        overlap2 = b2_rest - rod2_mag

        spring_force_mag2 = k * overlap2
        damping_force_mag2 = -d * (rel_v2_x * rod2_dir_x + rel_v2_y * rod2_dir_y)

        total_rod_force2_x = rod2_dir_x * (spring_force_mag2 + damping_force_mag2)
        total_rod_force2_y = rod2_dir_y * (spring_force_mag2 + damping_force_mag2)

        b1_ax -= total_rod_force2_x / b1_mass
        b1_ay -= total_rod_force2_y / b1_mass
        b2_ax += total_rod_force2_x / b2_mass
        b2_ay += total_rod_force2_y / b2_mass

        # Integration
        cart_v_x += cart_ax * sub_dt
        cart_x += cart_v_x * sub_dt

        b1_vx += b1_ax * sub_dt
        b1_vy += b1_ay * sub_dt
        b1_x += b1_vx * sub_dt
        b1_y += b1_vy * sub_dt

        b2_vx += b2_ax * sub_dt
        b2_vy += b2_ay * sub_dt
        b2_x += b2_vx * sub_dt
        b2_y += b2_vy * sub_dt

    # Calculate rewards and obs
    cart_obs_x = (2.0 * cart_x - 14.0) / 8.0
    cart_obs_v = math.tanh(cart_v_x / 6.0)

    rod1_x = b1_x - cart_x
    rod1_y = b1_y - TRACK_HEIGHT
    theta = math.atan2(rod1_y, rod1_x) / math.pi
    v = math.tanh(math.hypot(b1_vx, b1_vy) / 6.0)

    rod2_x = b2_x - b1_x
    rod2_y = b2_y - b1_y
    phi = math.atan2(rod2_y, rod2_x) / math.pi
    w = math.tanh(math.hypot(b2_vx, b2_vy) / 6.0)

    if b2_y > TRACK_HEIGHT + b1_rest + b2_rest - 0.5:
        reward = +2.0
    elif b2_y > TRACK_HEIGHT + b1_rest:
        reward = +1.0
    elif b2_y > TRACK_HEIGHT:
        reward = 0.0
    elif b2_y > TRACK_HEIGHT - (b1_rest / 2.0):
        reward = -0.5
    else:
        reward = -2.0

    frame = False
    if b1_y > TRACK_HEIGHT and b2_y > TRACK_HEIGHT + (b1_rest / 2.0):
        frame = True

    if cart_obs_x < -0.90 or cart_obs_x > 0.90:
        reward = -15.0

    return (cart_x, cart_v_x, 
            b1_x, b1_y, b1_vx, b1_vy, 
            b2_x, b2_y, b2_vx, b2_vy, 
            cart_obs_x, cart_obs_v, theta, v, phi, w, 
            reward, frame)

class DoublePendulumEnv:
    def __init__(self, start_var=0):
        self.start_var = start_var
        self.bob2 = None
        self.bob1 = None
        self.bob2_rest_length = None
        self.bob1_rest_length = None
        self.cart = None
        self.height_pts = 0.0

    def reset(self):
        self.height_pts = 0.0
        track_home = Vec(SCREEN_WIDTH / 2, TRACK_HEIGHT)

        side = random.choice([-1.0, 1.0])
        base_angle_shift = self.start_var * math.pi * side
        base_angle = 0.5 * math.pi + base_angle_shift
        
        theta = [base_angle + random.uniform(-0.1, 0.1), base_angle + random.uniform(-0.1, 0.1)]

        vel = [random.uniform(-3, 3), random.uniform(-5, 5)]
        vel = [self.start_var * v for v in vel]

        self.bob1_rest_length = 3.0
        self.bob2_rest_length = 3.0
        bob1_start_s = track_home + self.bob1_rest_length * Vec(math.cos(theta[0]), math.sin(theta[0]))
        bob2_start_s = bob1_start_s + self.bob2_rest_length * Vec(math.cos(theta[1]), math.sin(theta[1]))
        bob1_start_v = vel[0] * Vec(-math.sin(theta[0]), math.cos(theta[0]))
        bob2_start_v = bob1_start_v + vel[1] * Vec(-math.sin(theta[1]), math.cos(theta[1]))


        self.cart = PtMass(mass=15.0, s=track_home, radius_m=0.3)
        self.bob1 = PtMass(mass=5.0, s=bob1_start_s, v=bob1_start_v, radius_m=0.2)
        self.bob2 = PtMass(mass=4.0, s=bob2_start_s, v=bob2_start_v, radius_m=0.2)

        return self.observations()

    def step(self, action_force, dt=1/60.0):
        (self.cart.s.x, self.cart.v.x,
         self.bob1.s.x, self.bob1.s.y, self.bob1.v.x, self.bob1.v.y,
         self.bob2.s.x, self.bob2.s.y, self.bob2.v.x, self.bob2.v.y,
         obs0, obs1, obs2, obs3, obs4, obs5, reward, frame) = fast_physics_step(
             action_force, dt, GRAVITY, FRICTION_MULT,
             self.cart.s.x, self.cart.v.x, self.cart.mass,
             self.bob1.s.x, self.bob1.s.y, self.bob1.v.x, self.bob1.v.y, self.bob1.mass, self.bob1_rest_length,
             self.bob2.s.x, self.bob2.s.y, self.bob2.v.x, self.bob2.v.y, self.bob2.mass, self.bob2_rest_length,
             self.start_var
         )
         
        self.cart.s.y = 10.0
        self.cart.v.y = 0.0

        self.height_pts += reward
        if self.height_pts < 0.0:
            self.height_pts = 0.0

        final_reward = self.height_pts

        if obs0 <= -0.85 or obs0 >= 0.85:
            final_reward -= 5.0
            self.height_pts = max(0.0, self.height_pts - 1.0)

        return [obs0, obs1, obs2, obs3, obs4, obs5], final_reward, frame

    def observations(self):
        cart_x = (2*self.cart.s.x - SCREEN_WIDTH)/TRACK_LENGTH
        cart_v = math.tanh(self.cart.v.x / 6.0)  # attempt to give maximum info while normalize into [-1, 1]

        rod1 = self.bob1.s - self.cart.s
        theta = math.atan2(rod1.y, rod1.x) / math.pi
        v = math.tanh(self.bob1.v.magnitude / 6.0)

        rod2 = self.bob2.s - self.bob1.s
        phi = math.atan2(rod2.y, rod2.x) / math.pi
        w = math.tanh(self.bob2.v.magnitude / 6.0)

        return [cart_x, cart_v, theta, v, phi, w]

