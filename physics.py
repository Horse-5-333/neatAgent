import math
import random


SCREEN_WIDTH = 14  # meters
SCREEN_HEIGHT = 16
PPM = 50 # pixels per meter
EPSILON = 1e-6
TRACK_HEIGHT = 10
TRACK_LENGTH = 8
MAX_FORCE = 600 # N

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


class DoublePendulumEnv:
    def __init__(self, gravity=9.81*0.8, friction_multiplier=0.2):
        self.gravity = gravity
        self.friction_multiplier = friction_multiplier
        self.bob2 = None
        self.bob1 = None
        self.bob2_rest_length = None
        self.bob1_rest_length = None
        self.cart = None

    def reset(self):
        track_home = Vec(SCREEN_WIDTH / 2, TRACK_HEIGHT)

        # TODO: randomized starts of each bob to prevent overfitting
        theta = [random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)]
        # theta = [1 * math.pi / 2, 1 * math.pi / 2]
        # TODO: randomized velocities tangent to direction
        vel = [random.uniform(-3, 3), random.uniform(-5, 5)]

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
        sub_steps = 3
        sub_dt = dt / sub_steps

        for _ in range(sub_steps):
            self.cart.apply_force(MAX_FORCE * Vec(action_force, 0))
            # check if cart is off the track
            if SCREEN_WIDTH / 2 - TRACK_LENGTH / 2 > self.cart.s.x:
                self.cart.s.x = SCREEN_WIDTH / 2 - TRACK_LENGTH / 2
                self.cart.v.x = -1 * self.cart.v.x
            if SCREEN_WIDTH / 2 + TRACK_LENGTH / 2 < self.cart.s.x:
                self.cart.s.x = SCREEN_WIDTH / 2 + TRACK_LENGTH / 2
                self.cart.v.x = -1 * self.cart.v.x

            # --- GRAVITY & DRAG ---
            drag_coeff = 0.05 * self.friction_multiplier

            # Bob 1
            self.bob1.a.y -= self.gravity
            b1_v_mag = self.bob1.v.magnitude
            self.bob1.a.x -= drag_coeff * b1_v_mag * self.bob1.v.x / self.bob1.mass
            self.bob1.a.y -= drag_coeff * b1_v_mag * self.bob1.v.y / self.bob1.mass

            # Bob 2
            self.bob2.a.y -= self.gravity
            b2_v_mag = self.bob2.v.magnitude
            self.bob2.a.x -= drag_coeff * b2_v_mag * self.bob2.v.x / self.bob2.mass
            self.bob2.a.y -= drag_coeff * b2_v_mag * self.bob2.v.y / self.bob2.mass

            # Cart friction
            rolling_friction = 2.0 * self.friction_multiplier
            self.cart.apply_force(self.cart.v * -rolling_friction)

            # --- JOINT FRICTION (Cart & Bob 1) ---
            rod_x = self.bob1.s.x - self.cart.s.x
            rod_y = self.bob1.s.y - self.cart.s.y
            rod_mag = math.hypot(rod_x, rod_y)
            
            if rod_mag < EPSILON:
                rod_dir_x, rod_dir_y = 0, 1
            else:
                rod_dir_x, rod_dir_y = rod_x / rod_mag, rod_y / rod_mag
                
            tangent_x, tangent_y = -rod_dir_y, rod_dir_x

            rel_v_x = self.bob1.v.x - self.cart.v.x
            rel_v_y = self.bob1.v.y - self.cart.v.y
            swing_speed = rel_v_x * tangent_x + rel_v_y * tangent_y

            joint_friction_coeff = 0.8 * self.friction_multiplier
            joint_f_x = tangent_x * (-joint_friction_coeff * swing_speed)
            joint_f_y = tangent_y * (-joint_friction_coeff * swing_speed)
            joint_friction = Vec(joint_f_x, joint_f_y)

            self.bob1.apply_force(joint_friction)
            self.cart.apply_force(joint_friction * -1)

            # --- CONSTRAINT 1: Cart to Bob ---
            rod_x = self.bob1.s.x - self.cart.s.x
            rod_y = self.bob1.s.y - self.cart.s.y
            rod_mag = math.hypot(rod_x, rod_y)
            if rod_mag < EPSILON:
                rod_dir = Vec(0, 1)
            else:
                rod_dir = Vec(rod_x / rod_mag, rod_y / rod_mag)

            overlap = self.bob1_rest_length - rod_mag

            k = 10000
            d = 100

            spring_force_mag = k * overlap
            rel_v_x = self.bob1.v.x - self.cart.v.x
            rel_v_y = self.bob1.v.y - self.cart.v.y
            damping_force_mag = -d * (rel_v_x * rod_dir.x + rel_v_y * rod_dir.y)

            total_rod_force = rod_dir * (spring_force_mag + damping_force_mag)

            self.cart.apply_force(total_rod_force * -1)
            self.bob1.apply_force(total_rod_force)

            # --- CONSTRAINT 2: Bob to Bob2 ---
            rod2_x = self.bob2.s.x - self.bob1.s.x
            rod2_y = self.bob2.s.y - self.bob1.s.y
            rod2_mag = math.hypot(rod2_x, rod2_y)
            if rod2_mag < EPSILON:
                rod2_dir = Vec(0, 1)
            else:
                rod2_dir = Vec(rod2_x / rod2_mag, rod2_y / rod2_mag)

            rel_v2_x = self.bob2.v.x - self.bob1.v.x
            rel_v2_y = self.bob2.v.y - self.bob1.v.y

            overlap2 = self.bob2_rest_length - rod2_mag

            spring_force_mag2 = k * overlap2
            damping_force_mag2 = -d * (rel_v2_x * rod2_dir.x + rel_v2_y * rod2_dir.y)

            total_rod_force2 = rod2_dir * (spring_force_mag2 + damping_force_mag2)

            self.bob1.apply_force(total_rod_force2 * -1)
            self.bob2.apply_force(total_rod_force2)

            # --- INTEGRATION & KINEMATIC LOCKS ---
            self.cart.a.y = 0.0

            # Pre-integration forces are locked
            self.cart.update(sub_dt)
            self.bob1.update(sub_dt)
            self.bob2.update(sub_dt)

            # Hard kinematic lock the rail exactly after v, s updates
            self.cart.s.y = TRACK_HEIGHT
            self.cart.v.y = 0.0

        # calculate rewards
        obs = self.observations()

        bob1_height = (self.bob1.s.y - (TRACK_HEIGHT - self.bob1_rest_length)) / self.bob1_rest_length
        bob2_height = ((self.bob2.s.y - (TRACK_HEIGHT - self.bob1_rest_length - self.bob2_rest_length))
                       / (self.bob1_rest_length + self.bob2_rest_length))
        # delta_h_to_threshold = self.bob2.s.y - TRACK_HEIGHT - 0.9 * (self.bob1_rest_length + self.bob2_rest_length)
        reward = (bob1_height ** 2 +  bob2_height ** 2) / 2.0

        distance_to_center = obs[0]
        central_mult = 0.5 * math.cosh( -(distance_to_center ** 2) + 1) + 0.5
        reward *= central_mult

        # PREVENT RAPID SPAMMING WITH SPEED PENALTY
        # Punish both bob extreme speeds
        b1_speed_sq = self.bob1.v.magnitude_squared
        b2_speed_sq = 0 * self.bob2.v.magnitude_squared
        # Penalize softly above a certain speed
        if b1_speed_sq > 25.0:
            reward *= 25.0 / b1_speed_sq
        if b2_speed_sq > 25.0:
            reward *= 25.0 / b2_speed_sq

        return obs, reward

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

