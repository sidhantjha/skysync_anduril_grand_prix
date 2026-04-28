"""
reward_function.py
Based on Swift system (Kaufmann et al., Nature 2023)
Two-stage RL training pipeline implementation.

Four reward terms:
    1. Progress    -- reward for closing distance to next gate
    2. Perception  -- exponential reward for keeping gate in FOV
    3. Smoothness  -- penalty for high body rates and action changes
    4. Crash       -- fixed -5.0 penalty (from Swift paper Section 4)

Observation space (31-dimensional):
    [0:3]   position (x, y, z)
    [3:6]   velocity (vx, vy, vz)
    [6:10]  orientation quaternion (qw, qx, qy, qz)
    [10:13] angular velocity (wx, wy, wz)
    [13:21] next gate corners in camera frame (4 corners x 2D pixels)
    [21:24] relative position to next gate (dx, dy, dz)
    [24:27] next gate normal vector
    [27:31] previous action (throttle, roll_rate, pitch_rate, yaw_rate)

Action space -- CTBR (4-dimensional):
    [0] collective thrust (0 to 1)
    [1] roll rate  (rad/s)
    [2] pitch rate (rad/s)
    [3] yaw rate   (rad/s)

Run demo:   python reward_function.py
Run tests:  pytest reward_function.py -v
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ------------------------------------------------------------------ constants
CRASH_PENALTY = 5.0   # Swift paper Section 4
W_PROGRESS    = 1.0
W_PERCEPTION  = 1.0
W_SMOOTHNESS  = 0.02


# -------------------------------------------------------------------- state
@dataclass
class DroneState:
    position:          np.ndarray   # [x, y, z]
    velocity:          np.ndarray   # [vx, vy, vz]
    orientation:       np.ndarray   # [qw, qx, qy, qz]
    angular_velocity:  np.ndarray   # [wx, wy, wz]
    gate_corners_cam:  np.ndarray   # shape (4, 2)  pixel coords
    gate_relative_pos: np.ndarray   # [dx, dy, dz]  to next gate
    gate_normal:       np.ndarray   # unit normal of gate plane
    prev_action:       np.ndarray   # [throttle, roll, pitch, yaw]
    next_gate_id:      int  = 0
    crashed:           bool = False
    gate_passed:       bool = False

    @classmethod
    def from_vector(cls, obs: np.ndarray, prev_action: np.ndarray,
                    next_gate_id: int = 0, crashed: bool = False,
                    gate_passed: bool = False) -> "DroneState":
        assert len(obs) == 31, f"Expected 31-dim obs, got {len(obs)}"
        return cls(
            position          = obs[0:3].copy(),
            velocity          = obs[3:6].copy(),
            orientation       = obs[6:10].copy(),
            angular_velocity  = obs[10:13].copy(),
            gate_corners_cam  = obs[13:21].reshape(4, 2).copy(),
            gate_relative_pos = obs[21:24].copy(),
            gate_normal       = obs[24:27].copy(),
            prev_action       = prev_action.copy(),
            next_gate_id      = next_gate_id,
            crashed           = crashed,
            gate_passed       = gate_passed,
        )


# ------------------------------------------------------------------ reward
class DroneRacingReward:
    """
    Per-step reward:
        r_t = w_prog   * r_progress
            + w_perc   * r_perception
            - w_smooth * r_smoothness
            + r_crash

    Args:
        w_progress:    Gate progress weight. Default 1.0.
        w_perception:  FOV perception weight. Default 1.0.
        w_smoothness:  Smoothness penalty weight. Default 0.02.
        crash_penalty: Crash penalty magnitude. Default 5.0.
        image_width:   Camera width in pixels. Default 320.
        image_height:  Camera height in pixels. Default 240.
    """

    def __init__(self, w_progress=W_PROGRESS, w_perception=W_PERCEPTION,
                 w_smoothness=W_SMOOTHNESS, crash_penalty=CRASH_PENALTY,
                 image_width=320, image_height=240):
        self.w_progress    = w_progress
        self.w_perception  = w_perception
        self.w_smoothness  = w_smoothness
        self.crash_penalty = crash_penalty
        self.img_w         = image_width
        self.img_h         = image_height

    def compute(self, state: DroneState, action: np.ndarray,
                prev_state: DroneState) -> tuple[float, dict]:
        """
        Compute total reward and per-term breakdown for logging.

        Returns:
            total: scalar float
            info:  dict with individual term values
        """
        r_prog   = self._progress(state, prev_state)
        r_perc   = self._perception(state)
        r_smooth = self._smoothness(action, state.prev_action)
        r_crash  = self._crash(state)

        total = (  self.w_progress   * r_prog
                 + self.w_perception * r_perc
                 - self.w_smoothness * r_smooth
                 + r_crash )

        info = {
            "r_progress":   self.w_progress   * r_prog,
            "r_perception": self.w_perception * r_perc,
            "r_smoothness": self.w_smoothness * r_smooth,
            "r_crash":      r_crash,
            "r_total":      total,
        }
        return float(total), info

    # ------------------------------------------------------ term 1: progress
    def _progress(self, state: DroneState, prev_state: DroneState) -> float:
        """
        Reward for closing distance to the next gate.

            r_prog = ||prev_gate_rel|| - ||curr_gate_rel||

        Positive = got closer this step.
        Negative = drifted away this step.
        Uses relative position from the 31-dim observation vector --
        no world-frame gate map needed at runtime.

        Incentivises: always moving toward the next gate every timestep.
        Without this the drone has no incentive to move at all.
        """
        dist_now  = float(np.linalg.norm(state.gate_relative_pos))
        dist_prev = float(np.linalg.norm(prev_state.gate_relative_pos))
        return dist_prev - dist_now

    # -------------------------------------------------- term 2: perception
    def _perception(self, state: DroneState) -> float:
        """
        Exponential reward for keeping the gate in the camera FOV.

        Mechanism:
            1. Compute gate centre from mean of 4 corner pixels
            2. Normalize to [-1, 1] relative to image centre
            3. r_perc = exp(-alpha * (cx_norm^2 + cy_norm^2))

        Result: 1.0 when gate is perfectly centred,
                decays smoothly to 0 as gate moves to edge,
                0.0 when gate is completely outside FOV.

        The exponential form gives a strong gradient that guides
        trajectory planning to keep the gate visible at speed --
        critical for corner detection and PnP to work reliably.

        Incentivises: maintaining gate visibility through corners.
        Prevents the drone from flying fast but losing the gate.
        """
        corners = state.gate_corners_cam      # shape (4, 2)
        centre  = corners.mean(axis=0)        # (cx, cy)

        # Return 0 if gate is completely outside the image
        any_x = np.any((corners[:,0] >= 0) & (corners[:,0] < self.img_w))
        any_y = np.any((corners[:,1] >= 0) & (corners[:,1] < self.img_h))
        if not (any_x and any_y):
            return 0.0

        cx_norm = (centre[0] - self.img_w / 2.0) / (self.img_w / 2.0)
        cy_norm = (centre[1] - self.img_h / 2.0) / (self.img_h / 2.0)

        alpha = 3.0   # controls falloff speed
        return float(np.exp(-alpha * (cx_norm**2 + cy_norm**2)))

    # ------------------------------------------------- term 3: smoothness
    def _smoothness(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """
        Penalty for high body rates and abrupt action changes.

        Two components:
            1. Action delta:  ||a_t - a_{t-1}||^2
            2. Body rate mag: ||[roll, pitch, yaw rates]||^2

        This is the most critical sim-to-real transfer term.
        Smooth CTBR commands transfer to real hardware without
        oscillation. Chattering policies that work in sim fail
        immediately on real ESCs which have 5-8ms motor lag.

        Applied as a PENALTY (subtracted from total).
        Incentivises: smooth, physically realisable commands.
        """
        delta         = action - prev_action
        action_pen    = float(np.sum(delta**2))
        rate_pen      = float(np.sum(action[1:4]**2))  # roll/pitch/yaw rates
        return action_pen + 0.1 * rate_pen

    # ----------------------------------------------------- term 4: crash
    def _crash(self, state: DroneState) -> float:
        """
        Fixed penalty for gate collision or leaving track bounds.

        Value: -5.0 (directly from Swift paper Section 4).
        Applied on the single timestep of the crash event.
        Episode resets to a random gate after crash.

        Incentivises: staying in bounds and not hitting gate frames.
        """
        return -self.crash_penalty if state.crashed else 0.0


# ---------------------------------------------------------------- pseudocode
PPO_PSEUDOCODE = """
╔══════════════════════════════════════════════════════════════╗
║  STAGE 1: BASE RL TRAINING IN SIMULATION                     ║
╚══════════════════════════════════════════════════════════════╝

INITIALIZE:
    actor   = MLP(31 -> [128, 128] -> 4,  LeakyReLU)
    critic  = MLP(31 -> [128, 128] -> 1,  LeakyReLU)
    log_std = learnable parameter, shape (4,)
    optimizer = Adam(lr=3e-4)
    envs    = 100 parallel SimEnvironments
    reward_fn = DroneRacingReward()

WHILE total_interactions < 100_000_000:

    ROLLOUT COLLECTION:
        obs = [env.reset_at_random_gate(noise=0.1) for env in envs]

        FOR step IN range(1500):          # episode length

            mean     = actor(obs)         # (100, 4)
            std      = exp(log_std)
            action   = mean + std * N(0,1)
            log_prob = log_normal(action, mean, std).sum(-1)
            value    = critic(obs)        # (100,)

            FOR each env_i, action_i:
                next_state, crashed, gate_passed = env_i.step(action_i)
                reward, info = reward_fn.compute(next_state, action_i, state_i)

                IF crashed OR gate_passed:
                    reset to random gate with perturbation

            store (obs, action, log_prob, value, reward, done)
            total_interactions += 100

    ADVANTAGE ESTIMATION (GAE, gamma=0.99, lambda=0.95):
        advantages = GAE(rewards, values, dones)
        returns    = advantages + values
        advantages = normalize(advantages)

    PPO UPDATE (10 epochs, batch_size=2048):
        FOR each batch:
            ratio      = exp(new_log_prob - old_log_prob)
            surr1      = ratio * advantage
            surr2      = clip(ratio, 0.8, 1.2) * advantage
            actor_loss = -mean(min(surr1, surr2))
            value_loss = MSE(new_value, returns)
            entropy    = normal_entropy(std).mean()
            loss       = actor_loss + 0.5*value_loss - 0.01*entropy
            loss.backward() -> clip_grad_norm(0.5) -> optimizer.step()

╔══════════════════════════════════════════════════════════════╗
║  STAGE 2: SIM-TO-REAL FINE-TUNING                            ║
╚══════════════════════════════════════════════════════════════╝

STEP 1 -- Collect real-world residuals (~50 seconds of flight):
    real_data = deploy_on_real_drone(policy=actor, duration=50s)

STEP 2a -- Model perception noise (9 Gaussian Processes):
    FOR dim IN range(9):   # position(3), velocity(3), orientation(3)
        gp[dim].fit(X=state, y=observed_state - mocap_ground_truth)

STEP 2b -- Model dynamics errors (k-NN, k=5):
    residuals = real_next_state - simulate(state, action)
    knn.fit(X=[state; action], y=residuals)

STEP 3 -- Augmented fine-tuning (20M interactions):
    FOR each env:
        env.add_perception_noise(gp_models)
        env.add_dynamics_residual(knn)

    WHILE fine_tune_interactions < 20_000_000:
        run PPO update  # gate detector weights FROZEN
                        # only actor + critic weights update
"""


# ----------------------------------------------------------------- unit tests
def _s(gate_rel, corners=None, crashed=False, prev_action=None):
    if corners is None:
        corners = np.array([[100,80],[220,80],[220,160],[100,160]], dtype=np.float32)
    if prev_action is None:
        prev_action = np.zeros(4, dtype=np.float32)
    return DroneState(
        position=np.zeros(3,dtype=np.float32), velocity=np.zeros(3,dtype=np.float32),
        orientation=np.array([1,0,0,0],dtype=np.float32), angular_velocity=np.zeros(3,dtype=np.float32),
        gate_corners_cam=corners, gate_relative_pos=np.array(gate_rel,dtype=np.float32),
        gate_normal=np.array([0,0,1],dtype=np.float32), prev_action=prev_action,
        crashed=crashed,
    )

def _z(): return np.zeros(4, dtype=np.float32)

CENTRE = np.array([[100,80],[220,80],[220,160],[100,160]], dtype=np.float32)
EDGE   = np.array([[280,80],[315,80],[315,160],[280,160]], dtype=np.float32)
OUT    = np.array([[-50,-50],[-10,-50],[-10,-10],[-50,-10]], dtype=np.float32)


class TestProgress:
    def setup_method(self): self.rf = DroneRacingReward()
    def test_positive_approaching(self):
        _, i = self.rf.compute(_s([3,0,0]), _z(), _s([5,0,0]))
        assert i["r_progress"] > 0
    def test_negative_receding(self):
        _, i = self.rf.compute(_s([5,0,0]), _z(), _s([3,0,0]))
        assert i["r_progress"] < 0
    def test_zero_stationary(self):
        s = _s([3,0,0])
        _, i = self.rf.compute(s, _z(), s)
        assert abs(i["r_progress"]) < 1e-6

class TestPerception:
    def setup_method(self): self.rf = DroneRacingReward()
    def test_high_when_centred(self):
        s = _s([3,0,0], corners=CENTRE)
        _, i = self.rf.compute(s, _z(), s)
        assert i["r_perception"] > 0.8
    def test_lower_when_off_centre(self):
        sc = _s([3,0,0], corners=CENTRE)
        se = _s([3,0,0], corners=EDGE)
        _, ic = self.rf.compute(sc, _z(), sc)
        _, ie = self.rf.compute(se, _z(), se)
        assert ic["r_perception"] > ie["r_perception"]
    def test_zero_outside_fov(self):
        s = _s([3,0,0], corners=OUT)
        _, i = self.rf.compute(s, _z(), s)
        assert i["r_perception"] == 0.0

class TestSmoothness:
    def setup_method(self): self.rf = DroneRacingReward()
    def test_zero_same_action(self):
        a = np.array([0.5,0.1,0.1,0.0], dtype=np.float32)
        s = _s([3,0,0], prev_action=a.copy())
        _, i = self.rf.compute(s, a, s)
        assert i["r_smoothness"] < 1e-6
    def test_larger_delta_larger_penalty(self):
        s = _s([3,0,0])
        _, i1 = self.rf.compute(s, np.array([0.1,0,0,0],dtype=np.float32), s)
        _, i2 = self.rf.compute(s, np.array([1.0,0,0,0],dtype=np.float32), s)
        assert i2["r_smoothness"] > i1["r_smoothness"]
    def test_high_rates_penalised(self):
        s = _s([3,0,0])
        _, il = self.rf.compute(s, np.array([0.5,0.1,0.1,0.1],dtype=np.float32), s)
        _, ih = self.rf.compute(s, np.array([0.5,5.0,5.0,5.0],dtype=np.float32), s)
        assert ih["r_smoothness"] > il["r_smoothness"]

class TestCrash:
    def setup_method(self): self.rf = DroneRacingReward(crash_penalty=5.0)
    def test_crash_minus_five(self):
        _, i = self.rf.compute(_s([3,0,0], crashed=True), _z(), _s([3,0,0]))
        assert i["r_crash"] == -5.0
    def test_no_crash_zero(self):
        _, i = self.rf.compute(_s([3,0,0]), _z(), _s([3,0,0]))
        assert i["r_crash"] == 0.0
    def test_crash_dominates_progress(self):
        total, _ = self.rf.compute(_s([0,0,0], crashed=True), _z(), _s([10,0,0]))
        assert total < 0

class TestInfoDict:
    def test_all_keys(self):
        _, i = DroneRacingReward().compute(_s([3,0,0]), _z(), _s([5,0,0]))
        for k in ["r_progress","r_perception","r_smoothness","r_crash","r_total"]:
            assert k in i
    def test_total_equals_sum(self):
        _, i = DroneRacingReward().compute(_s([3,0,0]), _z(), _s([5,0,0]))
        expected = i["r_progress"] + i["r_perception"] - i["r_smoothness"] + i["r_crash"]
        assert abs(i["r_total"] - expected) < 1e-5


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    print("=" * 64)
    print("DroneRacingReward -- Swift Paper (Kaufmann et al., 2023)")
    print("=" * 64)

    rf = DroneRacingReward()

    scenarios = [
        ("Approaching gate, gate centred in FOV",
         _s([3,0,0], CENTRE), _s([5,0,0], CENTRE), _z()),
        ("Gate near FOV edge -- low perception",
         _s([3,0,0], EDGE),   _s([5,0,0], EDGE),   _z()),
        ("Jerky body rates -- smoothness penalty",
         _s([3,0,0], CENTRE), _s([5,0,0], CENTRE),
         np.array([0.5, 5.0, 5.0, 2.0], dtype=np.float32)),
        ("Crash -- dominates everything",
         _s([3,0,0], CENTRE, crashed=True), _s([5,0,0], CENTRE), _z()),
        ("Receding from gate -- negative progress",
         _s([8,0,0], CENTRE), _s([5,0,0], CENTRE), _z()),
    ]

    for name, curr, prev, action in scenarios:
        curr.prev_action = prev.prev_action
        total, info = rf.compute(curr, action, prev)
        print(f"\n{name}:")
        for k, v in info.items():
            bar = ("+" if v >= 0 else "") + "█" * min(int(abs(v) * 4), 28)
            print(f"  {k:18s}: {v:+8.3f}  {bar}")

    print("\n" + "=" * 64)
    print("PPO Training Pipeline:")
    print(PPO_PSEUDOCODE)
    print("Run tests: pytest reward_function.py -v")