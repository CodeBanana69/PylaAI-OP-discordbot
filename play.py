import math
import random
import time

import cv2
from state_finder import get_state
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info

brawl_stars_width, brawl_stars_height = 1920, 1080
debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
visual_debug = load_toml_as_dict("cfg/general_config.toml").get('visual_debug', 'no') == "yes"

def vlog(*args):
    if visual_debug:
        print("[DBG]", *args)
super_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['super']
gadget_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['gadget']
hypercharge_crop_area = load_toml_as_dict("./cfg/lobby_config.toml")['pixel_counter_crop_area']['hypercharge']

class Movement:

    def __init__(self, window_controller):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config["unstuck_movement_delay"],
            "duration": bot_config["unstuck_movement_hold_time"],
            "toggled": False,
            "started_at": time.time(),
            "fixed": ""
        }
        self.game_mode = bot_config["gamemode_type"]
        gadget_value = bot_config["bot_uses_gadgets"]
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.super_treshold = time_config["super"]
        self.gadget_treshold = time_config["gadget"]
        self.hypercharge_treshold = time_config["hypercharge"]
        self.walls_treshold = time_config["wall_detection"]
        self.keep_walls_in_memory = self.walls_treshold <= 1
        self.last_walls_data = []
        self.keys_hold = []
        self.time_since_different_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.window_controller = window_controller
        self.TILE_SIZE = 60
        # Wall-based stuck detector: samples wall bboxes on an interval, ignores
        # walls near the player (they flicker as he overlaps them), and flags
        # "stuck" when walls don't move for wall_stuck_timeout seconds while the
        # bot is trying to move. Triggers a semicircle escape maneuver.
        self.wall_stuck_enabled = str(bot_config.get("wall_stuck_enabled", "yes")).lower() in ("yes", "true", "1")
        general_config = load_toml_as_dict("cfg/general_config.toml")
        self.wall_stuck_debug = str(general_config.get("wall_stuck_debug", "no")).lower() in ("yes", "true", "1")
        self.wall_stuck_ignore_radius = float(bot_config.get("wall_stuck_ignore_radius", 150))
        self.wall_stuck_sample_interval = float(bot_config.get("wall_stuck_sample_interval", 0.2))
        self.wall_stuck_shift_threshold = float(bot_config.get("wall_stuck_shift_threshold", 3.0))
        self.wall_stuck_timeout = float(bot_config.get("wall_stuck_timeout", 3.0))
        self.wall_stuck_min_walls = int(bot_config.get("wall_stuck_min_walls", 3))
        self.wall_stuck_state = {
            "last_sample_time": 0.0,
            "last_wall_centers": None,   # np.ndarray (N, 2) of filtered wall centers
            "stationary_since": None,    # when walls first went stationary; None = not stationary
        }

        # Semicircle escape state. Alternates side globally between triggers.
        self.escape_retreat_duration = float(bot_config.get("escape_retreat_duration", 0.4))
        self.escape_arc_duration = float(bot_config.get("escape_arc_duration", 1.2))
        self.escape_arc_degrees = float(bot_config.get("escape_arc_degrees", 135.0))
        self.escape_state = {
            "phase": None,            # "retreat" | "arc" | None
            "started_at": 0.0,
            "retreat_angle": 0.0,
            "arc_side": 1,            # +1 = CCW, -1 = CW; flipped each trigger
        }
        self._next_arc_side = 1
        
    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    @staticmethod
    def get_horizontal_move_key(direction_x, opposite=False):
        if opposite:
            return "A" if direction_x > 0 else "D"
        return "D" if direction_x > 0 else "A"

    @staticmethod
    def get_vertical_move_key(direction_y, opposite=False):
        if opposite:
            return "W" if direction_y > 0 else "S"
        return "S" if direction_y > 0 else "W"

    def attack(self, touch_up=True, touch_down=True):
        self.window_controller.press_key("M", touch_up=touch_up, touch_down=touch_down)

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press_key("H")

    def use_gadget(self):
        print("Using gadget")
        self.window_controller.press_key("G")

    def use_super(self):
        print("Using super")
        self.window_controller.press_key("E")

    @staticmethod
    def get_random_attack_key():
        random_movement = random.choice(["A", "W", "S", "D"])
        random_movement += random.choice(["A", "W", "S", "D"])
        return random_movement

    @staticmethod
    def angle_from_direction(dx: float, dy: float) -> float:
        """Return joystick angle in degrees from a direction vector.

        Uses screen coordinates: 0° = right, 90° = down, 180° = left, 270° = up.
        """
        return math.degrees(math.atan2(dy, dx)) % 360

    @staticmethod
    def angle_opposite(angle_degrees: float) -> float:
        """Return the opposite direction angle (retreat)."""
        return (angle_degrees + 180) % 360

    @staticmethod
    def reverse_movement(movement):
        # Create a translation table
        movement = movement.lower()
        translation_table = str.maketrans("wasd", "sdwa")
        return movement.translate(translation_table)

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()
        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False
                vlog("unstuck: finished")
            else:
                vlog(f"unstuck: active → {self.fix_movement_keys['fixed']}")

            return self.fix_movement_keys['fixed']

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            self.time_since_different_movement = current_time

        # print(f"Last change: {self.time_since_different_movement}", f" self.hold: {self.keys_hold}",f" c movement: {movement}")
        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            reversed_movement = self.reverse_movement(movement)

            if reversed_movement == "s":
                reversed_movement = random.choice(['aw', 'dw'])
            elif reversed_movement == "w":
                reversed_movement = random.choice(['as', 'ds'])

            """
            If reverse movement is either "w" or "s" it means the bot is stuck
            going forward or backward. This happens when it doesn't detect a wall in front
            so to go around it it could either go to the left diagonal or right
            """

            self.fix_movement_keys['fixed'] = reversed_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            vlog(f"unstuck triggered: {movement} → {reversed_movement}")
            return reversed_movement

        return movement

    def _wslog(self, *args):
        """Dedicated logger for wall-stuck / escape — independent of vlog/visual_debug
        so the new unstuck machinery can be traced without dumping the full debug stream.
        """
        if self.wall_stuck_debug:
            print("[WS]", *args)

    def _wall_centers_filtered(self, walls, player_pos):
        """Return (N, 2) float array of wall centers, excluding walls whose
        center lies within wall_stuck_ignore_radius of the player (those
        flicker as the player overlaps them).
        """
        import numpy as np
        if not walls:
            return np.empty((0, 2), dtype=np.float32)
        centers = []
        px, py = player_pos
        r2 = self.wall_stuck_ignore_radius * self.wall_stuck_ignore_radius
        for box in walls:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            dx, dy = cx - px, cy - py
            if dx * dx + dy * dy >= r2:
                centers.append((cx, cy))
        return np.asarray(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32)

    def _avg_wall_shift(self, prev_centers, curr_centers):
        """Greedy nearest-neighbor match between two sets of wall centers.
        Returns mean pairwise distance (px). Returns None if either set is too
        small (can't form a reliable metric).
        """
        import numpy as np
        if prev_centers is None or len(prev_centers) < self.wall_stuck_min_walls:
            return None
        if len(curr_centers) < self.wall_stuck_min_walls:
            return None
        # For each prev center, find nearest curr center (O(N*M), fine for N~20)
        diffs = prev_centers[:, None, :] - curr_centers[None, :, :]
        d2 = (diffs * diffs).sum(axis=2)
        nearest = np.sqrt(d2.min(axis=1))
        return float(nearest.mean())

    def detect_wall_stuck(self, walls, player_pos, is_trying_to_move, current_time):
        """Wall-based stuck detector. Returns True if the walls around the
        player have been stationary longer than wall_stuck_timeout while the
        bot was issuing movement commands — meaning the bot is pressed against
        something and not actually moving.
        """
        if not self.wall_stuck_enabled or player_pos is None:
            return False
        state = self.wall_stuck_state
        if current_time - state["last_sample_time"] < self.wall_stuck_sample_interval:
            # Between samples: just honor the latest stationary flag
            if state["stationary_since"] is None or not is_trying_to_move:
                return False
            return (current_time - state["stationary_since"]) >= self.wall_stuck_timeout

        curr_centers = self._wall_centers_filtered(walls, player_pos)
        shift = self._avg_wall_shift(state["last_wall_centers"], curr_centers)
        state["last_wall_centers"] = curr_centers
        state["last_sample_time"] = current_time

        if shift is None:
            # Not enough walls to judge — treat as "unknown", don't advance timer
            state["stationary_since"] = None
            return False

        if shift < self.wall_stuck_shift_threshold:
            if state["stationary_since"] is None:
                state["stationary_since"] = current_time
            self._wslog(f"walls shift={shift:.2f}px, stationary for "
                        f"{current_time - state['stationary_since']:.2f}s "
                        f"(trying_to_move={is_trying_to_move})")
        else:
            if state["stationary_since"] is not None:
                self._wslog(f"walls moved again: shift={shift:.2f}px, resetting timer")
            state["stationary_since"] = None

        if state["stationary_since"] is None or not is_trying_to_move:
            return False
        return (current_time - state["stationary_since"]) >= self.wall_stuck_timeout

    def _reset_wall_stuck_state(self, current_time):
        """Clear the wall-stuck timer. Call after triggering an escape to
        avoid retriggering during/just after the maneuver.
        """
        self.wall_stuck_state["stationary_since"] = None
        self.wall_stuck_state["last_wall_centers"] = None
        self.wall_stuck_state["last_sample_time"] = current_time

    def start_semicircle_escape(self, angle, current_time):
        """Begin the retreat+arc escape maneuver. arc_side alternates globally
        between triggers.
        """
        side = self._next_arc_side
        self._next_arc_side = -side
        self.escape_state["phase"] = "retreat"
        self.escape_state["started_at"] = current_time
        self.escape_state["retreat_angle"] = self.angle_opposite(angle)
        self.escape_state["arc_side"] = side
        self._wslog(f"semicircle escape START: angle={angle:.1f}° "
                    f"retreat={self.escape_state['retreat_angle']:.1f}° "
                    f"side={'CCW' if side > 0 else 'CW'}")

    def semicircle_escape_step(self, current_time):
        """Return the current commanded angle for the active escape maneuver,
        or None if no maneuver is active / it just finished.
        """
        state = self.escape_state
        phase = state["phase"]
        if phase is None:
            return None
        elapsed = current_time - state["started_at"]

        if phase == "retreat":
            if elapsed < self.escape_retreat_duration:
                return state["retreat_angle"]
            # Transition: arc starts from retreat angle and sweeps arc_degrees
            state["phase"] = "arc"
            state["started_at"] = current_time
            self._wslog("semicircle escape: retreat done, starting arc")
            elapsed = 0.0
            phase = "arc"

        if phase == "arc":
            if elapsed >= self.escape_arc_duration:
                state["phase"] = None
                self._wslog("semicircle escape: finished")
                return None
            t = elapsed / self.escape_arc_duration  # 0..1
            sweep = self.escape_arc_degrees * t * state["arc_side"]
            return (state["retreat_angle"] + sweep) % 360

        return None


class Play(Movement):

    def __init__(self, main_info_model, tile_detector_model, window_controller):
        super().__init__(window_controller)

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main_info = Detect(main_info_model, classes=['enemy', 'teammate', 'player'])
        self.tile_detector_model_classes = bot_config["wall_model_classes"]
        self.Detect_tile_detector = Detect(
            tile_detector_model,
            classes=self.tile_detector_model_classes
        )

        self.time_since_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.time_since_hypercharge_checked = time.time()
        self.time_since_super_checked = time.time()
        self.time_since_walls_checked = 0
        self.time_since_movement_change = time.time()
        self.time_since_player_last_found = time.time()
        self.current_brawler = None
        self.is_hypercharge_ready = False
        self.is_gadget_ready = False
        self.is_super_ready = False
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement = None
        self.last_movement_time = time.time()
        self.locked_teammate = None
        self.locked_teammate_distance = float('inf')
        self.teammate_hysteresis = 0.20  # Switch only if another teammate is 20% closer
        self.wall_history = []
        self.wall_history_length = 3  # Number of frames to keep walls
        self.scene_data = []
        self.should_detect_walls = bot_config["gamemode"] in ["brawlball", "brawl_ball", "brawll ball", "showdown"]
        self.is_showdown = bot_config["gamemode"] == "showdown"
        self.minimum_movement_delay = bot_config["minimum_movement_delay"]
        self.no_detection_proceed_delay = time_config["no_detection_proceed"]
        self.gadget_pixels_minimum = bot_config["gadget_pixels_minimum"]
        self.hypercharge_pixels_minimum = bot_config["hypercharge_pixels_minimum"]
        self.super_pixels_minimum = bot_config["super_pixels_minimum"]
        self.wall_detection_confidence = bot_config["wall_detection_confidence"]
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self.time_since_holding_attack = None
        self.seconds_to_hold_attack_after_reaching_max = load_toml_as_dict("cfg/bot_config.toml")["seconds_to_hold_attack_after_reaching_max"]
        self.current_frame = None
        # Fog color (poison gas in showdown) — sampled from images/fog_sample.png.
        # Narrow range because the fog fully overlays whatever is under it.
        self.fog_hsv_low = (50, 95, 215)
        self.fog_hsv_high = (60, 125, 245)
        # Fog proximity override: movement flees fog when a real fog front is
        # within this distance. Attack logic is untouched.
        self.fog_flee_distance = 130
        # Confidence filters to avoid reacting to stray pixels:
        #   - morph opening kernel removes speckle noise
        #   - only connected fog blobs ≥ this many pixels are trusted
        #   - need at least this many trusted fog pixels inside the flee
        #     radius before the override kicks in
        self.fog_min_blob_pixels = 300
        self.fog_min_pixels_in_radius = 50
        # Run the fog-threat check once every N calls to get_showdown_movement.
        # Between checks the previous decision is reused.
        self.fog_check_every_n_frames = 3
        self._fog_check_counter = 0
        self._fog_threat_cached = None
        # Per-frame cache of the trusted fog mask, keyed by id(frame).
        # Cache covers one pipeline run so the mask is not rebuilt when both
        # detect_fog_threat and detect_fog_direction are called on the same frame.
        self._fog_mask_cache_frame_id = None
        self._fog_mask_cache_value = None
        self._fog_mask_cache_origin = None

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info['attack_range']
            safe_range = info['safe_range']
            super_range = info['super_range']
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info[brawler]['ignore_walls_for_attacks']
        elif skill_type == "super":
            return brawlers_info[brawler]['ignore_walls_for_supers']
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        return brawlers_info[brawler]['hold_attack'] > 0

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls):
        if not walls:
            return False

        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x, max_x = min(p1_t[0], p2_t[0]), max(p1_t[0], p2_t[0])
        min_y, max_y = min(p1_t[1], p2_t[1]), max(p1_t[1], p2_t[1])
        for wall in walls:
            x1, y1, x2, y2 = wall

            if max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2:
                continue

            rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                return True
        return False

    def no_enemy_movement(self, player_data, walls):
        player_position = self.get_player_pos(player_data)
        preferred_movement = 'W' if self.game_mode == 3 else 'D'  # Adjust based on game mode

        if not self.is_path_blocked(player_position, preferred_movement, walls):
            return preferred_movement
        else:
            # Try alternative movements
            alternative_moves = ['W', 'A', 'S', 'D']
            alternative_moves.remove(preferred_movement)
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_position, move, walls):
                    return move
            print("no movement possible ?")
            # If no movement is possible, return empty string
            return preferred_movement

    def _build_trusted_fog_mask(self, frame, roi_center, roi_radius):
        """Return (mask, (ox, oy)) or None.

        Only processes an ROI of side 2*roi_radius+1 around roi_center —
        we only care about fog that's close to the player.
        Mask contains only fog pixels that belong to a large, morphologically
        clean blob — not stray color noise. (ox, oy) is the ROI's top-left
        offset in frame coordinates so callers can translate back.

        Result is cached per-frame (keyed by id(frame) and ROI tuple).
        """
        if frame is None:
            return None

        cache_key = (id(frame), int(roi_center[0]), int(roi_center[1]), int(roi_radius))
        if self._fog_mask_cache_frame_id == cache_key:
            return self._fog_mask_cache_value

        import numpy as np
        h, w = frame.shape[:2]
        cx, cy = int(roi_center[0]), int(roi_center[1])
        x0, y0 = max(0, cx - roi_radius), max(0, cy - roi_radius)
        x1, y1 = min(w, cx + roi_radius + 1), min(h, cy + roi_radius + 1)
        if x0 >= x1 or y0 >= y1:
            self._fog_mask_cache_frame_id = cache_key
            self._fog_mask_cache_value = None
            return None
        region = frame[y0:y1, x0:x1]
        origin = (x0, y0)

        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        low = np.array(self.fog_hsv_low, dtype=np.uint8)
        high = np.array(self.fog_hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        result = None
        if num_labels > 1:
            trusted = np.zeros_like(mask)
            any_kept = False
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= self.fog_min_blob_pixels:
                    trusted[labels == label] = 255
                    any_kept = True
            if any_kept and cv2.countNonZero(trusted) > 0:
                result = (trusted, origin)

        self._fog_mask_cache_frame_id = cache_key
        self._fog_mask_cache_value = result
        return result

    def detect_fog_threat(self, frame, player_position):
        """Check whether a real fog front is within self.fog_flee_distance of
        the player. Returns the flee angle (away from local fog mass) if so,
        else None.

        Confidence pipeline:
          1. HSV threshold → raw mask.
          2. Morph open + size-filtered connected components → trusted mask.
          3. Count trusted fog pixels inside a disk of radius fog_flee_distance
             around the player. If count ≥ fog_min_pixels_in_radius, it's a
             real incoming front — not a stray artifact.
        The flee direction is the angle opposite to the centroid of the
        trusted fog pixels *inside the radius*, so we run away from the
        closest wall of fog, not from fog on the far side of the map.
        """
        r = self.fog_flee_distance
        built = self._build_trusted_fog_mask(frame, roi_center=player_position, roi_radius=r)
        if built is None:
            return None
        mask, (ox, oy) = built

        import numpy as np
        px, py = int(player_position[0]), int(player_position[1])
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None

        # Translate ROI-local coords to frame coords, then filter to circle
        dx_all = (xs + ox) - px
        dy_all = (ys + oy) - py
        dist_sq = dx_all * dx_all + dy_all * dy_all
        inside = dist_sq <= r * r
        count = int(inside.sum())
        if count < self.fog_min_pixels_in_radius:
            return None

        # Centroid of the nearby fog mass, then flee opposite direction
        cx = float(dx_all[inside].mean())
        cy = float(dy_all[inside].mean())
        if math.hypot(cx, cy) < 1:
            return None
        toward_fog = self.angle_from_direction(cx, cy)
        flee = self.angle_opposite(toward_fog)
        vlog(f"fog threat: {count}px within {r}px → flee angle={flee:.1f}° (fog at {toward_fog:.1f}°)")
        return flee

    def showdown_roam(self, player_data, walls):
        """Idle roam movement for showdown: rotate the joystick angle each
        call to look around. Close-fog avoidance is handled by the uniform
        fog-threat override in get_showdown_movement.
        """
        self._roam_spin_angle = (getattr(self, "_roam_spin_angle", 0.0) + 15.0) % 360
        vlog(f"roam: idle spin → angle={self._roam_spin_angle:.1f}°")
        return self._roam_spin_angle

    def showdown_follow_teammate(self, player_data, teammate_data, walls):
        """Move towards the closest visible teammate, avoiding walls."""
        player_pos = self.get_player_pos(player_data)

        # Find the closest detected teammate this frame
        closest_teammate = None
        closest_distance = float('inf')
        for tm in teammate_data:
            tm_pos = self.get_enemy_pos(tm)
            dist = self.get_distance(tm_pos, player_pos)
            if dist < closest_distance:
                closest_distance = dist
                closest_teammate = tm_pos

        if closest_teammate is None:
            self.locked_teammate = None
            self.locked_teammate_distance = float('inf')
            return self.showdown_roam(player_data, walls)

        # Hysteresis only applies when there are multiple teammates to choose from.
        # If we already have a locked target, check whether to switch to a closer one.
        # Either way, always update the locked target's position to this frame's value.
        if self.locked_teammate is not None:
            locked_dist = self.get_distance(self.locked_teammate, player_pos)
            if closest_distance < locked_dist * (1 - self.teammate_hysteresis):
                vlog(f"follow teammate: switched target ({int(locked_dist)}px → {int(closest_distance)}px)")
                self.locked_teammate = closest_teammate
                self.locked_teammate_distance = closest_distance
            else:
                # Same target (or similar) — update its position to the current frame
                self.locked_teammate = closest_teammate
                self.locked_teammate_distance = closest_distance
        else:
            self.locked_teammate = closest_teammate
            self.locked_teammate_distance = closest_distance

        direction_x = self.locked_teammate[0] - player_pos[0]
        direction_y = self.locked_teammate[1] - player_pos[1]

        angle = self.angle_from_direction(direction_x, direction_y)
        best = self.find_best_angle(player_pos, angle, walls)
        vlog(f"follow teammate → angle={best:.1f}° (desired={angle:.1f}°, dist={int(closest_distance)}px, "
             f"player={int(player_pos[0])},{int(player_pos[1])} tm={int(self.locked_teammate[0])},{int(self.locked_teammate[1])})")
        return best

    def get_showdown_movement(self, player_data, enemy_data, teammate_data, walls, brawler):
        """Showdown movement using analog joystick angles.

        Always returns a float angle in degrees (0–360).
        0° = right, 90° = down, 180° = left, 270° = up.
        """
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")

        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and \
                time.time() - self.time_since_holding_attack >= brawler_info['hold_attack'] + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)

        # Fog override is applied uniformly at the end so it works for all
        # three movement sources (chase/retreat enemy, follow teammate, roam).
        # Throttled: only actually run the detector once every N calls and
        # reuse the last decision in between — the fog advances slowly enough
        # that a few frames of staleness don't matter.
        self._fog_check_counter += 1
        if self._fog_check_counter >= self.fog_check_every_n_frames:
            self._fog_threat_cached = self.detect_fog_threat(self.current_frame, player_pos)
            self._fog_check_counter = 0
        fog_flee_angle = self._fog_threat_cached

        enemy_coords = None
        enemy_distance = None

        # --- No enemy in sight: follow teammate or roam ---
        if not self.is_there_enemy(enemy_data):
            if teammate_data:
                vlog(f"no enemy → follow teammate ({len(teammate_data)} visible)")
                angle = self.showdown_follow_teammate(player_data, teammate_data, walls)
            else:
                vlog("no enemy, no teammate → roam")
                angle = self.showdown_roam(player_data, walls)
        else:
            enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
            if enemy_coords is None:
                if teammate_data:
                    vlog("enemy detected but unreachable → follow teammate")
                    angle = self.showdown_follow_teammate(player_data, teammate_data, walls)
                else:
                    vlog("enemy detected but unreachable, no teammate → roam")
                    angle = self.showdown_roam(player_data, walls)
            else:
                # --- Compute exact angle toward/away from enemy, then wall-avoid ---
                direction_x = enemy_coords[0] - player_pos[0]
                direction_y = enemy_coords[1] - player_pos[1]
                toward_angle = self.angle_from_direction(direction_x, direction_y)

                if enemy_distance > safe_range:
                    desired = toward_angle
                    vlog(f"enemy detected → approach desired={desired:.1f}° (dist={int(enemy_distance)}px, safe={safe_range}px)")
                else:
                    desired = self.angle_opposite(toward_angle)
                    vlog(f"enemy too close → retreat desired={desired:.1f}° (dist={int(enemy_distance)}px, safe={safe_range}px)")

                angle = self.find_best_angle(player_pos, desired, walls)
                vlog(f"showdown: movement angle={angle:.1f}° (desired={desired:.1f}°)")

        # --- Fog proximity override ---
        # If trusted fog is close, replace movement with a flee angle. Attack
        # block below still fires independently based on enemy_distance.
        if fog_flee_angle is not None:
            angle = self.find_best_angle(player_pos, fog_flee_angle, walls)
            vlog(f"showdown: fog override → angle={angle:.1f}°")

        # --- Skills (only when an attackable enemy was found) ---
        if enemy_coords is None:
            return angle

        if self.is_super_ready and self.time_since_holding_attack is None:
            super_type = brawler_info['super_type']
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")
            if (enemy_hittable and
                    (enemy_distance <= super_range
                     or super_type in ["spawnable", "other"]
                     or (brawler in ["stu", "surge"] and super_type == "charge" and enemy_distance <= super_range + attack_range)
                    )):
                if self.is_hypercharge_ready:
                    self.use_hypercharge()
                    self.time_since_hypercharge_checked = time.time()
                    self.is_hypercharge_ready = False
                self.use_super()
                self.time_since_super_checked = time.time()
                self.is_super_ready = False

        vlog(f"showdown movement → angle={angle:.1f}°")

        if enemy_distance <= attack_range:
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
            vlog(f"enemy in attack range (dist={int(enemy_distance)}px, range={attack_range}px), hittable={enemy_hittable}")
            if enemy_hittable:
                if self.should_use_gadget and self.is_gadget_ready and self.time_since_holding_attack is None:
                    self.use_gadget()
                    self.time_since_gadget_checked = time.time()
                    self.is_gadget_ready = False

                if not must_brawler_hold_attack:
                    self.attack()
                else:
                    if self.time_since_holding_attack is None:
                        self.time_since_holding_attack = time.time()
                        self.attack(touch_up=False, touch_down=True)
                    elif time.time() - self.time_since_holding_attack >= self.brawlers_info[brawler]['hold_attack']:
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None
        else:
            vlog(f"enemy out of attack range (dist={int(enemy_distance)}px, range={attack_range}px)")

        return angle

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        if self.walls_block_line_of_sight(player_pos, enemy_pos, walls):
            return False
        return True

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            return closest_hittable
        elif closest_unhittable:
            return closest_unhittable

        return None, None

    def get_main_data(self, frame):
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        return data

    def is_path_blocked(self, player_pos, move_direction, walls, distance=None):  # Increased distance
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= distance
        if 's' in move_direction.lower():
            dy += distance
        if 'a' in move_direction.lower():
            dx -= distance
        if 'd' in move_direction.lower():
            dx += distance
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        return self.walls_block_line_of_sight(player_pos, new_pos, walls)

    def is_path_blocked_angle(self, player_pos, angle_degrees, walls, distance=None):
        """Check if the path in the given angle direction is blocked by walls.

        Uses two probe distances (half-tile and full-tile) so that walls that
        start very close to the player are also detected.
        """
        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # Check both close (half-tile) and far (full-tile) points
        for d in (distance * 0.5, distance):
            new_pos = (player_pos[0] + cos_a * d, player_pos[1] + sin_a * d)
            if self.walls_block_line_of_sight(player_pos, new_pos, walls):
                return True
        return False

    def find_best_angle(self, player_pos, desired_angle, walls, sweep_range=160, step=10):
        """Find the closest unblocked angle to desired_angle within ±sweep_range degrees.

        Sweeps outward from the desired angle in alternating left/right steps so
        the first hit is always the least deviation from the goal direction.
        Returns desired_angle unchanged if no walls (or no clear path found).
        """
        if not self.is_path_blocked_angle(player_pos, desired_angle, walls):
            return desired_angle

        for offset in range(step, sweep_range + 1, step):
            for sign in (1, -1):
                candidate = (desired_angle + sign * offset) % 360
                if not self.is_path_blocked_angle(player_pos, candidate, walls):
                    return candidate

        # Nothing clear found — return desired anyway (better than stopping)
        return desired_angle

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if "player" not in data.keys():
            incomplete = True  # This is required so track_no_detections can also keep track if enemy is missing

        if "enemy" not in data.keys():
            data['enemy'] = None

        if "teammate" not in data.keys():
            data['teammate'] = None

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        return False if incomplete else data

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None,
                "teammate": None,
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    def do_movement(self, movement):
        if isinstance(movement, float):
            # Analog joystick path: movement is an angle in degrees
            self.window_controller.move_joystick_angle(movement)
            self.keys_hold = []
        else:
            # Legacy WASD path
            movement = movement.lower()
            keys_to_keyDown = []
            keys_to_keyUp = []
            for key in ['w', 'a', 's', 'd']:
                if key in movement:
                    keys_to_keyDown.append(key)
                else:
                    keys_to_keyUp.append(key)

            if keys_to_keyDown:
                self.window_controller.keys_down(keys_to_keyDown)

            self.window_controller.keys_up(keys_to_keyUp)

            self.keys_hold = keys_to_keyDown

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    def _debounce_angle(self, angle: float, threshold_deg: float = 10.0) -> float:
        """Suppress small angle changes to avoid jitter.

        Only adopts the new angle if it differs by more than threshold_deg
        from the last committed angle, OR if no angle was committed yet.
        """
        if self.last_movement is None or not isinstance(self.last_movement, float):
            self.last_movement = angle
            self.last_movement_time = time.time()
            return angle

        diff = abs((angle - self.last_movement + 180) % 360 - 180)
        if diff > threshold_deg:
            self.last_movement = angle
            self.last_movement_time = time.time()

        return self.last_movement

    def loop(self, brawler, data, current_time):
        if self.is_showdown:
            movement = self.get_showdown_movement(
                player_data=data['player'][0],
                enemy_data=data['enemy'],
                teammate_data=data['teammate'],
                walls=data['wall'],
                brawler=brawler,
            )
            # Debounce small angle jitter before sending to joystick
            movement = self._debounce_angle(movement)
        else:
            movement = self.get_movement(player_data=data['player'][0], enemy_data=data['enemy'], walls=data['wall'], brawler=brawler)

        current_time = time.time()
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            if isinstance(movement, float):
                # 1. If a semicircle escape is already running, just advance it.
                escape_angle = self.semicircle_escape_step(current_time)
                if escape_angle is not None:
                    movement = escape_angle
                else:
                    # 2. Wall-based stuck detector triggers the semicircle escape.
                    player_pos = self.get_player_pos(data['player'][0]) if data.get('player') else None
                    walls = data.get('wall') or []
                    is_trying = isinstance(movement, float)
                    if self.detect_wall_stuck(walls, player_pos, is_trying, current_time):
                        self.start_semicircle_escape(movement, current_time)
                        self._reset_wall_stuck_state(current_time)
                        movement = self.semicircle_escape_step(current_time) or movement
            else:
                movement = self.unstuck_movement_if_needed(movement, current_time)
            self.do_movement(movement)
            self.time_since_movement = time.time()
        return movement

    def check_if_hypercharge_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(hypercharge_crop_area[0] * wr), int(hypercharge_crop_area[1] * hr)
        x2, y2 = int(hypercharge_crop_area[2] * wr), int(hypercharge_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        purple_pixels = count_hsv_pixels(screenshot, (137, 158, 159), (179, 255, 255))
        if debug:
            print("hypercharge purple pixels:", purple_pixels, "(if > ", self.hypercharge_pixels_minimum, " then hypercharge is ready)")
            cv2.imwrite(f"debug_frames/hypercharge_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if purple_pixels > self.hypercharge_pixels_minimum:
            return True
        return False

    def check_if_gadget_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(gadget_crop_area[0] * wr), int(gadget_crop_area[1] * hr)
        x2, y2 = int(gadget_crop_area[2] * wr), int(gadget_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        green_pixels = count_hsv_pixels(screenshot, (57, 219, 165), (62, 255, 255))
        if debug:
            print("gadget green pixels:", green_pixels, "(if > ", self.gadget_pixels_minimum, " then gadget is ready)")
            cv2.imwrite(f"debug_frames/gadget_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if green_pixels > self.gadget_pixels_minimum:
            return True
        return False

    def check_if_super_ready(self, frame):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(super_crop_area[0] * wr), int(super_crop_area[1] * hr)
        x2, y2 = int(super_crop_area[2] * wr), int(super_crop_area[3] * hr)
        screenshot = frame[y1:y2, x1:x2]
        yellow_pixels = count_hsv_pixels(screenshot, (17, 170, 200), (27, 255, 255))
        if debug:
            print("super yellow pixels:", yellow_pixels, "(if > ", self.super_pixels_minimum, " then super is ready)")
            cv2.imwrite(f"debug_frames/super_debug_{int(time.time())}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        if yellow_pixels > self.super_pixels_minimum:
            return True
        return False

    def get_tile_data(self, frame):
        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    def process_tile_data(self, tile_data):
        walls = []
        for class_name, boxes in tile_data.items():
            if class_name != 'bush':
                walls.extend(boxes)

        # Add walls to history
        self.wall_history.append(walls)
        if len(self.wall_history) > self.wall_history_length:
            self.wall_history.pop(0)
        # Combine walls from history
        combined_walls = self.combine_walls_from_history()

        return combined_walls

    def combine_walls_from_history(self):
        unique_walls = {tuple(wall) for walls in self.wall_history for wall in walls}
        return list(unique_walls)

    def get_movement(self, player_data, enemy_data, walls, brawler):
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")
        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        # if a brawler has been holding an attack for its max duration + the bot setting, then we release
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and time.time() - self.time_since_holding_attack >= brawler_info['hold_attack'] + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)
        if debug: print("found player pos:", player_pos)
        if not self.is_there_enemy(enemy_data):
            return self.no_enemy_movement(player_data, walls)
        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None:
            return self.no_enemy_movement(player_data, walls)
        if debug: print("found enemy pos:", enemy_coords)
        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        # Determine initial movement direction
        if enemy_distance > safe_range:  # Move towards the enemy
            move_horizontal = self.get_horizontal_move_key(direction_x)
            move_vertical = self.get_vertical_move_key(direction_y)
        else:  # Move away from the enemy
            move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
            move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

        movement_options = [move_horizontal + move_vertical]
        if self.game_mode == 3:
            movement_options += [move_vertical, move_horizontal]
        elif self.game_mode == 5:
            movement_options += [move_horizontal, move_vertical]
        else:
            raise ValueError("Gamemode type is invalid")

        # Check for walls and adjust movement
        for move in movement_options:
            if not self.is_path_blocked(player_pos, move, walls):
                movement = move
                break
        else:
            print("default paths are blocked")
            # If all preferred directions are blocked, try other directions
            alternative_moves = ['W', 'A', 'S', 'D']
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break
            else:
                # if no movement is available, we still try to go in the best direction
                # because it's better than doing nothing
                movement = move_horizontal + move_vertical

        current_time = time.time()
        if movement != self.last_movement:
            if current_time - self.last_movement_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_time = current_time
            else:
                movement = self.last_movement  # Continue previous movement
        else:
            self.last_movement_time = current_time  # Reset timer if movement didn't change

        if self.is_super_ready and self.time_since_holding_attack is None:
            super_type = brawler_info['super_type']
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")

            if (enemy_hittable and
                    (enemy_distance <= super_range
                     or super_type in ["spawnable", "other"]
                     or (brawler in ["stu", "surge"] and super_type == "charge" and enemy_distance <= super_range + attack_range)
                    )):
                if self.is_hypercharge_ready:
                    self.use_hypercharge()
                    self.time_since_hypercharge_checked = time.time()
                    self.is_hypercharge_ready = False
                self.use_super()
                self.time_since_super_checked = time.time()
                self.is_super_ready = False

        # Attack if enemy is within attack range and hittable
        if enemy_distance <= attack_range:
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
            if enemy_hittable:
                if self.should_use_gadget == True and self.is_gadget_ready and self.time_since_holding_attack is None:
                    self.use_gadget()
                    self.time_since_gadget_checked = time.time()
                    self.is_gadget_ready = False

                if not must_brawler_hold_attack:
                    self.attack()
                else:
                    if self.time_since_holding_attack is None:
                        self.time_since_holding_attack = time.time()
                        self.attack(touch_up=False, touch_down=True)
                    elif time.time() - self.time_since_holding_attack >= self.brawlers_info[brawler]['hold_attack']:
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None


        return movement

    def main(self, frame, brawler, main):
        current_time = time.time()
        data = self.get_main_data(frame)
        if self.should_detect_walls and current_time - self.time_since_walls_checked > self.walls_treshold:

            tile_data = self.get_tile_data(frame)

            walls = self.process_tile_data(tile_data)

            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            data['wall'] = walls
        elif self.keep_walls_in_memory:
            data['wall'] = self.last_walls_data


        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            self.time_since_player_last_found = time.time()
            if main.state != "match":
                main.state = get_state(frame)
                if main.state != "match":
                    data = None
        if not data:
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.keys_up(list("wasd"))
            self.time_since_different_movement = time.time()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame)
                if current_state != "match":
                    self.time_since_last_proceeding = current_time
                else:
                    print("haven't detected the player in a while proceeding")
                    self.window_controller.press_key("Q")
                    self.time_since_last_proceeding = time.time()
            return
        self.time_since_last_proceeding = time.time()
        self.is_hypercharge_ready = False
        if current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
        self.is_gadget_ready = False
        if current_time - self.time_since_gadget_checked > self.gadget_treshold:
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
        self.is_super_ready = False
        if current_time - self.time_since_super_checked > self.super_treshold:
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time

        self.current_frame = frame
        movement = self.loop(brawler, data, current_time)

        if visual_debug:
            self.show_visual_debug(frame, data, brawler)

        # if data:
        #     # Record scene data
        #     self.scene_data.append({
        #         'frame_number': len(self.scene_data),
        #         'player': data.get('player', []),
        #         'enemy': data.get('enemy', []),
        #         'wall': data.get('wall', []),
        #         'movement': movement,
        #     })

    def show_visual_debug(self, frame, data, brawler=None):
        import numpy as np
        img = frame.copy() if isinstance(frame, np.ndarray) else np.array(frame)

        # --- Fog overlay ---
        # Only draw the fog tint + centroid arrow when a fog threat is strong
        # enough to trigger evasion (same thresholds as detect_fog_threat):
        # trusted mask inside flee-radius must contain >= fog_min_pixels_in_radius.
        if data.get("player"):
            px, py = self.get_player_pos(data["player"][0])
            r = self.fog_flee_distance
            built = self._build_trusted_fog_mask(frame, roi_center=(px, py), roi_radius=r)
            if built is not None:
                mask, (ox, oy) = built
                ys, xs = np.nonzero(mask)
                if xs.size > 0:
                    dx_all = (xs + ox) - px
                    dy_all = (ys + oy) - py
                    dist_sq = dx_all * dx_all + dy_all * dy_all
                    inside = dist_sq <= r * r
                    if int(inside.sum()) >= self.fog_min_pixels_in_radius:
                        # Paint only the trusted, in-radius pixels magenta
                        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        xs_in = xs[inside] + ox
                        ys_in = ys[inside] + oy
                        full_mask[ys_in, xs_in] = 255
                        tint = np.zeros_like(img)
                        tint[:, :] = (255, 0, 255)  # magenta in RGB
                        img = np.where(full_mask[..., None] > 0,
                                       cv2.addWeighted(img, 0.5, tint, 0.5, 0),
                                       img)
                        fog_cx = int(dx_all[inside].mean() + px)
                        fog_cy = int(dy_all[inside].mean() + py)
                        cv2.circle(img, (fog_cx, fog_cy), 8, (255, 0, 255), -1)
                        cv2.putText(img, "fog", (fog_cx + 10, fog_cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        cv2.arrowedLine(img, (int(px), int(py)), (fog_cx, fog_cy),
                                        (255, 0, 255), 2, tipLength=0.15)

        # Colors in RGB (frame is kept in RGB; converted to BGR only for imshow).
        colors = {
            "player":   (0, 255, 0),    # green
            "teammate": (0, 0, 255),    # blue
            "enemy":    (255, 0, 0),    # red
            "wall":     (128, 128, 128),  # gray
        }
        for key, color in colors.items():
            boxes = data.get(key)
            if not boxes:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, key, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw attack/super ranges around the player based on brawlers_info.json.
        if brawler and data.get("player"):
            info = self.brawlers_info.get(brawler)
            if info:
                px, py = self.get_player_pos(data["player"][0])
                center = (int(px), int(py))
                attack_range = int(info.get("attack_range", 0))
                super_range = int(info.get("super_range", 0))
                if attack_range > 0:
                    cv2.circle(img, center, attack_range, (160, 32, 240), 2)  # purple
                if super_range > 0:
                    cv2.circle(img, center, super_range, (255, 255, 0), 2)  # yellow

        cv2.imshow("Pyla Visual Debug", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    @staticmethod
    def movement_to_direction(movement):
        mapping = {
            'w': 'up',
            'a': 'left',
            's': 'down',
            'd': 'right',
            'wa': 'up-left',
            'aw': 'up-left',
            'wd': 'up-right',
            'dw': 'up-right',
            'sa': 'down-left',
            'as': 'down-left',
            'sd': 'down-right',
            'ds': 'down-right',
        }
        movement = movement.lower()
        movement = ''.join(sorted(movement))
        return mapping.get(movement, 'idle' if movement == '' else movement)
