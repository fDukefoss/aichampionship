import pygame
from time import sleep
#import requests
#from typing import List, Optional
from ..mathematics.randomizer import seed, random_choice, random_number
from ..elements.car import Car
from ..elements.road import Road
from ..elements.sensor import Sensor
from ..mathematics.vector import Vector
import json

# Define constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
CAR_COLORS = ['yellow', 'blue', 'red']
MAX_TICKS = 60 * 60  # 60 seconds @ 60 fps
MAX_MS = 60 * 1000600   # 60 seconds flat

# Define game state
class GameState:
    def __init__(self, api_url: str):
        self.ego = None
        self.cars = []
        self.car_bucket = []
        self.sensors = []
        self.road = None
        self.statistics = None
        self.sensors_enabled = True
        self.api_url = api_url
        self.crashed = False
        self.elapsed_game_time = 0
        self.distance = 0
        self.latest_action = "NOTHING"
        self.ticks = 0

STATE = None


def intersects(rect1, rect2):
    return rect1.colliderect(rect2)

# Game logic
def handle_action(action: str):
    if action == "ACCELERATE":
        STATE.ego.speed_up()
    elif action == "DECELERATE":
        STATE.ego.slow_down()
    elif action == "STEER_LEFT":
        STATE.ego.turn(-0.1)
    elif action == "STEER_RIGHT":
        STATE.ego.turn(0.1)
    else:
        pass

def update_cars():
    for car in STATE.cars:
        car.update(STATE.ego)


def remove_passed_cars():
    min_distance = -1000
    max_distance = SCREEN_WIDTH + 1000
    cars_to_keep = []
    cars_to_retire = []

    for car in STATE.cars:
        if car.x < min_distance or car.x > max_distance:
            cars_to_retire.append(car)
        else:
            cars_to_keep.append(car)

    for car in cars_to_retire:
        STATE.car_bucket.append(car)
        car.lane = None

    STATE.cars = cars_to_keep

def place_car():
    if len(STATE.cars) > LANE_COUNT:
        return

    speed_coeff_modifier = 5
    x_offset_behind = -0.5
    x_offset_in_front = 1.5

    open_lanes = [lane for lane in STATE.road.lanes if not any(c.lane == lane for c in STATE.cars if c != STATE.ego)]
    lane = random_choice(open_lanes)
    x_offset = random_choice([x_offset_behind, x_offset_in_front])
    horizontal_velocity_coefficient = random_number() * speed_coeff_modifier

    car = STATE.car_bucket.pop() if STATE.car_bucket else None
    if not car:
        return

    velocity_x = STATE.ego.velocity.x + horizontal_velocity_coefficient if x_offset == x_offset_behind else STATE.ego.velocity.x - horizontal_velocity_coefficient
    car.velocity = Vector(velocity_x, 0)
    STATE.cars.append(car)

    car_sprite = car.sprite
    car.x = (SCREEN_WIDTH * x_offset) - (car_sprite.get_width() // 2)
    car.y = int((lane.y_start + lane.y_end) / 2 - car_sprite.get_height() / 2)
    car.lane = lane


def get_action():
    """
    Reads pygame events and returns an action string based on arrow keys or spacebar.
    Up: ACCELERATE, Down: DECELERATE, Left: STEER_LEFT, Right: STEER_RIGHT, Space: NOTHING
    """

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


    # Holding down keys
    keys = pygame.key.get_pressed()

    # Priority: accelerate, decelerate, steer left, steer right, nothing
    if keys[pygame.K_RIGHT]:
        return "ACCELERATE"
    if keys[pygame.K_LEFT]:
        return "DECELERATE"
    if keys[pygame.K_UP]:
        return "STEER_LEFT"
    if keys[pygame.K_DOWN]:
        return "STEER_RIGHT"
    if keys[pygame.K_SPACE]:
        return "NOTHING"

    # Just clicking once and it keeps doing it until a new press
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                return "ACCELERATE"
            elif event.key == pygame.K_LEFT:
                return "DECELERATE"
            elif event.key == pygame.K_UP:
                return "STEER_LEFT"
            elif event.key == pygame.K_DOWN:
                return "STEER_RIGHT"
            elif event.key == pygame.K_SPACE:
                return "NOTHING"
    
    
    # If no relevant key is pressed, repeat last action or do nothing
    #return STATE.latest_action if hasattr(STATE, "latest_action") else "NOTHING"
    return "NOTHING"

def get_action_json():
    """
    Get action depending on tick from the actions_log.json.
    Finds the action for the current STATE.ticks.
    """
    try:
        with open("actions_log.json", "r") as f:
            actions = json.load(f)
            for entry in actions:
                if entry.get("tick") == STATE.ticks:
                    return entry.get("action", "NOTHING")
            return "NOTHING"
    except FileNotFoundError:
        return "NOTHING"


def initialize_game_state( api_url: str, seed_value: str, sensor_removal = 0):
    seed(seed_value)
    global STATE
    STATE = GameState(api_url)

    # Create environment
    STATE.road = Road(SCREEN_WIDTH, SCREEN_HEIGHT, LANE_COUNT)
    middle_lane = STATE.road.middle_lane()
    lane_height = STATE.road.get_lane_height()

    # Create ego car
    ego_velocity = Vector(10, 0)
    STATE.ego = Car("yellow", ego_velocity, lane=middle_lane, target_height=int(lane_height * 0.8))
    ego_sprite = STATE.ego.sprite
    STATE.ego.x = (SCREEN_WIDTH // 2) - (ego_sprite.get_width() // 2)
    STATE.ego.y = int((middle_lane.y_start + middle_lane.y_end) / 2 - ego_sprite.get_height() / 2)
    sensor_options = [
            (90, "front"),
            (135, "right_front"),
            (180, "right_side"),
            (225, "right_back"),
            (270, "back"),
            (315, "left_back"),
            (0, "left_side"),
            (45, "left_front"),
            (22.5, "left_side_front"),
            (67.5, "front_left_front"),
            (112.5, "front_right_front"),
            (157.5, "right_side_front"),
            (202.5, "right_side_back"),
            (247.5, "back_right_back"),
            (292.5, "back_left_back"),
            (337.5, "left_side_back"),
        ]

    for _ in range(sensor_removal): # Removes random sensors
        random_sensor = random_choice(sensor_options)
        sensor_options.remove(random_sensor)
    STATE.sensors = [
        Sensor(STATE.ego, angle, name, STATE)
        for angle, name in sensor_options
    ]

    # Create other cars and add to car bucket
    for i in range(0, LANE_COUNT - 1):
        car_colors = ["blue", "red"]
        color = random_choice(car_colors)
        car = Car(color, Vector(8, 0), target_height=int(lane_height * 0.8))
        STATE.car_bucket.append(car)

    STATE.cars = [STATE.ego]

def update_game(current_action: str):
    handle_action(current_action)
    STATE.distance += STATE.ego.velocity.x
    update_cars()
    remove_passed_cars()
    place_car()
    for sensor in STATE.sensors:
        sensor.update()

    return STATE
    
# Main game loop
ACTION_LOG = []

def game_loop(verbose: bool = True, log_actions: bool = True, log_path: str = "actions_log.json"):
    global STATE
    clock = pygame.time.Clock()
    screen = None
    if verbose:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Race Car Game")

    while True:
        delta = clock.tick(60)  # Limit to 60 FPS
        STATE.elapsed_game_time += delta
        STATE.ticks += 1


        if STATE.crashed or STATE.ticks > MAX_TICKS or STATE.elapsed_game_time > MAX_MS:
            print(f"Game over: Crashed: {STATE.crashed}, Ticks: {STATE.ticks}, Elapsed time: {STATE.elapsed_game_time} ms, Distance: {STATE.distance}")
            break

        # Handle action - get_action() is a method for using arrow keys to steer - implement own logic here!
        action = get_action()

        # Log the action with tick
        if log_actions:
            ACTION_LOG.append({"tick": STATE.ticks, "action": action})

        handle_action(action)

        STATE.distance += STATE.ego.velocity.x
        update_cars()
        remove_passed_cars()
        place_car()

        print("Current action:", action)
        print("Currnet tick:", STATE.ticks)

        # Update sensors
        for sensor in STATE.sensors:
            sensor.update()
        
        # Handle collisions
        for car in STATE.cars:
            if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
                STATE.crashed = True
        
        # Check collision with walls
        for wall in STATE.road.walls:
            if intersects(STATE.ego.rect, wall.rect):
                STATE.crashed = True

        # Render game (only if verbose)
        if verbose:
            screen.fill((0, 0, 0))  # Clear the screen with black

            # Draw the road background
            screen.blit(STATE.road.surface, (0, 0))

            # Draw all walls
            for wall in STATE.road.walls:
                wall.draw(screen)

            # Draw all cars
            for car in STATE.cars:
                if car.sprite:
                    screen.blit(car.sprite, (car.x, car.y))
                    bounds = car.get_bounds()
                    color = (255, 0, 0) if car == STATE.ego else (0, 255, 0)
                    pygame.draw.rect(screen, color, bounds, width=2)
                else:
                    pygame.draw.rect(screen, (255, 255, 0) if car == STATE.ego else (0, 0, 255), car.rect)

            # Draw sensors if enabled
            if STATE.sensors_enabled:
                for sensor in STATE.sensors:
                    sensor.draw(screen)

            pygame.display.flip()

    # # Save actions to file after game ends
    # import os
    # if log_actions:
    #     log_dir = os.path.dirname(log_path)
    #     if log_dir and not os.path.exists(log_dir):
    #         os.makedirs(log_dir, exist_ok=True)
    #     with open(log_path, "w") as f:
    #         json.dump(ACTION_LOG, f, indent=2)

# Initialization - not used
def init(api_url: str):
    global STATE
    STATE = GameState(api_url)
    print(f"Game initialized with API URL: {api_url}")


# Entry point
if __name__ == "__main__":
    seed_value = None
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)  # Replace with actual API URL
    game_loop(verbose=True)  # Change to verbose=False for headless mode
    pygame.quit()