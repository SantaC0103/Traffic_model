import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

ACCELERATION_UP = 3
ACCELERATION_DOWN = -6
# Maximum speed of 30m/s, consider as 100km/h in real life
MAX_SPEED = 30
MAX_DISTANCE = 50
TRAFFIC_LIGHTS_SWITCH_TIME = 60

ROUTE_LENGTH = 2000
CROSSOVER_POSITION = 1000


class Vehicle:
    def __init__(self, max_speed=0, min_distance=0.0):
        self.speed = 0
        self.acceleration = 0
        self.pos = 0
        self.start = -1
        self.end = -1
        # Add two parameters
        self.max_speed = max_speed
        self.min_distance = min_distance
        # Add respond time.
        # mode: 0-normal 1-brake early 2-brake late
        self.respond = np.random.randint(0, 3)

    def __repr__(self):
        # This function is used to test
        if (self.start != -1):
            return f"1 direction: {self.direction},  a:{self.acceleration}, speed: {self.speed}, pos: {self.pos}, start: {self.start}, end: {self.end}, {self.max_speed}, {self.min_distance}\n"
        else:
            return f"direction: {self.direction},  a:{self.acceleration}, speed: {self.speed}, pos: {self.pos}, start: {self.start}, end: {self.end}, {self.max_speed}, {self.min_distance}\n"


def init_para():
    speed = MAX_SPEED * np.random.rand(1)[0]
    distance = MAX_DISTANCE * np.random.rand(1)[0]

    if speed == 0:
        speed = 1
    if distance == 0:
        distance = 1
    return speed, distance


def finished(vehicle_list):
    # Determine if all vehicles are off route
    for vehicle in vehicle_list:
        if vehicle.end == -1:
            return False
    return True


def update_vehicle(vehicle_list, crossover, Time):

    for i in range(len(vehicle_list) - 1, -1, -1):
        vehicle = vehicle_list[i]
        
        vehicle_before = None
        if i != 0:
            vehicle_before = vehicle_list[i - 1]
        if vehicle.start != -1 and vehicle.end == -1:
            light = (Time // TRAFFIC_LIGHTS_SWITCH_TIME) % 2
            if vehicle_before and vehicle_before.end == -1:
                if vehicle.speed > vehicle_before.speed:
                    tmp = vehicle_before.speed ** 2
                else:
                    tmp = 0
                distance = (vehicle_before.pos - vehicle.pos) - 2 + (vehicle.speed ** 2 - tmp) / (2 * ACCELERATION_DOWN)
            else:
                distance = 1 << 30
            if vehicle.respond == 1:
                distance -= vehicle.speed
            if vehicle.respond == 2:
                distance += vehicle.speed

            if vehicle.speed != vehicle.max_speed and distance > vehicle.min_distance and vehicle.acceleration <= 0:
                vehicle.acceleration = ACCELERATION_UP

            elif distance <= vehicle.min_distance or (vehicle.pos < crossover + (vehicle.speed ** 2) / (
                    2 * ACCELERATION_DOWN) < vehicle.pos + 10 and light):
                vehicle.acceleration = ACCELERATION_DOWN

            vehicle.speed += vehicle.acceleration

            if vehicle.speed >= vehicle.max_speed:
                vehicle.speed = vehicle.max_speed
                vehicle.acceleration = 0
            if vehicle.speed <= 0:
                vehicle.speed = 0
                vehicle.acceleration = 0

            # Preventing vehicles failure to brake at traffic lights
            prev_pos = vehicle.pos
            vehicle.pos += vehicle.speed

            # Add crash
            if vehicle_before and vehicle.pos > vehicle_before.pos:
                vehicle.pos = vehicle_before.pos
                vehicle_before.speed = vehicle.speed = 0
                vehicle_before.acceleration = vehicle_before.acceleration = 0

            if light and prev_pos < crossover < vehicle.pos:
                vehicle.pos = prev_pos
                vehicle.acceleration = vehicle.speed = 0

            # Vehicle exits lane, save end time
            if vehicle.pos >= ROUTE_LENGTH:
                vehicle.end = Time


def evaluate(vehicle_list, launch_time):
    # In this case, the number of vehicle is not fixed
    vehicle_num = len(vehicle_list)

    # # Initialze all vehicles
    for vehicle in vehicle_list:
        vehicle.start = -1
        vehicle.end = -1
        vehicle.pos = 0
        vehicle.speed = 0
        vehicle.acceleration = 0

    crossover = CROSSOVER_POSITION
    # time
    t = 0
    # Current number of vehicles entered
    i = 0
    
    # Do simulation if there are vehicles in the lane
    while not finished(vehicle_list):
        if i < vehicle_num and t == launch_time[i]:
            vehicle_list[i].start = t
            i += 1

        update_vehicle(vehicle_list, crossover, t)
        t += 1

    # Calculate the average time taken
    duration = sum(map(lambda vehicle: vehicle.end - vehicle.start + 1, vehicle_list)) / vehicle_num
    return duration


def find(value, weight_list, c):
    # Find the probability interval of value
    for i in range(len(weight_list) - 1):
        if weight_list[i] <= value * c < weight_list[i + 1]:
            return i
    return -1


def run(fitness, cost):
    np.random.seed(int(time.time()))
    # Current number of vehicles
    cnt = 0
    # Total number of vehicles
    # The value is not important
    tot_cnt = 10000
    # time
    t = 0

    # Probability of vehicle launch
    p_l = 0.3
    # Probability of mutation operator
    p_m = 0.05
    # Vehicle list
    vehicle_list = []
    # Vehicle launch time
    launch_time = []

    order = []
    weight_list = [0]

    c = 0

    while cnt < tot_cnt:
        rand_value = np.random.rand(1)[0]

        # Vehicles ready to launch
        if rand_value < p_l:
            cnt += 1  # Keep adding the number of vehicles
            launch_time.append(t)
            if cnt <= 10:
                para = init_para()
                vehicle = Vehicle(para[0], para[1])
                vehicle_list.append(vehicle)
            else:
                # Crossover operator
                rand_value = np.random.rand(1)[0]
                pos1 = find(rand_value, weight_list, c)
                rand_value = np.random.rand(1)[0]
                pos2 = find(rand_value, weight_list, c)
                # Select two parent solutions
                sol1 = (vehicle_list[order[pos1]].max_speed, vehicle_list[order[pos1]].min_distance)
                sol2 = (vehicle_list[order[pos2]].max_speed, vehicle_list[order[pos2]].min_distance)

                # Generating new child solutions
                new_distance = (sol1[1] + sol2[1]) / 2
                new_speed = max(sol1[0], sol2[0])

                # Mutation operator
                rand_value = np.random.rand(1)[0]
                if rand_value < p_m:
                    new_speed = new_speed + np.random.normal(0, 3)
                    new_speed = min(max(new_speed, 1), MAX_SPEED)
                    new_distance = new_distance + np.random.normal(0, 3)
                    new_distance = min(max(new_distance, 1), MAX_DISTANCE)
                vehicle = Vehicle(new_speed, new_distance)
                vehicle_list.append(vehicle)

            # Calculate the average time taken after adding one vehicle
            fitness.append(evaluate(vehicle_list, launch_time))
            # Calculate the order
            order = list(pd.Series(fitness).sort_values().index)
            # Calculate the constants
            c = (1 - np.exp(-cnt)) / (1 - np.exp(-1))
            # Calculate the weight list
            weight_list.append(weight_list[-1] + np.exp(1-cnt))
            # Calculate time cost
            time_cost = vehicle_list[-1].end - vehicle_list[-1].start
            cost.append(time_cost)
            print("Vehicle num: {0:>5d}, Last Vehicle time consumption: {1}, {2}".format(cnt, fitness[-1], (vehicle_list[-1].max_speed, vehicle_list[-1].min_distance)))

        t += 1


if __name__ == "__main__":
    fitness = []
    time_consumption = []
    try:
        run(fitness, time_consumption)
    except:
        # Output diagrams when terminate
        plt.plot(fitness)
        plt.xlabel("Vehicle Number")
        plt.ylabel("Fitness")
        plt.show()
        diff = []
        index = []
        for i in range(20, len(fitness)):
            diff.append(fitness[i] - fitness[i - 1])
            index.append(i)
        plt.plot(index, diff)
        plt.xlabel("Vehicle Number")
        plt.ylabel("Average Time Difference")
        plt.show()
        with open('phase2_1.csv', 'w') as f:
            f.write("fitness\n")
            f.write("\n".join(map(str, fitness)))
