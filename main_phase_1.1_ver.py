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
    def __init__(self):
        # The direction of travel of the vehicle, 
        # Takes values of 0 and 1, with 0 for west to east and 1 for north to south.
        self.direction = -1
        self.speed = 0
        self.acceleration = 0
        # Position of the vehicle
        self.pos = 0
        # Time for vehicle start to enter the lane
        self.start = -1
        # Time for vehicle to exit lane (end time)
        self.end = -1
        # Add respond time.
        # mode: 0-normal 1-brake early 2-brake late
        self.respond = np.random.randint(0, 3)

    def __repr__(self):
        # This function is used to test
        if (self.start != -1):
            return f"direction: {self.direction},  a:{self.acceleration}, speed: {self.speed}, pos: {self.pos}, start: {self.start}, end: {self.end}\n"
        else:
            return ""


def init_pop(population, pop_size):
    # Initialize the population
    for i in range(pop_size):
        # Randomisation vehicle speed and distance
        speed = MAX_SPEED * np.random.rand(1)[0]
        distance = MAX_DISTANCE * np.random.rand(1)[0]
       
        if speed == 0:
            speed = 1
        if distance == 0:
            distance = 1
        population.append((speed, distance))
    return population


def finished(vehicle_list):
    # Determine if all vehicles are off route
    for vehicle in vehicle_list:
        if vehicle.end == -1:
            return False
    return True


def update_vehicle(vehicle_list, crossover, Time, max_speed, min_distance):
    '''
    updates the vehicle's travel status at Time.
    The incoming parameters are the list of vehicles, the position of the crossroad, the current point in time, 
    the maximum permissible speed and the minimum permissible distance between vehicles.

    As the model is single lane, we only need to consider the previous car that is closest to the current vehicle.
    In vehicle_list，the smaller the subscript, the more forward the vehicle is on the road.
    So it is treated from back to front to consider the driver's reaction time.
    '''

    for i in range(len(vehicle_list)-1, -1, -1):  # This loop is designed to process from back to front
        vehicle = vehicle_list[i]

        # The car in front of the current car.
        # If the current car is the first, there is no car in front of it
        vehicle_before = None
        if i != 0:
            vehicle_before = vehicle_list[i - 1]
        if vehicle.start != -1 and vehicle.end == -1:
            # Car has entered the lane

            # Determine the traffic light in the direction of the current vehicle, 
            # if green, release vehicle
            light = ((Time // TRAFFIC_LIGHTS_SWITCH_TIME) % 2) ^ vehicle.direction  # # light = 1 then red, otherwise green
            if vehicle_before and vehicle_before.end == -1:
                # Calculate the distance between the current car and the car in front.
                # Remember it takes some distance to brake from the current speed.
                distance = (vehicle_before.pos - vehicle.pos) - 2 + (vehicle.speed ** 2) / (2 * ACCELERATION_DOWN)
            else:
                distance = 1 << 30
            if vehicle.respond == 1:
                distance -= vehicle.speed
            if vehicle.respond == 2:
                distance += vehicle.speed

            # When the speed has not reached maximum speed, 
            # and the distance between the two vehicles is less than the safe distance,
            # and the acceleration conditions allow, the vehicle can accelerate.
            if vehicle.speed != max_speed and distance > min_distance and vehicle.acceleration <= 0:
                vehicle.acceleration = ACCELERATION_UP
            # Otherwise, deceleration
            elif distance <= min_distance or (vehicle.pos < crossover + (vehicle.speed ** 2) / (2 * ACCELERATION_DOWN) < vehicle.pos + 10 and light):
                vehicle.acceleration = ACCELERATION_DOWN

            vehicle.speed += vehicle.acceleration

            if vehicle.speed >= max_speed:
                vehicle.speed = max_speed
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


def evaluate(sol):

    vehicle_num = 100
    # Take the parameter combination (max_speed, min_distance) as solution
    max_speed, min_distance = sol
    # Randomly generate the time for vehicles to enter the lane
    launch_time = np.random.choice(a=vehicle_num * 3, size=vehicle_num, replace=False)
    launch_time = sorted(launch_time)
    # Randomly generate the direction
    direction = np.random.randint(2, size=vehicle_num)

    # Initialze the vehicle list
    vehicle_list = []
    for i in range(vehicle_num):
        vehicle = Vehicle()
        vehicle_list.append(vehicle)

    crossover = CROSSOVER_POSITION
    # time
    t = 0
    # Current number of vehicles entered
    i = 0

    # Do simulation if there are vehicles in the lane
    while not finished(vehicle_list):
        if i < vehicle_num and t == launch_time[i]:

            vehicle_list[i].start = t
            vehicle_list[i].direction = direction[i]
            vehicle_list[i].pos = 0
            i += 1

        update_vehicle(vehicle_list, crossover, t, max_speed, min_distance)
        t += 1
    # Calculate the average time taken
    duration = sum(map(lambda vehicle: vehicle.end - vehicle.start + 1, vehicle_list)) / vehicle_num
    return duration


def find(value, weight_list):
    # Find the probability interval of value
    for i in range(len(weight_list) - 1):
        if weight_list[i] <= value < weight_list[i + 1]:
            return i + 1
    return -1


def GA():

    # Set random seed to time seed
    np.random.seed(int(time.time()))
    # Population list
    pop = []
    # Population size
    pop_size = 10
    # Probability of crossover operator
    p_c = 0.9
    # Probability of mutation operator.
    p_m = 0.1
    # Evolutionary generation.
    # This value can be lower if you want to end the evolution earlier
    iteration_generations = 10000

    # Initialize the population
    pop = init_pop(pop, pop_size)
    all_pop_list = []
    all_fitness_list = []

    # Calculate constants
    c = pop_size - (1 - np.exp(-pop_size)) / (1 - np.exp(-1))
    # Building weight lists
    weight_list = []
    s = 0
    for i in range(pop_size):
        s += 1 - np.exp(-i)
        weight_list.append(s / c)

    # Ranking of solutions
    order = []
    # Generations
    gen = 0
    while gen <= iteration_generations:
        # Evaluate all solutions in the population
        fitness = list(map(evaluate, pop))

        all_pop_list += pop
        all_fitness_list += fitness

        df = pd.DataFrame(all_pop_list, all_fitness_list)
        df.to_csv("GA_out.csv")

        fitness = pd.Series(fitness)
        # Order all solutions
        order = list(fitness.sort_values().index)

        new_pop = []
        for i in range(pop_size):
            # Confirm which operator to use (p_c or p_m)
            status = np.random.rand(1)[0]

            if status < p_c:
                # Crossover operator
                rand_value = np.random.rand(1)[0]
                pos1 = pop_size - find(rand_value, weight_list) - 1
                rand_value = np.random.rand(1)[0]
                pos2 = pop_size - find(rand_value, weight_list) - 1
                # Select two parent solutions
                sol1 = pop[order[pos1]]
                sol2 = pop[order[pos2]]

                # Generating new child solutions
                new_speed = max(sol1[0], sol2[0])
                new_distance = (sol1[1] + sol2[1]) / 2
                new_pop.append((new_speed, new_distance))
            else:
                # mutation operator
                rand_value = np.random.rand(1)[0]
                pos = pop_size - find(rand_value, weight_list) - 1
                par_sol = pop[order[pos]]
                if rand_value < p_m:
                    # Add a normal random disturbance
                    new_speed = par_sol[0] + np.random.normal(0, 3)
                    new_speed = min(max(new_speed, 1), MAX_SPEED)
                    new_distance = par_sol[1] + np.random.normal(0, 3)
                    new_distance = min(max(new_distance, 1), MAX_DISTANCE)
                    new_pop.append((new_speed, new_distance))
                else:
                    new_pop.append(par_sol)

        # The current child becomes the parent of the next generation
        pop = new_pop
        gen += 1
        print("Gen: {0:>6d}, Current minimal average time: {1}".format(gen, min(fitness)))

    return pop[order[0]]


if __name__ == "__main__":
    print(GA())
    '''
    interval = 5
    fitness = []
    speed = []
    for i in range(5 * (interval), (30 * interval) + 1):
        s = 0
        for j in range(5):
            s += evaluate((i / interval, 2))
        fitness.append(s / 5)
        print('distance', i / interval)
        speed.append(i / interval)
    print(speed)
    print(fitness)
    df = pd.DataFrame(speed, fitness)
    df.to_csv("distance.csv")

    plt.plot(speed, fitness)  
    plt.xlabel('Speed')
    plt.ylabel('Average time(Fitness)')
    plt.show()
    '''


