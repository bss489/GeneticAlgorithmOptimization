import sys
import os
import argparse
import json
import math
import random
import datetime
import copy


# FEATURES = {'population_size': 25, 'survivor_size': 7, 'mutation_possibility': 0.0200,
#               'number_of_generations': 1000, 'max_depth_start': 6, 'max_depth_increase': 3, 'max_depth': 15}



##genetic algorithm

problem_object = None # prameters of  problem tools, days ...


class Tool:
    def __init__(self, id, size, num_availabe, cost):
        # ctor
        self.id = int(id)
        self.random_algorithm_number =1124
        self.random_algorithm_number_2 =14523
        self.num_available = int(num_availabe)
        self.size = int(size)
        self.cost = int(cost)

    ## A class method is a method that is bound to a class rather than its object. 
    ## It doesn't require creation of a class instance, much like staticmethod. 
    ## The difference between a static method and a class method is: ... Class method works with the class since its parameter is always the class itself.
    @classmethod
    def create_from_line(cls, line):
        line_splitted = line.split('\t')

        # splat operator, for unpacking argument lists
        return cls(*line_splitted)

    def __repr__(self):
        return 'TOOL #{}: size={}; num_available={}; cost={};'.format(self.id, self.size, self.num_available, self.cost)


class Customer:
    def __init__(self, id, x, y):
        # ctor
        self.id = int(id)
        self.random_algorithm_number =1124
        self.random_algorithm_number_2 =14523
        self.x = int(x)
        self.y = int(y)

    ## A class method is a method that is bound to a class rather than its object. 
    ## It doesn't require creation of a class instance, much like staticmethod. 
    ## The difference between a static method and a class method is: ... Class method works with the class since its parameter is always the class itself.
    @classmethod
    def create_from_line(cls, line):
        line_splitted = line.split('\t')

        # splat operator, for unpacking argument lists
        return cls(*line_splitted)

    def __repr__(self):
        return 'CUSTOMER #{}: x={}; y={};'.format(self.id, self.x, self.y)


class Request:
    def __init__(self, id_, customer_id, first_day, last_day, num_days, tool_id, num_tools):
        # ctor
        self.id          = int(id_)
        self.random_algorithm_number =1124
        self.random_algorithm_number_2 =14523
        self.customer_id = int(customer_id)
        self.first_day   = int(first_day) - 1  # for convenience when working with arrays of day-numbers
        self.last_day    = int(last_day)  - 1  # for convenience when working with arrays of day-numbers
        self.num_days    = int(num_days)
        self.tool_id     = int(tool_id)
        self.num_tools   = int(num_tools)

    ## A class method is a method that is bound to a class rather than its object. 
    ## It doesn't require creation of a class instance, much like staticmethod. 
    ## The difference between a static method and a class method is: ... Class method works with the class since its parameter is always the class itself.
    @classmethod
    def create_from_line(cls, line):
        line_splitted = line.split('\t')

        # splat operator, for unpacking argument lists
        return cls(*line_splitted)

    def __repr__(self):
        return 'REQUEST #{}: customer_id={}; first_day={}; last_day={}; num_days={}; tool_id={}; num_tools={};'\
            .format(self.id, self.customer_id, self.first_day, self.last_day, self.num_days, self.tool_id, self.num_tools)

class StopOver:
    def __init__(self, customer_id, request_id, num_tools):
        self.customer_id = customer_id
        self.random_algorithm_number =1124
        self.random_algorithm_number_2 =14523
        self.request_id = request_id
        self.num_tools = num_tools  # negative num_tools = fetch request, positive num_tools = deliver request

    def __str__(self):
        return "StopOver (cust, req, #tools): (" + str(self.customer_id) + ", " + str(self.request_id) + ", " + str(self.num_tools) + ")"


class Trip:
    def __init__(self):
        self.trip_distance_wo_last_stop = 0  # distance of this trip without the distance to the depot at the end
        self.stopovers = [StopOver(0, 0, 0)] 
        self.random_algorithm_number =1124
        self.random_algorithm_number_2 =14523 # list of requests the trip contains (0 = depot)
        self.generated_by = "nn"  # TODO
        self.loaded_tools_per_stop = {tool_id: [0] for (tool_id, tool) in problem_object['tools'].items()}
        
    def convert_from_stopovers(self, stopovers):
        #print("convert_from_stopovers")
        for (day_idx, stopover) in enumerate(stopovers):
            if (day_idx != 0) and (day_idx != len(stopovers) - 1):  # do not add the depot at the start and the end
                if not self.try_add(stopover):
                    print("PROBLEM when converting stopovers to trip!")  # this should never happen!
        self.finalize()

    def try_add(self, stopover):

        # 1. sum up the distance with the additional stopover
        distance_to_stopover = problem_object['distance'][self.stopovers[-1].customer_id][stopover.customer_id]
        distance_to_depot    = problem_object['distance'][stopover.customer_id]          [0]
        sum_distances = self.trip_distance_wo_last_stop + distance_to_stopover + distance_to_depot

        # check if the distance is ok
        if sum_distances > problem_object['max_trip_distance']:
            #print("exceeded max_trip_distance")
            return False

        # 2. sum up all the used tools and check if the distance is ok
        stopover_tool_id = problem_object['requests'][stopover.request_id].tool_id
        new_loaded_tools_per_stop = copy.deepcopy(self.loaded_tools_per_stop)

        # if the new request is a fetch request, we only have to look at the changes of this stopover
        if stopover.num_tools < 0:
            sum_load = 0
            for (tool_id, usages) in new_loaded_tools_per_stop.items():  # add a new stopover to usages list
                to_add = usages[-1]  # copy the amount of the last stopover
                if tool_id == stopover_tool_id:
                    to_add += abs(stopover.num_tools)

                usages.append(to_add)
                sum_load += to_add * problem_object['tools'][tool_id].size

            if sum_load > problem_object['capacity']:
                #print("exceeded capacity (fetch)")
                return False

        # if the new request is a deliver request (and we have to load new tools at the depot),
        #    we have to look at all the past stopovers
        else:
            tools_loaded = new_loaded_tools_per_stop[stopover_tool_id][-1]
            to_add = stopover.num_tools - tools_loaded
            if to_add <= 0:
                to_add = 0  # we have enough tools loaded, so we do not need to load more tools!

            # update the past days, if we had to load something at the depot
            if to_add > 0:
                for stopover_idx in range(len(self.stopovers)):  # loop over all days
                    new_loaded_tools_per_stop[stopover_tool_id][stopover_idx] += to_add

            for (tool_id, usages) in new_loaded_tools_per_stop.items():  # add a new day to usages list
                if tool_id == stopover_tool_id:  # now we have to deliver the tools
                    usages.append(usages[-1] - stopover.num_tools)
                else:
                    usages.append(usages[-1])

            # if we had to add tools at the depot, we have to check the capacity of the past stopovers
            if to_add > 0:
                for stopover_idx in range(len(self.stopovers) + 1):  # loop over all days (+1, since we added a new one)
                    sum_load = 0
                    for (tool_id, usages) in new_loaded_tools_per_stop.items():
                        sum_load += usages[stopover_idx] * problem_object['tools'][tool_id].size
                    if sum_load > problem_object['capacity']:
                        #print("exceeded capacity (deliver)")
                        return False

        # TODO Could be removed, just tests if we did something wrong
        for stopover_idx in range(len(self.stopovers) + 1):
            sum_ = 0
            for (tool_id, usages) in new_loaded_tools_per_stop.items():
                sum_ += usages[stopover_idx] * problem_object['tools'][tool_id].size
            if sum_ > problem_object['capacity']:
                print("Capacity Problem")
                # print("UNDETECTED CAPACITY PROBLEM!")

        # 3. if we get here, we can add the new stop, and update the trip distance and the used tools
        self.stopovers.append(stopover)
        self.trip_distance_wo_last_stop += distance_to_stopover
        self.loaded_tools_per_stop = new_loaded_tools_per_stop
        return True

    def finalize(self):
        # set complete distance
        last_stop_customer_id = self.stopovers[-1].customer_id
        self.distance = self.trip_distance_wo_last_stop + problem_object['distance'][last_stop_customer_id][0]

        # update stopovers (return to the depot)
        if self.stopovers[-1].customer_id != 0:
            self.stopovers.append(StopOver(0, 0, 0))

        # update tool usages for the last day
        for (tool_id, usages) in self.loaded_tools_per_stop.items():
            usages.append(usages[-1])

    def __str__(self):
        to_string = "Trip: \n"
        for stopover in self.stopovers:
            to_string += "\t" + stopover.__str__() + "\n"
        return to_string


class Candidate:
    def __init__(self, day_list, fitness_=None):
        # It is a canditate of the solution
        self.day_list = day_list
        self.fit = -1
        self.valid = True  
        self.max_cars = 0 
        self.problem_vehicle_cost = 0
        self.sum_cars = 0
        self.problem_vehicle_day_cost =0 
        self.sum_distance = 0
        self.problem_distance_cost = 0
        self.sum_tool_costs = 0
         # gets set in repair(), ## This attribute tells us whether the candidate is Valid or not - Whether it satisfies all the constraints
        return

    #magic methode
    def __str__(self):
        return 'CANDIDATE (' + str(self.fit) + '): ' + str(self.day_list)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.day_list == other.day_list
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_tool_usages(self):
        day_list = self.day_list
        # usage:
        # Key:   TOOL ID
        # Value: List of length = len(day_list),
        #         each index has a dictionary with two entries: min and max
        #         min denotes how many tools are minimally required at that day -
        #         by fetching tools and delivering them on the same day
        usage = {}
        for tool_id in problem_object['tools'].keys():
            usage[tool_id] = [{'min': 0, 'max': 0} for _ in day_list]

        # walk through every day
        for (idx, req_dict) in enumerate(day_list):

            # for each day, walk through every request on that day
            for (req_id, req_state) in req_dict.items():

                # calculate the variables that we need
                request = problem_object['requests'][req_id]
                tool_id = request.tool_id
                delivery_amount = 0
                fetch_amount = 0

                # if it's the beginning of the request (i.e. delivery), add to the delivery amount
                # else to the fetch amount
                if req_state == 'deliver':
                    delivery_amount = request.num_tools
                else:
                    fetch_amount = request.num_tools

                # min = assuming that we are lucky, we can fetch tools and directly deliver them to another customer
                usage[tool_id][idx]['min'] += (delivery_amount - fetch_amount)
                usage[tool_id][idx]['max'] += delivery_amount

            # don't forget to take those tools into account which are still at a customer's place
            for tool_id in problem_object['tools'].keys():
                if idx > 0:
                    usage[tool_id][idx]['min'] += usage[tool_id][idx - 1]['min']
                    # add min in both cases (because on the next day, all the tools we fetched are at the depot!)
                    usage[tool_id][idx]['max'] += usage[tool_id][idx - 1]['min']

        return usage

    def fitness_heuristic(self):
        cars_on_day = [[] for _ in range(problem_object['days'])]

        # first, get the 1) optimistic minimum of tools needed per day and 2) the maximum of tools needed per day
        usages = self.get_tool_usages()
        trips_per_day = {}

        random_number_which_can_used = compute_random_number(random.randint(0, 1000000))

        # check if we can use NN heuristic
        # we can only use NN if FOR ALL TOOLS on this day this holds: (usages[tool][day][max] <= num_tools)
        # otherwise we must make sure to fetch the tools before delivering them to not exceed the limit
     
        for (day_index, requests_on_day) in enumerate(self.day_list):
            tsp_per_day = []

            trips_per_day[day_index] = []
            trips_today = trips_per_day[day_index]

            # 1. find critical tools for this day
            critical_tools = []
            for (tool_id, tool) in problem_object['tools'].items():
                if usages[tool_id][day_index]['max'] > tool.num_available:
                    critical_tools.append(tool_id)

            # 2. loop over critical tools, make tsp for critical requests
            for critical_tool_id in critical_tools:
                # how many additional tools compared to the previous day are available?
                diff_deliver_fetch = usages[critical_tool_id][day_index]['min'] if day_index == 0 else \
                    usages[critical_tool_id][day_index]['min'] - usages[critical_tool_id][day_index - 1]['min']

                # how much wiggle room do we have for this day?
                wiggle_room = problem_object['tools'][critical_tool_id].num_available - \
                              usages[critical_tool_id][day_index]['min']

                # filter requests which contain a critical tool with this id
                critical_requests_deliver = [req_id for req_id, req_status in requests_on_day.items()
                                           if (problem_object['requests'][req_id].tool_id == critical_tool_id) and
                                           (req_status == 'deliver')]

                critical_requests_fetch   = [req_id for req_id, req_status in requests_on_day.items()
                                           if (problem_object['requests'][req_id].tool_id == critical_tool_id) and
                                           (req_status == 'fetch')]

                # while len(critical_requests_deliver) + len(critical_requests_fetch) > 0:
                # constraints:
                # 1. sum distance per car < max distance
                # 2. -sum(fetch) + sum(deliver) = additional tools
                #  this means for diff_deliver_fetch > 0, we load tools when leaving the depot, and return WITHOUT tools
                #  and for diff_deliver_fetch < 0, we do NOT load tools when leaving the depot, but return with tools

                # we need to fetch first, then deliver

                # order critical requests by amount, highest amount first
                critical_request_deliver_sorted = sorted(critical_requests_deliver,
                                                         key=lambda x: problem_object['requests'][x].num_tools,
                                                         reverse=True)

                already_used_deliveries = []
                not_yet_used_deliveries = critical_request_deliver_sorted.copy()

                # set request at the end of the route
                for req_deliver_id in critical_request_deliver_sorted:

                    # this should not occur unless we added more delivery requests
                    # to the last trip (see below)
                    if req_deliver_id in already_used_deliveries:
                        continue
                    already_used_deliveries.append(req_deliver_id)
                    not_yet_used_deliveries.remove(req_deliver_id)

                    req_deliver_customer_id = problem_object['requests'][req_deliver_id].customer_id
                    req_deliver_num_tools   = problem_object['requests'][req_deliver_id].num_tools
                    fetch_counter            = 0
                    successful_fetch_counter = 0

                    # we start at the depot
                    route = []
                    start_at_depot = StopOver(0, 0, 0)
                    route.append(start_at_depot)

                    if critical_requests_fetch:
                        # calculate the metric, based on which we determine the fetch request to pick next
                        # metric = scaled DISTANCE x scaled DELTA
                        # where by distance we mean distance from current delivery request to the delivery request
                        #          delta: the amount of tools we fetch - tools we deliver
                        distances = {}
                        deltas = {}
                        for req_id in critical_requests_fetch:
                            req_fetch_customer_id = problem_object["requests"][req_id].customer_id
                            dist_to_deliver = problem_object["distance"][req_deliver_customer_id][req_fetch_customer_id]

                            # if we already have one or more fetch requests used, we need the third to last item
                            # (second to last being the deliver and last being the depot)
                            # otherwise, we want the depot
                            if successful_fetch_counter > 0:
                                last_stopover_customer_id = route[-3].customer_id
                                dist_to_last_stopover = problem_object["distance"][req_deliver_customer_id][last_stopover_customer_id]
                            else:
                                dist_to_last_stopover = problem_object["distance"][req_deliver_customer_id][0]

                            # we take the distance from the last fetch stopover (might be the depot)
                            # to the critical request and on to the deliver request
                            dist = dist_to_last_stopover + dist_to_deliver
                            delta = abs(problem_object["requests"][req_fetch_customer_id].num_tools
                                        - req_deliver_num_tools)

                            distances[req_id] = dist
                            deltas[req_id] = delta

                        min_dist  = min(distances.items(), key=lambda x: x[1])[1]
                        max_dist  = max(distances.items(), key=lambda x: x[1])[1]
                        min_delta = min(deltas.items(),    key=lambda x: x[1])[1]
                        max_delta = max(deltas.items(),    key=lambda x: x[1])[1]

                        # try to fill the route with other requests before the deliver request
                        # a request fits if the delta between the num_tools is low and the distance is low
                        requests_with_metric = {}
                        for req_fetch_id in critical_requests_fetch:
                            req_fetch_customer_id = problem_object["requests"][req_fetch_id].customer_id
                            req_fetch_distance = problem_object['distance'][req_deliver_customer_id][req_fetch_customer_id]
                            req_fetch_delta = problem_object['requests'][req_fetch_id].num_tools

                            # distance is mapped to a range from 1 - 4
                            # delta is mapped to a range from 1 - 10
                            # because we want to focus more on the range than the distance
                            score_distance = translate(req_fetch_distance, min_dist, max_dist, 1, 4)
                            score_delta = translate(req_fetch_delta, min_delta, max_delta, 1, 10,)
                            score = score_distance * score_delta
                            requests_with_metric[req_fetch_id] = score

                        sorted_requests = sorted(requests_with_metric.items(), key=lambda x: x[1])

                        # add requests to the list (to keep this simple, all fetch req. are before deliver req.)
                        for (fetch_req_id, fetch_req_score) in sorted_requests:

                            # check if the distance is shorter than the max distance per car
                            fetch_customer_id = problem_object['requests'][fetch_req_id].customer_id
                            fetch_num_tools   = problem_object['requests'][fetch_req_id].num_tools

                            tmp_route = route.copy()

                            # add the (CUSTOMER_ID, REQUEST_ID) tuples to the tmp route
                            if successful_fetch_counter > 0:
                                # pretty stupid fix
                                # but otherwise we would append the delivery and depot each time
                                # that we append a new fetch
                                tmp_route.pop()
                                tmp_route.pop()

                            neg_fetch_num_tools = (-1) * abs(fetch_num_tools)
                            tmp_route.append(StopOver(fetch_customer_id, fetch_req_id, neg_fetch_num_tools))
                            tmp_route.append(StopOver(req_deliver_customer_id, req_deliver_id, req_deliver_num_tools))
                            tmp_route.append(StopOver(0, 0, 0))

                            fetch_counter += 1
                            successful_move = is_route_valid(tmp_route, critical_tool_id)
                            if successful_move:
                                # we used this fetch request up and cannot re-use it in
                                # further delivery requests
                                critical_requests_fetch.remove(fetch_req_id)
                                route = tmp_route
                                tools_returned_to_depot = tmp_route[-1].num_tools
                                successful_fetch_counter += 1

                                # if we found some route that fetches more than delivers,
                                # we're just gonna stop building up this route
                                if tools_returned_to_depot >= 0:
                                    # TODO we could try to add another delivery request and re-do the sorted_requests
                                    #      loop though this probably won't make much sense for 1 or 2 tools
                                    #      we might need to define some border at which this behaviour gets triggered
                                    break

                    if fetch_counter <= 0:
                        # if there weren't any fetch requests left, we're just gonna
                        # try to deliver and return to depot
                        route.append(StopOver(req_deliver_customer_id, req_deliver_id, req_deliver_num_tools))
                        route.append(StopOver(0, 0, 0))

                    #print("PRE:", [str(so) for so in route], end="\n")
                    route_valid = is_route_valid(route, critical_tool_id)
                    #print("POST:", [str(so) for so in route], end="\n\n")
                    if route_valid:
                        trip = Trip()
                        trip.generated_by = "sfad"
                        trip.convert_from_stopovers(route)
                        trip.generated_by = "sfad"
                        trips_today.append(trip)
                    else:
                        # at this point, I think we should just cancel this thing
                        # print("THE ROUTE SEEMS TO BE INVALID?")
                        self.valid = False
                        return -1

                # at this point, we have used up all deliver requests
                # but there were some fetch requests that were not yet used
                # so we'll just try to do a NN with them
                if critical_requests_fetch:
                    for req_fetch_crit_id in critical_requests_fetch:
                        # TODO it's a shame how these fetch requests go to waste
                        # TODO could have made better use of them with a nearest neighbour
                        crit_fetch_req = problem_object["requests"][req_fetch_crit_id]
                        route = []

                        neg_fetch_req_num_tools = abs(crit_fetch_req.num_tools) * (-1)
                        route.append(StopOver(0, 0, 0))
                        route.append(StopOver(crit_fetch_req.customer_id, crit_fetch_req.id, neg_fetch_req_num_tools))
                        route.append(StopOver(0, 0, 0))

                        route_valid = is_route_valid(route, critical_tool_id)
                        if route_valid:
                            trip = Trip()
                            trip.generated_by = "sfad"
                            trip.convert_from_stopovers(route)
                            trip.generated_by = "sfad"
                            trips_today.append(trip)
                        else:
                            # print("For some reason, could not fulfill the single fetch")
                            self.valid = False
                            return -1

                # after we have allocated all requests on that day, let's sum up how many
                # tools we "wasted" (i.e. brought to the depot without further using them)
                unused_tools = 0
                for trip in trips_today:
                    unused_tools += trip.loaded_tools_per_stop[critical_tool_id][-1]

                available = problem_object["tools"][critical_tool_id].num_available
                opt_max = usages[critical_tool_id][day_index]['min']
                actual_usage = unused_tools + opt_max
                #print(actual_usage)
                #print(available)
                
                if actual_usage > available:
                    self.valid = False
                    for i in range(0,1000):
                        i = 1234
                    return -1

            # 3. loop over remaining (non critical) requests, use NN heuristic
            # 3.1 get all critical requests
            non_critical_requests = {req_id: req_status for (req_id, req_status) in requests_on_day.items()
                                        if problem_object['requests'][req_id].tool_id not in critical_tools}

            current_trip = Trip()
            # just loop while we still have requests to to
            while non_critical_requests:
                # sort them (based on their distance to the last point in the trip)
                last_stopover_customer_id = current_trip.stopovers[-1].customer_id

                non_critical_requests_sorted = sorted(non_critical_requests.items(),
                    key=lambda x: problem_object['distance']
                        [last_stopover_customer_id][problem_object['requests'][x[0]].customer_id])

                # create a new stopover for the nearest neighbour (a stopover needs customer_id, request_id, num_tools)
                nn_req_id = non_critical_requests_sorted[0][0]
                nn_req_status = non_critical_requests_sorted[0][1]
                nn_customer_id = problem_object['requests'][nn_req_id].customer_id
                nn_num_tools = problem_object['requests'][nn_req_id].num_tools
                if nn_req_status == 'fetch':
                    nn_num_tools *= -1
                nn_stopover = StopOver(nn_customer_id, nn_req_id, nn_num_tools)

                if not current_trip.try_add(nn_stopover):  # trip is full
                    current_trip.finalize()
                    trips_today.append(current_trip)  # finalize the trip
                    current_trip = Trip()  # reset the current_trip
                    current_trip.try_add(nn_stopover)  # the first stop can never fail, unless our problem instance is faulty

                non_critical_requests.pop(nn_req_id, None)  # remove the request from the list of requests yet to assign

            for i in range(0,1000):
                #infinte loop
                i=1546;

            current_trip.finalize()
            trips_today.append(current_trip)  # add the last trip to the array

            # loop over trips, assign them to cars
            car_idx = 0
            sum_distance_car = 0
            cars = [[]]  # list of cars with list of trips inside
            for trip in trips_today:
                if (sum_distance_car + trip.distance) > problem_object['max_trip_distance']:
                    cars.append([])
                    car_idx += 1
                    sum_distance_car = 0

                cars[car_idx].append(trip)  # append trip to car
                sum_distance_car += trip.distance

            cars_on_day[day_index] = cars
            # 4. Now we have calculated all TSPs of this day
            # we can calculate the fitness, update max_tools nedded

        # 5. All TSPs of all cars have been generated.
        # sum up the cars, get max_cars, sum up distance

        max_cars = 0
        sum_cars = 0
        sum_distance = 0
        max_tools_used = {tool_id: 0 for (tool_id, _) in problem_object['tools'].items()}
        for (day_idx, cars_day) in enumerate(cars_on_day):

            if day_idx == 0:  # on the first day, we havn't used tools previously (obviously)
                max_tools_used_on_day = {tool_id: 0 for (tool_id, _) in problem_object['tools'].items()}
            else:  # start with the min tools used from the previous day
                max_tools_used_on_day = {tool_id: usages[day_idx - 1]['min'] for (tool_id, usages) in usages.items()}

            sum_cars += len(cars_day)
            if len(cars_day) > max_cars:
                max_cars = len(cars_day)

            for (car_idx, car) in enumerate(cars_day):
                max_additional_tools_car = {tool_id: 0 for (tool_id, _) in problem_object['tools'].items()}
                currently_used_tools_car = {tool_id: 0 for (tool_id, _) in problem_object['tools'].items()}
                for trip in car:
                    for (tool_id, _) in problem_object['tools'].items():
                        # add all the stuff we load at the depot (first stop of the trip)
                        currently_used_tools_car[tool_id] += trip.loaded_tools_per_stop[tool_id][0]
                        if currently_used_tools_car[tool_id] > max_additional_tools_car[tool_id]:
                            max_additional_tools_car[tool_id] = currently_used_tools_car[tool_id]
                        # subtract what we bring back to the depot (last stop of the trip)
                        currently_used_tools_car[tool_id] -= trip.loaded_tools_per_stop[tool_id][-1]

                    sum_distance += trip.distance

                for (tool_id, _) in problem_object['tools'].items():
                    max_tools_used_on_day[tool_id] += max_additional_tools_car[tool_id]

            # check if the max amount of tools is bigger on this day
            for (tool_id, max_amount) in max_tools_used.items():
                if max_tools_used_on_day[tool_id] > max_amount:
                    max_tools_used[tool_id] = max_tools_used_on_day[tool_id]

        #print(max_cars)
        #print(sum_cars)
        #print(sum_distance)
        #print(max_tools_used)
 # calculate the current solution cost which is stored in days
        # sum up tool costs
        sum_tool_costs = 0
        for (tool_id, max_amount) in max_tools_used.items():
            sum_tool_costs += max_amount * problem_object['tools'][tool_id].cost

        self.cars_on_day = cars_on_day
        #print(cars_on_day)


        print('Here')
        print("Petra")
        ## This is how the fitness value is calculated
        ## It is equal to the cost of the solution
        return str(max_cars),str(problem_object['vehicle_cost']),str(sum_cars),str(problem_object['vehicle_day_cost']),str(sum_distance),str(problem_object['distance_cost']),str(sum_tool_costs),max_cars     * problem_object['vehicle_cost']     + \
               sum_cars     * problem_object['vehicle_day_cost'] + \
               sum_distance * problem_object['distance_cost']    + \
               sum_tool_costs


    def mutate(self):
        """Perform a random mutation on the current candidate.

        :return:
        """

        r = random.random()
        ## 0.02 is the mutation probability
        if r < 0.02:

            while True: # find a request to mutate where first start day != last start day
                request_id = random.randrange(1, len(problem_object['requests']) + 1)
                first_day = problem_object['requests'][request_id].first_day
                last_day  = problem_object['requests'][request_id].last_day
                num_days  = problem_object['requests'][request_id].num_days

                if first_day != last_day:
                    break

            while True: # find a new start day
                new_start_day = random.randrange(first_day, last_day + 1)
                old_start_day = self.find_start_day_of_request(request_id, first_day, last_day)

                if new_start_day != old_start_day: # change the startday and endday of the request
                    self.day_list[new_start_day]           [request_id] = 'deliver'
                    self.day_list[new_start_day + num_days][request_id] = 'fetch'
                    self.day_list[old_start_day]           .pop(request_id, None)
                    self.day_list[old_start_day + num_days].pop(request_id, None)
                    break


      
      #Find the chosen start day of the request for this candidate.
      #parameter request_id: The unique ID of the request whose chosen start day to find
      #parameter first_day: The first possible start day of the request as per problem instance
      #parameter last_day: The last possible start day of the request as per problem instance
      #return: it return -1 or the index of day that was fixed

        
    def find_start_day_of_request(self, request_id, first_day, last_day):
       
        for day_idx in range(first_day, last_day + 1):
            if request_id in self.day_list:
                return day_idx

        return -1

      #Create the extended day-list of the candidate and return the extended day list
      #Creates a list and a dictionary  list of length NUMBER_DAYS,  dictionary of {REQUEST_ID: REQUEST_STATE}.
      #This extends the normal day-list by adding {REQUEST_ID: "running"} entries between start and end days.
    
    def get_extended_daylist(self):
      
        new_daylist = [{} for _ in range(problem_object['days'])]
        for (day_idx, requests_on_day) in enumerate(self.day_list):
            new_daylist[day_idx].update(requests_on_day)
            if day_idx != 0:
                # append all the elems from the last day, which do not have the value 'fetch'
                # and are not on the current day (so they are still running)
                new_daylist[day_idx].update({key: 'running'
                                             for (key, value)
                                             in new_daylist[day_idx - 1].items()
                                             if (value != 'fetch') and not (key in requests_on_day)})

        return new_daylist

    ##Repair the Candidate's request schedule.
    ##Checks if there are any violations of the maximum AVAILABLE number of tools due to peaks in requests.
    ##If that is the case, it tries to move the requests around, such that there are no longer any violations
    ##of the maximum available number.

    ##If this is possible or there haven't been any violations in the first place, the valid field of the
    ##Candidate is set to True.
    ##Otherwise, the valid field of the Candidate is set to False.
    ##:return: nothing
    ##"""
    def repair(self):        
        usages = self.get_tool_usages()
        extended_day_list = self.get_extended_daylist()

        random_number_which_used = compute_random_number(random.randint(0, 1000000))

        for (tool_id, usages_per_day) in usages.items():
            indexed_usages_per_day = enumerate(usages_per_day)
            (day_idx, peak_amount) = max(indexed_usages_per_day, key=lambda x: x[1]['min'])
            tool_availability = problem_object["tools"][tool_id].num_available

            # if the peak does not exceed the availability of the tool, we can ignore it :)
            if peak_amount['min'] <= tool_availability:
                continue

            # possibly involved requests:
            # all requests on the peak day
            possibly_involved_requests = extended_day_list[day_idx]
            involved_requests = []
            start_day_dict = {}

            # calculate the involved requests:
            # those, which are DELIVER or RUNNING
            # and actually request the wanted tool
            for req_id in possibly_involved_requests.keys():
                req_state = possibly_involved_requests[req_id]

                if req_state == "deliver" or req_state == "running":
                    req = problem_object["requests"][req_id]

                    if req.tool_id == tool_id:
                        involved_requests.append(req)

            # calculate the possible start positions of the requests...
            for request in involved_requests:
                first_start_day = request.first_day
                last_start_day = request.last_day
                start_day_dict[request.id] = list(range(first_start_day, last_start_day + 1))

            max_depth = 6
            while True:
                #print("max_depth:", max_depth)
                repair_result = self.rec_repair(start_day_dict, extended_day_list, 0, max_depth)

                if repair_result is None:
                    if max_depth < 15:
                        max_depth += 3
                        continue
                    else:
                        self.valid = False
                        return
                else:
                    self.valid = True
                    # to create a "normal" day-list from the extended day-list,
                    # we only need to delete the "running" entries from the latter

                    new_day_list = []
                    for requests_per_day in repair_result:
                        new_day_list.append({req_id: state for req_id, state in requests_per_day.items() if state != "running"})

                    # make the result stick
                    self.day_list = new_day_list
                    break
        #The recursive part of the repair.

        #This function keeps track of the recursion depth used and will return None, if the maximum depth has been
        #exceeded.
        #param move_dict: The dictionary containing the not-yet-tried moves per request
        #param current_extended_daylist: The extended day-list on which to base the movements
        #param depth: The current depth of the iteration
        #param max_depth: The maximum depth of the iteration
        #return: The extended day-list representing a fix, or None if no such day-list could be found
    

    def rec_repair(self, move_dict, current_extended_daylist, depth, max_depth):


        chosen_request = None
        max_impact = 0
        next_move_dict = move_dict.copy()
        tool_id = None
        tool_availability = None

        for req_id in move_dict:
            # if the list of possible start days is not empty...
            if move_dict[req_id]:
                tmp_request = problem_object["requests"][req_id]
                if tmp_request.num_tools > max_impact:
                    max_impact = tmp_request.num_tools
                    chosen_request = tmp_request
                    tool_id = chosen_request.tool_id
                    tool_availability = problem_object["tools"][tool_id].num_available

        if chosen_request is None:
            # at this point, we have exhausted all possible moves
            return None

        # we don't want to move the same request twice in recursive calls
        next_move_dict[chosen_request.id] = []

        # look at all the possible positions for the chosen request
        for position in move_dict[chosen_request.id]:
            tmp_ext_daylist = current_extended_daylist.copy()
            start_day = position
            end_day = position + chosen_request.num_days

            # update the extended day list:
            # remove all old occurrences of the chosen request
            # and set up the new ones
            for (day_idx, req_dict) in enumerate(tmp_ext_daylist):
                req_dict.pop(chosen_request.id, None)

                if start_day == day_idx:
                    tmp_ext_daylist[day_idx][chosen_request.id] = "deliver"
                elif end_day == day_idx:
                    tmp_ext_daylist[day_idx][chosen_request.id] = "fetch"
                elif start_day < day_idx < end_day:
                    tmp_ext_daylist[day_idx][chosen_request.id] = "running"

            # get usages for our tool and the new daylist
            tmp_usages = tool_usages_from_extended_daylist(tmp_ext_daylist)[tool_id]
            new_peak = max(tmp_usages)

            # if this move fixed the peak, let's return the extended day list that fixed the problem
            if new_peak <= tool_availability:
                return tmp_ext_daylist

            # if we haven't repaired the problem at this stage, let's try to go deeper
            if depth < max_depth:
                deeper_result = self.rec_repair(next_move_dict, tmp_ext_daylist, depth + 1, max_depth)
                if deeper_result is not None:
                    return deeper_result

        # at this point, we have exhausted all possibilities and still not found a solution
        return None


    # this function is used to transslate 
    #:parameter  value :
    #:parameter left_min:
    #:parameter  left_max:
    #:parameter  right_min:
    #:parameter right_max:
    #:return:


def translate(value, left_min, left_max, right_min, right_max,):
   
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # if left_min == left_max, we can pick an arbitrary value for
    # what we return
    left_span = float(left_span)
    if left_span == 0:
        return right_max

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / left_span

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)





#check if the route is valid.
#Check the supplied route against several limits, such as maximum driving distance and vehicle capacity.
#It is assumed that the first and last StopOver in the route are the depot.
#The first and last StopOver's num_tools fields may be altered in order to reflect if the route
#requires some tools to be picked up from the depot at the beginning or some tools are dropped off
#at the depot at the end of the trip.
#param route: The route (i.e. list of StopOvers) to check
#return: True if the route is valid, False otherwise


def is_route_valid(route, tool_id):
    
    sum_distance = 0
    loaded = 0
    max_load = 0
    tool_size = problem_object["tools"][tool_id].size

    for (idx, stopover) in enumerate(route):
        customer_id   = stopover.customer_id
        request_id    = stopover.request_id
        change_amount = stopover.num_tools

        if idx == 0:
            continue

        last_customer_id = route[idx-1].customer_id
        sum_distance += problem_object['distance'][last_customer_id][customer_id]

        # change amount is subtracted because we use negative numbers to
        # indicate FETCH requests (which is where tools get loaded)
        # and positive numbers to indicate DELIVER requests (which is where
        # we lower the amount of loaded tools)
        loaded -= change_amount

        if (loaded * tool_size) > problem_object["capacity"]:
            return False
        elif sum_distance > problem_object["max_trip_distance"]:
            return False

        max_load = max(max_load, loaded)

    if loaded < 0:
        depot_load = loaded

        # if we need to fetch something at the depot already,
        # we have to take into account that we are
        if ((max_load + abs(depot_load)) * tool_size) > problem_object["capacity"]:
            return False

        route[0].num_tools = depot_load

    elif loaded > 0:
        route[-1].num_tools = loaded

    return True




    #Calculate the tool usages from the given extended day list.
    #The result will be a dictionary with Tool IDs as keys and a list as value.
    #The value list has one entry per day of the problem and the entry's value equals to the amount of tools
    #required at this day as per optimistic maximum
    #(i.e. tools still at a customer's place + tools requested - tools returned)
    #param ext_day_list: The extended day-list to calculate the daily uses from
    #return: The usage dictionary in form of {TOOL_ID: [LIST_OF_USES_PER_DAY]}

def tool_usages_from_extended_daylist(ext_day_list):

    usage = {}
    for tool_id in problem_object["tools"]:
        usage[tool_id] = [0 for day in ext_day_list]

    for (day_idx, req_dict) in enumerate(ext_day_list):
        for (req_id, req_state) in req_dict.items():
            request = problem_object["requests"][req_id]
            tool_id = request.tool_id
            delivery_amount = 0
            running_amount = 0
            fetch_amount = 0

            if req_state == "deliver":
                delivery_amount = request.num_tools
            elif req_state == "running":
                running_amount = request.num_tools
            elif req_state == "fetch":
                fetch_amount = request.num_tools

            change_amount = (delivery_amount + running_amount) - fetch_amount
            usage[tool_id][day_idx] += change_amount

    return usage






##Create the initial population for the genetic algorithm
##return: a list of candidates
## It creates an initial population of a particular specified size which is specified by the 

def initial_population(population_size):
    # for each request, pick a random starting day and the corresponding end day
    population = []
    i = 0
    total_days=problem_object['days']

    random_number_which_used = compute_random_number(random.randint(0, 1000000))
    ## It is looped till the population size. In our case, the population size is (25)
    while i < population_size:
        ## day_list is a list of dictionaries
        ## Each element of this list will be a dictionary
        ## day_list[1] - It will be a dict
        ## day_list[1][2] ---- [2] is the key for dict day_list[1]
        ## dict d
        ## d['Azam'] --- Value of the key
        ## dict day_list[1]
        ## day_list[1][2] -- VAlue of the key 2 for dict day_list[1]
        ## request_id is the key && day_list[start_day] is a dictionary
        day_list = [{} for _ in range(total_days)]

        random_number_which_used = compute_random_number(random.randint(0, 1000000))

        ## Looping over all the requests ( because we have to satisfy/fulfill all of them)
        ## Satisfying a request means to provide all the tools to the customer which are needed in the appropriate quantity - which isprovided in the request id
        for key, request in problem_object['requests'].items():
            ##  Populating the start day of fulfiling the requests in a way that it satisfies the range
            ## start day means the day when the vehicle will deliver the tools for the request if to the relevant customer
            start_day = random.randrange(request.first_day,
                                         request.last_day + 1)  # randrange excludes the stop point of the range.
            
            
            ## On the start day, for this request id  -  We are delivering the tools
            day_list[start_day][request.id] = 'deliver'

            ## On the start day + request.num_days, for this request id  -  We are fetching the tools
            day_list[start_day + request.num_days][request.id] = 'fetch'
        
        ## We make a candidate solution here, It is allocated in a random fashion by the genetic algorithm
        candidate = Candidate(day_list)

        ## This function is a important function and it checks whether the candidate is "VALID" or "INVALID"
        ## All the other constraint checks are done here. If all constraints are satisfies, then it is a VALID candidate
        ## Otherwise, it is invalid
        candidate.repair()
        
        ## If the candidate is not VALID, then we need to find some other candidate
        if not candidate.valid:  # we need to create an additional candidate
            continue

        
        ## It is in a way the cost required to solve the problem. IT takes into account all the different coses - Vehicle Cost + Distance Cost + Tool Cost
        x = candidate.fitness_heuristic()
        
        if x ==-1:
            continue
        max_cars_used = x[0]
        problem_vehicle_cost_ =x[1]
        sum_cars_ = x[2]
        problem_vehicle_day_cost_ = x[3]
        sum_distance_ = x[4]
        problem_distance_cost_ = x[5]
        sum_tool_costs_ = x[6]
        fit_ = x[7]
        if fit_ == -1:
            continue # candidate is not valid

        candidate.fit = fit_
        candidate.max_cars = max_cars_used
        candidate.problem_vehicle_cost = problem_vehicle_cost_
        candidate.sum_cars = sum_cars_
        candidate.problem_vehicle_day_cost = problem_vehicle_day_cost_ 
        candidate.sum_distance = sum_distance_
        candidate.problem_distance_cost = problem_distance_cost_
        candidate.sum_tool_costs = sum_tool_costs_
        ## Add the candidate which was valid to the population
        population.append(candidate)
        i += 1
        # print("Found the {}. candidate!".format(i))

    ## Returns the population
    return population
   
   
   
   
   
   #Let Candidates a and b create a child element, inheriting some characteristics of each

    #param a:
    #param b:
    #return:
def combine(a, b):
    ## This is the solution format.
    ## All the candidates will have this format only
    new_day_list = [{} for _ in range(problem_object['days'])]


    ## For each request( (charateristic)- because it will be fulifiled on a particular start day ) , we are
    # generate a randomnumber. Depending upon the value of the random number, we fulfil the request either like parent a, or like parent b 

    ## e.g Request 1,2,3
    ## Cand 1 : Solving req 1 on day 2                 Cand 2: Solving req 1 on day 1
     #         solving req 2 on day 1                           Solving req 2 on day 3
     #         solving req 3 on day 4                           Solving req 3 on day 5

     ## Child/Crossover Candiate ( New Canditate):  Solving req 1 - day 2 
                                                 #  Solving req 2 - day 3
                                                #   Solving req 3 - day 5

    for (request_id, request) in problem_object['requests'].items():
        ## We generate a random number
        r = random.random()

        if r < 0.5:  # use the startday and endday from candidate a
            chosen_candidate = a

        else:  # use the startday and endday from candidate b
            chosen_candidate = b

        for day_idx in range(request.first_day, request.last_day + 1):
            if request_id in chosen_candidate.day_list[day_idx]:
                new_day_list[day_idx]                   [request_id] = 'deliver'
                new_day_list[day_idx + request.num_days][request_id] = 'fetch'
                break

    new_candidate = Candidate(new_day_list)
    new_candidate.repair()  # Check this candidate
    
    x = new_candidate.fitness_heuristic()

    if(x==-1):
        new_candidate.fit = -1
        return new_candidate
    
    max_cars_used = x[0]
    problem_vehicle_cost_ =x[1]
    sum_cars_ = x[2]
    problem_vehicle_day_cost_ = x[3]
    sum_distance_ = x[4]
    problem_distance_cost_ = x[5]
    sum_tool_costs_ = x[6]
    fit_ = x[7]

    new_candidate.fit = fit_
    new_candidate.max_cars = max_cars_used
    new_candidate.problem_vehicle_cost = problem_vehicle_cost_
    new_candidate.sum_cars = sum_cars_
    new_candidate.problem_vehicle_day_cost = problem_vehicle_day_cost_ 
    new_candidate.sum_distance = sum_distance_
    new_candidate.problem_distance_cost = problem_distance_cost_
    new_candidate.sum_tool_costs = sum_tool_costs_

    return new_candidate
    
    


    #From the values list, find a pair which is not in the blocked_list.

    #param values:
    #param scale:
    #param blocked_values:
    #return: A tuple of Candidates


def find_mating_pair(values, scale, blocked_values=None):


    if len(values) < 2:
        raise Exception('Population too small')

    if blocked_values is None:
        blocked_values = []

    val_0 = get_random_candidate(values, scale)
    val_1 = None

    while (val_1 is None) or (val_0 == val_1) \
            or ((val_0, val_1) in blocked_values) or ((val_1, val_0) in blocked_values):
        val_1 = get_random_candidate(values, scale)

    return (val_0[2], val_1[2])



    #Fetch a random candidate from the supplied values list.

    #Generate a random value R in the range [0, SCALE) and fetch that element from the values list
    #where LOWER <= R < UPPER.

    #param values: A list containing triples of the form (LOWER, UPPER, ELEMENT)
    #param scale: The sum of all fitness values
    #return: An element from the values list

def randommnumber():
    x = random.random()
    return x

def get_random_candidate(values, scale):
    ## We generate a random number, multiply it with a scale
    print('We would be generating a random number first')
    
    r = randommnumber() * scale

    ## We loop over the values, till we find one which satisfies the range criteria
    for v in values:
        # v[0] -> LOWER,v[1] -> UPPER
        if v[0] <= r < v[1]:
            return v


def solve(problem,ttl):
    ## Problem Object is a global object as it is used at multiple places 
    global problem_object
    problem_object = problem

    ## The time at which the algorithm runs
    start_time= datetime.datetime.now()
    print('Starting now: ' + start_time.isoformat())

    random_number_which_used = compute_random_number(random.randint(0, 1000000))
    ## First STEP of the GENETIC ALGORITHM - in which we populate the candiates 
    # create initial population
    ## we are creating an initial population of size 25! ( FEATURES['population_size'] = 25 in our case )
    population = initial_population(25)

    random_number_which_used=compute_random_number_2(0,10000)
    ## Sort the population based on the fitness value
    ## lamda p: p.fit - Sort it on the basis of fit.
    population = sorted(population, key=lambda p: p.fit)

    ## Print the length of the population
    print("population size:", len(population))
    total_population = 1000


    ## We are looping over all the Candidates of the population
    for i in range(0,total_population):
        ## This is used to get the time
        ## This is the start time
        problem_start_time=problem['starttime']

        random_number_which_used=compute_random_number_2(0,10000)
        
        ## This is the current time
        timenow=datetime.datetime.now()
        
        ## If the difference in time ( time taken by our genetic algorithm is greater than (Total_time - 3)seconds, we stop).
        ## This is because we would require 3 seconds for printing the best potential solution to the output file
        if (timenow - problem_start_time).seconds > (ttl-3):
            continue;
        
        ## Sort the population based on the fitness values of the candidates
        population_sorted = sorted(population, key=lambda p: p.fit)

        ## This is the value of the lowest fitness
        ## Because it is being sorted on the fitness 
        lowest_fitness  = population_sorted[0].fit
        

        ## This is the highest fitness value
        highest_fitness = population_sorted[-1:][0].fit
        

        ## Depending upon the values of the fitness, we define the fitness range
        fitness_range = []
        upper = 0
        lower=0

        for elem in population_sorted:
            # we want that elems with a smaller fitness have a higher chance to be chosen for combination!
            inverted_fitness = lowest_fitness + (highest_fitness - elem.fit)
            lower = upper
            upper = lower + inverted_fitness
            fitness_range.append((lower, upper, elem))

    
        
        sum_fitness_values = fitness_range[-1][1] + 1 # the upper bound of the last entry.

        # create new population through crossover

        blocked_values = []
        new_population = []
        num_new_candidates = 0
        
        population_size = 25

        ## Survivors are the candidates of parent generation.
        ## We will only save the survivor_size number of candidates of the parent generation 
        survivor_size  = 7       
        
        
        ## The total population will always be equal to the initial population size even after crossovers
        ## For doing that, we would preserve survivor_size number of candidates of the parent generation, and the rest of 
        # the candidates will be the new candidates that we get after crossovers
        ## population_size - survivor_size is the 
        while num_new_candidates < population_size - survivor_size:

            # select crossover candidates (candidates with higher fitness have a higher chance to get reproduced)
            values=fitness_range

            blocked_values=blocked_values


            ## THis tells us about the range of the fitness values
            scale=sum_fitness_values

            if blocked_values is None:
                blocked_values = []

            ## For getting the first candidate to perform crossover, we select a random Candidate from our current Population.
            val_0 = get_random_candidate(values, scale)

            val_1 = None


            while (val_1 is None) or (val_0 == val_1) \
                    or ((val_0, val_1) in blocked_values) or ((val_1, val_0) in blocked_values):
                val_1 = get_random_candidate(values, scale)

            ## These are the two Candidates that we take for CROSSOVERS

            ## Candidate1 is one which will be used for CROSSOVER
            one=val_0[2] 

            ## Candidate2 is one which will be used for CROSSOVER
            two=val_1[2]

            ## New candidate is formed by applying cross over technique using the two candidates
            new_candidate = combine(one, two)

            ## We have to check that this solution satisfies all constraints of the problem
            ## If it satisfies them, then it is VALID, otherwise it is INVALID

            if not new_candidate.valid:  # we need to generate an additional candidate
                continue

            # mutate (happens randomly)
            ## As 
            new_candidate.mutate()

            if new_candidate in population:  # we need to generate an additional candidate
                # print('Already in population')
                continue

            ## If the new candidate is not in the population, then we add it to the new_population
            new_population.append(new_candidate)

            ## We add the Candidate 1 and Candidate 2 in the blocked values, because we don't want them to be selected again.
            ## The new  candidates formed after crossover have characteristics similar to the two parents. Hence, we block the candidates 
            ## We immediately block the candidates that are used to form children, so that there is diversity in the characteristics of the new candidates
            blocked_values.append((one, two))

            ## we are incremeneting the num of new candidates
            num_new_candidates += 1

        # select survivors (the best ones survive => the ones with the lowest fitness)
        population = sorted(population, key=lambda p: p.fit)

        ## We select/save only the first survivor size (7) candidates from our original/parent population  
        ## We are not considering the remaining 18 candidates of our parent generation
        population = population[:survivor_size]

        ## We are addinn the old population of 7 remaining candidates with the new population of 18 new candidates
        new_population.extend(population)

        ## This new population will again have a size of 25
        ## Now our population will be the 18 new candidates and old population of 7 remaining candidates
        population = new_population

        population = sorted(population, key=lambda p: p.fit)
        # print('Best  fitness: ', population[0].fit)
        population[0].fitness_heuristic()
        # print('Worst fitness: ', population[-1] .fit)

    end_time = datetime.datetime.now()
    print('Took me: ' + str((end_time - start_time).seconds) + 's')
    
    ## When the time is reached, out of the current population of 25 Candidate solutions, we return the best one
    # The best one will have the minimum fit value (The minimum cost) 
    # return the best solution
    ## Because we are always sorting the population of 25 candidate solutions, hence the best solution will be at the 1st element ( 0 index )
    
    
    
    return population[0]  


def main(argv=None):
    ## Store all the arguments that we enter in the command line as a list
    global command_line_arguments
    command_line_arguments = sys.argv

    ## Parse te command Line arguments stored and populate the variables
    input_file_name = command_line_arguments[1]
    ttl= int(command_line_arguments[4])


    ## Having a population size of 25.
    ## Population size has to be specified by the user. The more the population size is, the better the probability of finding a good/optimal solution
    ## But, having a large population size also requires a lot of time - Because we have to populate the candidates and then apply cross-overs etc
    population_size_genetic_algoriithm = 25
    
    cur_seed = int(command_line_arguments[3])

    random.seed(cur_seed)
    output_file_name = command_line_arguments[2]

    # read the file, remove the blank lines ( lines.strip() )
    with open(input_file_name, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]


    ## lines variables stores all the non empty lines in the file
    ## These different variables are used for storing the values and then we filter on the lines so that we parse the 
    
    
    ## 'problem' is a dictionary that stores all relevant information about our problem
    problem = {}


    problem['tools'] = {}
    problem['customers'] = {}
    problem['requests'] = {}

    state = 'none'
    dataset='DATASET'
    DAYS='DAYS'
    CAPACITY='CAPACITY'
    MAX_TRIP_DISTANCE='MAX_TRIP_DISTANCE'
    DEPOT_COORDINATE='DEPOT_COORDINATE'
    VEHICLE_COST='VEHICLE_COST'
    VEHICLE_DAY_COST='VEHICLE_DAY_COST'
    DISTANCE_COST='DISTANCE_COST'
    TOOLS='TOOLS'
    COORDINATES="COORDINATES"
    REQUESTS="REQUESTS"
    DISTANCE="DISTANCE"

    for i in range(0,1000):
        i = 1234
    
    for j in range(0,1000):
        j=random.randint(0,1000)

    # decide what information to save for each line
    for line in lines:
        ## line.split('=', 1)[1].strip() - This assigns the value present at the RHS side of the equation 
        ## e.g DAYS = 10 , we want to capture/store the '10' and we get it using the split('=') function
        ## We check whether a line starts with a particular word, and then depending on that we store its value in a variable
        if line.startswith(dataset):
            problem['dataset'] = line.split('=', 1)[1].strip()
        
        elif line.startswith(CAPACITY):
            problem['capacity'] = int(line.split('=', 1)[1].strip())

        elif line.startswith(DEPOT_COORDINATE):
            problem['depot_coordinate'] = int(line.split('=', 1)[1].strip())

        elif line.startswith(DISTANCE_COST):
            problem['distance_cost'] = int(line.split('=', 1)[1].strip())

        elif line.startswith('NAME'):
            problem['name'] = line.split('=', 1)[1].strip()

        elif line.startswith(DAYS):
            problem['days'] = int(line.split('=', 1)[1].strip())

       
        elif line.startswith(MAX_TRIP_DISTANCE):
            problem['max_trip_distance'] = int(line.split('=', 1)[1].strip())

        elif line.startswith(VEHICLE_DAY_COST):
            problem['vehicle_day_cost'] = int(line.split('=', 1)[1].strip())        

        elif line.startswith(VEHICLE_COST):
            problem['vehicle_cost'] = int(line.split('=', 1)[1].strip())

        ## state has been used here for our convenience so that we read and parse the input correctly
        elif line.startswith(REQUESTS):
            state = 'requests'

        elif line.startswith(TOOLS):
            state = 'tools'

        
        elif line.startswith(DISTANCE):
            state = 'distance'

        elif line.startswith(COORDINATES):
            state = 'customers'  # coordinates are renamed as customers for better understanding

        elif line.strip() != '':
            if state == 'tools':
                tool = Tool.create_from_line(line)
                problem['tools'].update({tool.id:tool})  # dictionary entry; key = tool.id, value = tool

            elif state == 'requests':
                request = Request.create_from_line(line)
                problem['requests'].update({request.id:request})  # dictionary entry; key = request.id, value = request

            elif state == 'customers':
                customer = Customer.create_from_line(line)
                problem['customers'].update({customer.id:customer})


    random_number_which_used = compute_random_number(random.randint(0, 1000000))      
    ## ## Given the coordinates of all the customers, it calculates the distance matrix       
    create_distance_matrix(problem)
    
    ##It tells us about the time the program started
    problem['starttime']=datetime.datetime.now()

    ## solve() function is the main function. The genetic algorithm is applied here only
    best_solution = solve(problem,ttl)
    
    randommm_numberr = compute_random_number_2(0,10000)
    
    problem_instance=problem
    filename_input=input_file_name
    output_str = ""

    output_str += 'DATASET = {}\n'.format(problem_instance['dataset'])
    output_str += 'NAME = {}\n\n' .format(problem_instance['name'])

    max_num_vehicles = 0 
    for (day_idx, cars_day) in enumerate(best_solution.cars_on_day):
        output_str += 'DAY = {}\n'.format(day_idx + 1)
        #output_str += 'NUMBER_OF_VEHICLES = {}\n'.format(len(cars_day))
        for (car_idx, car) in enumerate(cars_day):
            output_str += '{}\tR\t'.format(car_idx + 1)
            max_num_vehicles=max(max_num_vehicles,len(cars_day))
            for (trip_idx, trip) in enumerate(car):
                for (stopover_idx, stopover) in enumerate(trip.stopovers):
                    if stopover_idx == 0 and trip_idx != 0:  # ignore the depot if this is not the first trip
                        continue
                    if stopover.num_tools < 0:
                        output_str += "-"
                    output_str += '{}\t'.format(stopover.request_id)

            output_str += "\n"
        output_str += "\n"
    
    print(output_str)
    print(max_num_vehicles)

    # inputfile = command_line_arguments[1].split('.\Instances\\')[1]
    # ix= inputfile.split('.txt')[0]+'.csv'
    # inputfile=ix
    # wr =  open(inputfile,'w') 

    # wr.write('Total Cost = ')
    # wr.write(str(best_solution.fit))
    # wr.write('\n')
    # wr.write('Max Number of Vehicles used = ')
    # wr.write(str(best_solution.max_cars))
    # wr.write('\n')
    # wr.write('Vehicle Cost = ')
    # wr.write(str(best_solution.problem_vehicle_cost))
    # wr.write('\n')
    # wr.write('Vehicle Day Cost = ')
    # wr.write(str(best_solution.problem_vehicle_day_cost))
    # wr.write('\n')
    # wr.write('Total Vehicle Days = ')
    # wr.write(str(best_solution.sum_cars))
    # wr.write('\n')
    # wr.write('Total Distance = ')
    # wr.write(str(best_solution.sum_distance))
    # wr.write('\n')
    # wr.write('Distance Cost = ')
    # wr.write(str(best_solution.problem_distance_cost))
    # wr.write('\n')
    # wr.write('Tool Cost = ')
    # wr.write(str(best_solution.sum_tool_costs))

    print('Total Cost = ')
    print(str(best_solution.fit))
    print('Max Number of Vehicles used = ')
    print(str(best_solution.max_cars))
    print('Vehicle Cost = ')
    print(str(best_solution.problem_vehicle_cost))
    print('Vehicle Day Cost = ')
    print(str(best_solution.problem_vehicle_day_cost))
    print('Total Vehicle Days = ')
    print(str(best_solution.sum_cars))
    print('Total Distance = ')
    print(str(best_solution.sum_distance))
    print('Distance Cost = ')
    print(str(best_solution.problem_distance_cost))
    print('Tool Cost = ')
    print(str(best_solution.sum_tool_costs))

    filename_output = output_file_name

    f = open(filename_output, 'w')
    f.write(output_str)
    f.close()



## Given the coordinates of two points, it calculates the euclidean distance b/w them
def distance_between_points(p1, p2):
    diff_x = math.fabs(p1.x - p2.x)
    diff_y = math.fabs(p1.y - p2.y)
    return math.floor(math.sqrt(math.pow(diff_x, 2) + math.pow(diff_y, 2)))


## Given the coordinates of all the customers, it calculates the distance matrix
def create_distance_matrix(problem):
    len_customers = range(len(problem['customers']))
    problem['distance'] = [[0 for _ in len_customers] for _ in len_customers]

    for k1, v1 in problem['customers'].items():
        for k2, v2 in problem['customers'].items():
            problem['distance'][k1][k2] = distance_between_points(v1, v2)



def compute_random_number(x):
    rand_num = 0 
    for i in range(0,100):
        rand_num = int(random.randint(0,x))
    return rand_num

def compute_random_number_2(x,y):
    rand_num = random.randint(x,y)
    for i in range(0,100):
        rand_num = int(random.randint(x,y))
    return rand_num


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
