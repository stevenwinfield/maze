import random

from collections import defaultdict, deque

from maze import DOWN, Goody, LEFT, PING, Position, RIGHT, STEP, UP, STAY, Baddy

REVERSE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
ALL_DIRS = {UP, DOWN, LEFT, RIGHT}
DIRECTION = {diff: direction for direction, diff in STEP.iteritems()}  # e.g. Position(0, 1) -> UP

class MazeData(object):
    ''' Stores information we incrementally discover about the maze we're in, maintains of some helpful stats,
        and allows this historical information to be interrogated
    '''
    def __init__(self):
        self.accessible = defaultdict(set)  # a map from position to set of positions that are accessible from there
        self.obstructions = set()  # a set of positions where obstructions are known
        self.unexplored = set()  # a set of positions that are known spaces, but which have not been visited
        self.max_x = self.max_y = self.min_x = self.min_y = 0

    def update(self, current_position, obstruction):
        ''' Update our internal state, given where we currently are and the nearby obstructions '''
        # If this position was previously marked as unexplored then unmark it
        self.unexplored.discard(current_position)

        # Check all directions from here
        for direction in ALL_DIRS:
            nearby_position = (current_position + STEP[direction])
            if obstruction[direction]:
                # Record a nearby obstruction
                self.obstructions.add(nearby_position)
            else:
                # Record a nearby space...
                self.accessible[current_position].add(nearby_position)
                # ...which might be unexplored...
                if nearby_position not in self.accessible:
                    self.unexplored.add(nearby_position)
                # ...but we definitely know we can get back here from there
                self.accessible[nearby_position].add(current_position)

            self.max_x = max(self.max_x, nearby_position.x)
            self.min_x = min(self.min_x, nearby_position.x)
            self.max_y = max(self.max_y, nearby_position.y)
            self.min_y = min(self.min_y, nearby_position.y)

    def minimum_width(self):
        ''' Return what we know to be the minimum width that this maze can be '''
        return self.max_x - self.min_x + 1

    def minimum_height(self):
        ''' Return what we know to be the minimum height that this maze can be '''
        return self.max_y - self.min_y + 1

    def has_been_explored(self, position):
        ''' Returns True if this position has been visited before, else False '''
        return position in self.accessible

    def unexplored_positions(self, position):
        ''' Return a set of positions that have not been explored from a given position, which
            must have been explored itself.
        '''
        if not self.has_been_explored(position):
            raise ValueError("{} has not yet been explored".format(position))
        return self.accessible[position] & self.unexplored

    def unexplored_directions(self, position):
        ''' Return a list of directions that have not been explored from a given position, which
            must have been explored itself.
        '''
        unexplored_positions = self.unexplored_positions(position)
        return [DIRECTION[unexplored_position - position] for unexplored_position in unexplored_positions]

    def compute_path(self, source, destination):
        ''' Compute a path between two positions - source and destination - using information already gathered
            about the maze.
            Both must be explored points.
            Returns a list of the directions in which steps must be taken.
        '''
        if not (self.has_been_explored(source) and self.has_been_explored(destination)):
            raise ValueError("Both source and destination must have been explored!")

        # Uses the A* search algorithm - https://en.wikipedia.org/wiki/A*_search_algorithm

        def heuristic_cost_estimate(position1, position2):
            return (position1 - position2).l1_norm()

        def reconstruct_path(came_from, destination):
            step_end = destination
            path = deque()
            while True:
                try:
                    step_start = came_from[step_end]
                    path.appendleft(DIRECTION[step_end - step_start])
                    step_end = step_start
                except KeyError:
                    break
            return path

        closed_set = set()  # The set of nodes already evaluated.

        # The set of currently discovered nodes still to be evaluated.
        # Initially, only the start node is known.
        open_set = {source}

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        came_from = {}

        # For each node, the cost of getting from the start node to that node.
        g_score = defaultdict(lambda: float("inf"))

        # The cost of going from start to start is zero.
        g_score[source] = 0

        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        f_score = defaultdict(lambda: float("inf"))
        # For the first node, that value is completely heuristic.
        f_score[source] = heuristic_cost_estimate(source, destination)

        while open_set:
            current = min(open_set, key=lambda pos: f_score[pos])  # Use a heapq?
            if current == destination:
                return reconstruct_path(came_from, destination)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.accessible[current]:
                if neighbor in closed_set:
                    continue  # Ignore a neighbor which has already been evaluated.
                # The distance from start to a neighbor
                tentative_g_score = g_score[current] + 1  # distance between neighbours is always 1
                if neighbor not in open_set:
                    open_set.add(neighbor)  # Discovered a new node
                elif tentative_g_score >= g_score[neighbor]:
                    continue  # This is not a better path.

                # This path is the best until now. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_cost_estimate(neighbor, destination)

        # We shouldn't get here, so let's throw rather than implicitly return None.
        raise RuntimeError("Failed to compute a path even though both endpoints have been explored!")


class Explorer(Goody):
    ''' A Goody who tries to quickly learn the maze's layout, biasing his travels towards the other Goody if possible.
        Once the Goodies' paths cross then we can plot a route back to the crossing point.
    '''

    # The states that this Goody can be in
    exploring, backtracking, pathfinding = range(3)

    def __init__(self):
        # Randomly choose a ping frequency - we shouldn't always be pinging at the same time as the other Goody
        self.ping_frequency = random.randint(10, 20)

        self.position = Position(0, 0)  # We define the origin as the place we start from

        # containers and variables for the things we've learned
        self.maze_data = MazeData()
        self.last_goody_position = None
        self.last_goody_position_stale = True
        self.last_baddy_position = None
        self.last_baddy_position_stale = True
        self.last_ping_response_turn_count = 0  # In units of turns
        self.turn_count = 0
        self.steps = []  # A history of steps, used for backtracking

        # When pathfinding, path is a deque of directions to move in, and path_destination is where we're heading
        self.path = None
        self.path_destination = None

        self.state = Explorer.exploring  # Set the initial state

        # A map from state to the method that will be used to choose what to do
        self._decision_maker = {Explorer.exploring:    self.explore,
                                Explorer.backtracking: self.backtrack,
                                Explorer.pathfinding:  self.pathfind}


    def take_turn(self, obstruction, ping_response):
        ''' Called by the Game. PING if required, otherwise dispatch to a decision-making method.
            Do some common maintenance too.
        '''
        self.turn_count += 1

        # Ping, if it's that time
        if self.turn_count - self.last_ping_response_turn_count == self.ping_frequency:
            return PING

        # Process the ping response into something easier to work with
        if ping_response is not None:
            self.process_ping_response(ping_response)

        # Update our knowledge of the maze with what we've just learned
        self.maze_data.update(self.position, obstruction)

        # If we're already at the last know position of the goody or baddy then we mark it as stale
        if self.position == self.last_goody_position:
            self.last_goody_position_stale = True
        if self.position == self.last_baddy_position:
            self.last_baddy_position_stale = True

        # Hand off the decision making to one of our methods
        chosen = self._decision_maker[self.state](obstruction)

        # Update our record of where we are
        if chosen not in (STAY, PING):
            self.position += STEP[chosen]

        # If we're not backtracking along our path then record the steps we take
        if not self.state == Explorer.backtracking:
            self.steps.append(chosen)

        return chosen

    def process_ping_response(self, ping_response):
        ''' Convert the positions of the other players into our co-ordinates, whose origin is our initial position '''
        self.last_ping_response_turn_count = self.turn_count
        for player, relative_position in ping_response.iteritems():
            position = self.position + relative_position
            if isinstance(player, Goody):
                self.last_goody_position = position
                self.last_goody_position_stale = False
            else:
                self.last_baddy_position = position
                self.last_baddy_position_stale = False

    def change_state(self, new_state, obstruction, **kwargs):
        ''' Called when a state change is needed '''
        self.state = new_state
        chosen = self._decision_maker[self.state](obstruction, **kwargs)
        return chosen

    def explore(self, obstruction):
        ''' Follow unexplored directions, with a bias towards the other Goody if possible.
            If we get stuck then begin backtracking.
        '''

        # If the other Goody has been somewhere that we know how to get to then let's go back there
        # Unless that place is where we are now - because they're not here!
        if (self.position != self.last_goody_position and
            not self.last_goody_position_stale and
            self.maze_data.has_been_explored(self.last_goody_position)):
            return self.change_state(Explorer.pathfinding, obstruction)

        # Otherwise continue exploring
        choices = self.maze_data.unexplored_directions(self.position)

        if choices:
            # If there's only one way to go then choose it
            if len(choices) == 1:
                return choices[0]

            # If we have ping data then we can bias our search towards the other goody
            if self.last_goody_position is not None:

                good_choices = set()  # This will contain one or two directions, depending on where the other Goody is
                relative_position = self.last_goody_position - self.position
                if relative_position.x > 0:
                    good_choices.add(RIGHT)
                elif relative_position.x < 0:
                    good_choices.add(LEFT)
                if relative_position.y > 0:
                    good_choices.add(UP)
                elif relative_position.y < 0:
                    good_choices.add(DOWN)

                good_choices &= set(choices)  # Remove directions that we've already explored
                if good_choices:  # If there's anything left then make our choice from them
                    choices = list(good_choices)

            chosen = random.choice(choices)

            return chosen
        else:
            # Nothing is unexplored around here - start backtracking
            return self.change_state(Explorer.backtracking, obstruction, check_unexplored=False)


    def backtrack(self, obstruction, check_unexplored=True):
        ''' Walk back along the route we took to get here until we find a spot with some unexplored directions '''

        # If the other Goody has been somewhere that we know how to get to then let's go back there
        if (self.position != self.last_goody_position and
            not self.last_goody_position_stale and
            self.maze_data.has_been_explored(self.last_goody_position)):
            return self.change_state(Explorer.pathfinding, obstruction)

        if check_unexplored and self.maze_data.unexplored_directions(self.position):
            return self.change_state(Explorer.exploring, obstruction)

        if self.steps:
            chosen = REVERSE[self.steps.pop()]
            return chosen
        else:
            # So we've backtracked all the way to where we started.
            # If we haven't already, check for unexplored directions
            if not check_unexplored and self.maze_data.unexplored_directions(self.position):
                return self.change_state(Explorer.exploring, obstruction)

            # So we must have explored the whole maze, but we don't know where the other Goody is yet.
            # Time to send a ping
            return PING


    def pathfind(self, obstruction):
        # If the other Goody has moved somewhere we haven't explored yet then switch back to exploring
        if not self.maze_data.has_been_explored(self.last_goody_position):
            return self.change_state(Explorer.exploring, obstruction)

        # Compute path if none exists or if we get updated information about the other Goody
        if self.path is None or self.path_destination != self.last_goody_position:
            self.path_destination = self.last_goody_position
            self.path = self.maze_data.compute_path(self.position, self.last_goody_position)
        if not self.path:  # Empty - we're at the place we thought they'd be
            self.path = None
            self.path_destination = None
            # Switch back to exploring - maybe we'll bump into them around here
            return self.change_state(Explorer.exploring, obstruction)
        else:
            # Proceed towards the last known position of the other Goody
            return self.path.popleft()


class Ninja(Baddy):
    ''' A Baddy who tries to quickly learn the maze's layout, biasing his travels towards a nearby Goody if possible.
        Once the Goodies' paths crosses us then we can plot a route back to that point
    '''

    # The states that this Goody can be in
    exploring, backtracking, pathfinding = range(3)

    def __init__(self):
        self.position = Position(0, 0)  # We define the origin as the place we start from

        # containers and variables for the things we've learned
        self.maze_data = MazeData()
        self.last_goody_position = defaultdict(lambda: None)
        self.last_goody_position_stale = defaultdict(lambda: True)
        self.last_ping_response_turn_count = 0  # In units of turns
        self.turn_count = 0
        self.steps = []  # A history of steps, used for backtracking

        # When pathfinding, path is a deque of directions to move in, and path_destination is where we're heading
        self.path = None
        self.path_destination = None

        self.state = Ninja.exploring  # Set the initial state

        # A map from state to the method that will be used to choose what to do
        self._decision_maker = {Ninja.exploring:    self.explore,
                                Ninja.backtracking: self.backtrack,
                                Ninja.pathfinding:  self.pathfind}


    def take_turn(self, obstruction, ping_response):
        ''' Called by the Game. Dispatch to a decision-making method. Do some common maintenance too. '''
        self.turn_count += 1

        # Process the ping response into something easier to work with
        if ping_response is not None:
            self.process_ping_response(ping_response)

        # Update our knowledge of the maze with what we've just learned
        self.maze_data.update(self.position, obstruction)

        # If we're already at the last know position of one of the goodies then we mark it as stale
        if self.last_goody_position:
            for goody, position in self.last_goody_position.iteritems():
                if self.position == position:
                    self.last_goody_position_stale[goody] = True

        # Hand off the decision making to one of our methods
        chosen = self._decision_maker[self.state](obstruction)

        # Update our record of where we are
        if chosen != STAY:
            self.position += STEP[chosen]

        # If we're not backtracking along our path then record the steps we take
        if not self.state == Ninja.backtracking:
            self.steps.append(chosen)

        return chosen

    def process_ping_response(self, ping_response):
        ''' Convert the positions of the other players into our co-ordinates, whose origin is our initial position '''
        self.last_ping_response_turn_count = self.turn_count
        for player, relative_position in ping_response.iteritems():  # Assume we'll get the order each time
            position = self.position + relative_position
            self.last_goody_position[player] = position
            self.last_goody_position_stale[player] = False

    def change_state(self, new_state, obstruction, **kwargs):
        ''' Called when a state change is needed '''
        self.state = new_state
        chosen = self._decision_maker[self.state](obstruction, **kwargs)
        return chosen

    def nearby_goody(self):
        ''' Return the position of the goody that is nearest (along a straight line), if possible, else None '''
        if not self.last_goody_position:
            return None
        else:
            return min(self.last_goody_position.itervalues(), key=lambda x: (self.position - x).l1_norm())

    def explore(self, obstruction):
        ''' Follow unexplored directions, with a bias towards the other Goody if possible.
            If we get stuck then begin backtracking.
        '''

        # If a Goody has been somewhere that we know how to get to then let's go back there
        # Unless that place is where we are now - because they're not here!
        if self.last_goody_position:
            for player, position in self.last_goody_position.iteritems():
                if (self.position != position and
                    not self.last_goody_position_stale[player] and
                    self.maze_data.has_been_explored(position)):
                    return self.change_state(Explorer.pathfinding, obstruction, goto=position)

        # Otherwise continue exploring
        choices = self.maze_data.unexplored_directions(self.position)

        if choices:
            # If there's only one way to go then choose it
            if len(choices) == 1:
                return choices[0]

            # If we have ping data then we can bias our search towards a nearby goody
            nearby_goody_pos = self.nearby_goody()
            if nearby_goody_pos is not None:

                good_choices = set()  # This will contain one or two directions, depending on where the other Goody is
                relative_position = nearby_goody_pos - self.position
                if relative_position.x > 0:
                    good_choices.add(RIGHT)
                elif relative_position.x < 0:
                    good_choices.add(LEFT)
                if relative_position.y > 0:
                    good_choices.add(UP)
                elif relative_position.y < 0:
                    good_choices.add(DOWN)

                good_choices &= set(choices)  # Remove directions that we've already explored
                if good_choices:  # If there's anything left then make our choice from them
                    choices = list(good_choices)

            chosen = random.choice(choices)

            return chosen
        else:
            # Nothing is unexplored around here - start backtracking
            return self.change_state(Ninja.backtracking, obstruction, check_unexplored=False)


    def backtrack(self, obstruction, check_unexplored=True):
        ''' Walk back along the route we took to get here until we find a spot with some unexplored directions '''

        # If a Goody has been somewhere that we know how to get to then let's go back there
        # Unless that place is where we are now - because they're not here!
        if self.last_goody_position:
            for player, position in self.last_goody_position.iteritems():
                if (self.position != position and
                    not self.last_goody_position_stale[player] and
                    self.maze_data.has_been_explored(position)):
                    return self.change_state(Explorer.pathfinding, obstruction, goto=position)

        if check_unexplored and self.maze_data.unexplored_directions(self.position):
            return self.change_state(Ninja.exploring, obstruction)

        if self.steps:
            chosen = REVERSE[self.steps.pop()]
            return chosen
        else:
            # So we've backtracked all the way to where we started.
            # If we haven't already, check for unexplored directions
            if not check_unexplored and self.maze_data.unexplored_directions(self.position):
                return self.change_state(Ninja.exploring, obstruction)

            # So we must have explored the whole maze, but we don't know where the Goodies are yet.
            # The must be wandering aimlessly. Let's sit here and wait for them!
            return STAY


    def pathfind(self, obstruction, goto=None):
        # If the Goodies have moved somewhere we haven't explored yet then switch back to exploring
        if all(not self.maze_data.has_been_explored(position) for position in self.last_goody_position.itervalues()):
            return self.change_state(Ninja.exploring, obstruction)

        # Compute path if none exists or if we get updated information about the other Goody
        if goto is not None or self.path is None or self.path_destination not in self.last_goody_position.itervalues():
            if goto:  # Go where we're told to...
                self.path_destination = goto
            else:  # ... or seek out the nearest goody (according to L1 norm)
                self.path_destination = min((position for position in self.last_goody_position.itervalues()
                                             if self.maze_data.has_been_explored(position)),
                                             key=lambda x: (self.position - x).l1_norm())

            self.path = self.maze_data.compute_path(self.position, self.path_destination)

        if not self.path:  # Empty - we're at the place we thought they'd be
            self.path = None
            self.path_destination = None
            # Switch back to exploring - maybe we'll bump into them around here
            return self.change_state(Ninja.exploring, obstruction)
        else:
            # Proceed towards the last known position of the other Goody
            return self.path.popleft()
