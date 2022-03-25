ids = ["316508126", "316299098"]
from copy import deepcopy


def current_loc_client(state, client):
    return state["clients"][client]["location"]


def norm(vector):
    a = [x ** 2 for x in vector]
    b = sum(a)
    return b ** 0.5


def legal_move(state, client):
    # possible move for a man in the current state
    n = len(state['map'])  # n=#rows
    m = len(state['map'][0])  # m = #cols
    current_position = current_loc_client(state, client)
    possible_moves = [True, True, True, True, True]
    map = state["map"]
    if current_position[0] == 0:  # cant go up
        possible_moves[0] = False
    elif map[current_position[0] - 1][current_position[1]] == "I":
        possible_moves[0] = False
    if current_position[0] == n - 1:  # cant go down
        possible_moves[1] = False
    elif map[current_position[0] + 1][current_position[1]] == "I":
        possible_moves[1] = False
    if current_position[1] == 0:  # cant go left
        possible_moves[2] = False
    elif map[current_position[0]][current_position[1] - 1] == "I":
        possible_moves[2] = False
    if current_position[1] == m - 1:  # cant go right
        possible_moves[3] = False
    elif map[current_position[0]][current_position[1] + 1] == "I":
        possible_moves[3] = False
    return possible_moves


def movement(state):
    prob = {x: [] for x in state["clients"].keys()}
    for client in state["clients"].keys():
        tmp = list(state["clients"][client]["probabilities"])
        legal = legal_move(state, client)
        for i in range(len(tmp)):
            if not legal[i]:
                tmp[i] = 0
        norm_tmp = norm(tmp)
        for i in range(len(tmp)):
            tmp[i] = tmp[i] / norm_tmp
        prob[client] = tmp
    return prob


def expec(state, prob):
    next_move = {x: [] for x in state["clients"].keys()}
    for client in state["clients"].keys():
        # p = prob[client]
        p = prob
        loc = current_loc_client(state, client)
        # find max probabilty action
        max_prob = p[0]
        max_index = 0
        for index in range(len(p)):
            if p[index] > max_prob:
                max_prob = p[index]
                max_index = index
        if max_index == 0:
            next_move[client] = [loc[0] - 1, loc[1]]  # up
        elif max_index == 1:
            next_move[client] = [loc[0] + 1, loc[1]]  # down
        elif max_index == 2:
            next_move[client] = [loc[0], loc[1] - 1]  # left
        elif max_index == 3:
            next_move[client] = [loc[0], loc[1] + 1]  # right
        elif max_index == 4:
            next_move[client] = [loc[0], loc[1]]  # stay
    return next_move


def l_max(vec1, vec2):
    x_diff = abs(vec1[0] - vec2[0])
    y_diff = abs(vec1[1] - vec2[1])
    if x_diff > y_diff:
        return x_diff
    return y_diff


def drone_to_person(state, move):
    drones = state["drones"]
    dr_to_pr = {x: 0 for x in state["drones"].keys()}
    for drone in drones.keys():
        dis = {x: 0 for x in state["clients"].keys()}
        drone_loc = drones[drone]
        for client in dis.keys():
            dis[client] = l_max(drone_loc, move[client])
        dr_to_pr[drone] = min(dis, key=dis.get)
    return dr_to_pr


def drone_to_pac(state):
    drones = state["drones"]
    dr_to_pac = {x: 0 for x in state["drones"].keys()}
    for drone in drones.keys():
        dis = {x: 0 for x in state["packages"].keys()}
        drone_loc = drones[drone]
        for package in dis.keys():
            dis[package] = l_max(drone_loc, state["packages"][package])
        dr_to_pac[drone] = min(dis, key=dis.get)
    return dr_to_pac


def grid_for_pac(state, packages):
    m = len(state["map"])
    n = len(state["map"][0])
    grid = [[100 for col in range(n)] for row in range(m)]
    for pac in packages:
        pac_loc = list(state["packages"][pac])
        grid[pac_loc[0]][pac_loc[1]] = 0
        for row in range(m):
            for col in range(n):
                tmp = l_max(pac_loc, [row, col])
                if tmp < grid[row][col]:
                    grid[row][col] = tmp
    for row in range(m):
        for col in range(n):
            if state["map"][row][col] == "I":
                grid[row][col] = 100000
    return grid


def grid_for_client(state, client_loc):
    m = len(state["map"])
    n = len(state["map"][0])
    grid = [[100 for col in range(n)] for row in range(m)]
    # client_loc = state["clients"][client]["location"]
    grid[client_loc[0]][client_loc[1]] = 0
    for row in range(m):
        for col in range(n):
            tmp = l_max(client_loc, [row,col])
            if tmp < grid[row][col]:
                grid[row][col] = tmp
    for row in range(m):
        for col in range(n):
            if state["map"][row][col] == "I":
                grid[row][col] = 1000000
    return grid


def policy(location, grid):
    x = location[0]
    y = location[1]
    m = len(grid)
    n = len(grid[0])
    best = grid[x][y]
    loc = location
    for i in range(-1, 2):
        for j in range(-1, 2):
            if m > x + i >= 0 and n > y + j >= 0:
                if (x + i) != x or (y+j) != y:
                    if grid[x + i][y + j] < best:
                        best = grid[x + i][y + j]
                        loc = [x + i, y + j]
    return loc, best


def pac_to_client(state):
  pac_to_cl = {x: "free" for x in state["packages"].keys()}
  for i in state["clients"]:
    w = state["clients"][i]["packages"]
    for j in w:
      pac_to_cl[j] = i
  return pac_to_cl


def avg(lst):
    if len(lst) == 0:
        return 0
    else:
        return sum(lst) / len(lst)


def grid_for_pac_per_client(state,packages,client):
    m = len(state["map"])
    n = len(state["map"][0])
    grid = [[100 for col in range(n)] for row in range(m)]
    for pac in packages:
        if pac in state["clients"][client]["packages"]:
            pac_loc = list(state["packages"][pac])
            grid[pac_loc[0]][pac_loc[1]] = 0
            for row in range(m):
                for col in range(n):
                    tmp = l_max(pac_loc, [row, col])
                    if tmp < grid[row][col]:
                        grid[row][col] = tmp
    for row in range(m):
        for col in range(n):
            if state["map"][row][col] == "I":
                grid[row][col] = 100000
    return grid

def man_by_pack(state, pack):
    for man in state["clients"].keys():
        if pack in state["clients"][man]["packages"]:
            return man
    return False

def want_more(state, client):
    for pack in state["packages"]:
        if type(state["packages"][pack]) is tuple and pack in state["clients"][client]["packages"]:
            return True
    return False

def relevant_pack(state): #return packs on floor
    rel = []
    for pack in state["packages"].keys():
        if type(state["packages"][pack]) is tuple:
            rel.append(pack)
    return rel

class DroneAgent:
    def __init__(self, initial):
        self.grid_packages = grid_for_pac(initial, initial["packages"].keys())
        self.drones_current = {x: "free" for x in initial["drones"].keys()}
        self.pac_to_client = pac_to_client(initial)
        self.m = len(initial["map"])
        self.n = len(initial["map"][0])
        self.to_go = len(self.pac_to_client.keys())
        self.to_pick = [x for x in initial["packages"].keys()]
        self.packages_amount_at_begin = len(initial["packages"].keys())
        self.current_stored_in_drone = {d: 0 for d in initial["drones"].keys()}
        self.man_by_drone = {d:0 for d in initial["drones"].keys()}
        self.drones_pack_on = {x: [] for x in initial["drones"].keys()}

        self.init_grid_packages = grid_for_pac(initial, initial["packages"].keys())
        self.init_drones_current = {x: "free" for x in initial["drones"].keys()}
        self.init_pac_to_client = pac_to_client(initial)
        self.init_to_go = len(self.pac_to_client.keys())
        self.init_to_pick = [x for x in initial["packages"].keys()]
        self.init_man_by_drone = {d:0 for d in initial["drones"].keys()}
        self.init_current_stored_in_drone = {d: 0 for d in initial["drones"].keys()}
        self.init_drones_pack_on = {x: [] for x in initial["drones"].keys()}

        self.tester = []

        self.turn = 0
        self.suc = []
        self.total = initial["turns to go"]
        self.drone_num = len(initial["drones"])
        self.packages_num = len(initial["packages"])

        if self.m + self.n - self.drone_num + self.packages_num < 35:
            self.limit = 35
        else:
            self.limit = self.m + self.n - self.drone_num + self.packages_num

    def act(self, state):
        if self.turn > self.limit:
            self.grid_packages = deepcopy(self.init_grid_packages)
            self.drones_current = deepcopy(self.init_drones_current)
            self.pac_to_client = deepcopy(self.init_pac_to_client)
            self.to_go = deepcopy(self.init_to_go)
            self.to_pick = deepcopy(self.init_to_pick)
            self.suc.append(self.turn)
            self.turn = 0
            self.man_by_drone = deepcopy(self.init_man_by_drone)
            self.current_stored_in_drone = deepcopy(self.init_current_stored_in_drone)
            return "reset"
        if self.to_go == 0:
            if self.packages_amount_at_begin > 1 and avg(self.suc) < state["turns to go"]:
                self.grid_packages = deepcopy(self.init_grid_packages)
                self.drones_current = deepcopy(self.init_drones_current)
                self.pac_to_client = deepcopy(self.init_pac_to_client)
                self.to_go = deepcopy(self.init_to_go)
                self.to_pick = deepcopy(self.init_to_pick)
                self.suc.append(self.turn)
                self.turn = 0
                self.man_by_drone = deepcopy(self.init_man_by_drone)
                self.current_stored_in_drone = deepcopy(self.init_current_stored_in_drone)
                return "reset"
            else:
                return "terminate"
        self.turn += 1
        drones = state["drones"].keys()
        action = []
        picked_this_turn = []

        for drone in drones:
            drone_loc = state["drones"][drone]
            if self.current_stored_in_drone[drone] == 0:  #GO LIFT ANYTHING
                action, picked_this_turn, is_pick, pack_lifted = self.action_free(state, drone, action, picked_this_turn)
                self.tester.append("A")
                self.tester.append(drone)
                if is_pick == 1:
                    self.current_stored_in_drone[drone] += 1
                    client = man_by_pack(state, picked_this_turn[-1])
                    self.drones_current[drone] = client
                    self.man_by_drone[drone] = client
                    self.drones_pack_on[drone].append(pack_lifted)
            else:
                client = self.man_by_drone[drone]
                if self.current_stored_in_drone[drone] == 1 and want_more(state, client): #GO LIFT ONE MORE THING
                    action, picked_this_turn, is_pick, pack_lifted = self.action_free(state, drone, action, picked_this_turn, client)
                    self.tester.append("B")
                    self.tester.append(drone)
                    if is_pick == 1:
                        self.current_stored_in_drone[drone] += 1
                        # self.drones_current[drone] = client
                        self.drones_pack_on[drone].append(pack_lifted)

                else: #GO TO DELIVER
                    self.tester.append("C")
                    self.tester.append(drone)
                    next_loc_pac, val_pac = policy(drone_loc, self.grid_packages)
                    mov = movement(state)
                    # print("***",self.drones_current[drone])
                    des = client
                    client_loc = expec(state, mov[des])
                    client_loc = client_loc[des]
                    next_loc, val = policy(drone_loc, grid_for_client(state, client_loc))
                    if state["drones"][drone][0] != state["clients"][des]["location"][0] or\
                                state["drones"][drone][1] != state["clients"][des]["location"][1]:
                        action.append(("move", drone, tuple(next_loc)))
                        continue
                    elif val == 0:
                        if state["drones"][drone][0] == state["clients"][des]["location"][0] and\
                                state["drones"][drone][1] == state["clients"][des]["location"][1]:
                            action.append(("deliver", drone, des, self.drones_pack_on[drone].pop()))
                            self.current_stored_in_drone[drone] -= 1
                            if self.current_stored_in_drone[drone] == 0:
                                self.man_by_drone[drone] = 0
                                self.drones_current[drone] = "free"
                            self.to_go -= 1
                        else:
                            action.append(("wait", drone))
                    else:
                        action.append(("wait", drone))
        # print(tuple(action))
        return tuple(action)

    def action_free(self, state, drone, action, picked_this_turn, client=False):
        is_pick = 0
        lifted_pack = "nothing"
        drone_loc = state["drones"][drone]
        if client:
            next_loc, val = policy(drone_loc, grid_for_pac_per_client(state, relevant_pack(state), client))
            self.tester.append("verify")
        else:
            next_loc, val = policy(drone_loc, self.grid_packages)
        if next_loc[0] != drone_loc[0] or next_loc[1] != drone_loc[1]:
            action.append(("move", drone, tuple(next_loc)))
            self.tester.append("123")
        elif val == 0:
            self.tester.append("456")
            for pack, loc in state["packages"].items():
                if loc[0] == next_loc[0] and loc[1] == next_loc[1] and pack not in picked_this_turn:
                    action.append(("pick up", drone, pack))
                    picked_this_turn.append(pack)
                    is_pick = 1
                    lifted_pack = pack
                    self.to_pick.remove(pack)
                    self.grid_packages = grid_for_pac(state, self.to_pick)
                    self.drones_current[drone] = pack
                    break
            if is_pick != 1:
                action.append(("wait", drone))
        else:
            action.append(("wait", drone))
            self.tester.append("789")
        return action, picked_this_turn, is_pick, lifted_pack

