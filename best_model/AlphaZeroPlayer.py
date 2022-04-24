import random
import numpy as np

from TicTacToe import TicTacToe
from NNet import NNet

# Note: 'state' is a string, 'action' is an integer
class AlphaMCTS:
    def __init__(self, game: TicTacToe):
        self.game = game
        # expected reward (q-value) for taking action a from state s
        self.Q = {}
        # the number of time we took action a from stat s across simulations
        self.N =  {}
        # the inital estimate of taking an action from state s according to the policy returned by the current neural network
        self.P =  {}
        # all visited moves
        self.visited = []
        # c = degree of exploration
        self.c = 0.1
    
    def best_action(self, num_sims: int, state: str, nnet: NNet):
        for _ in range(num_sims):
            self.search(state, self.game, nnet)
        pi = np.array(self.pi(state))
        return np.argmax(pi)

    def pi(self, state: str, temp=1.):
        #print("state: ", state)
        counts = self.N[state]
        #print("Counts: ", counts)
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum <= 0.0:
            return [float(x)  for x in counts]
        #print("sum: ", counts_sum)
        probs = [float(x) / counts_sum for x in counts]
        #print("probs: ", probs)
        return probs
    

    def add_unseen_states(self, state: str):
        moves = self.game.size*self.game.size
        if state not in self.Q:
            self.Q[state] = np.array([0. for _ in range(moves)])
        if state not in self.N:
            self.N[state] = np.array([0. for _ in range(moves)])


    def search(self, state: str, game: TicTacToe, nnet: NNet):
        self.add_unseen_states(state)

        if game.has_game_ended(state):
            return -game.get_reward(state)
        
        if state not in self.visited:
            self.visited.append(state)
            action, v = nnet.predict(np.array([game.vectorize_game_state(state)]))
            self.P[state] = action
            return -v

        max_u, best_a = -float('inf'), -1
        for a in game.get_legal_actions(state):
            # UCB score = Q + U
            # c = exploration constant
            u = self.Q[state][a] + self.c*self.P[state][a]*np.sqrt(sum(self.N[state])/ (1 + self.N[state][a]))
            if u > max_u:
                max_u = u
                best_a = a

        a = best_a
        sp = game.get_next_state(state, a)
        v = self.search(sp, game, nnet)

        self.Q[state][a] = (self.N[state][a] * self.Q[state][a] + v) / (self.N[state][a] + 1)
        self.N[state][a] += 1.
        return -v

def pick_random_player(game):
    players = {"*": 3}
    player = None
    if random.random() > 0.5:
        player = "O"
        players[player] = 1
        players[game.get_next_player(player)] = 2
    else:
        player = "X"
        players[player] = 1
        players[game.get_next_player(player)] = 2
    return players, player

def play_game(nnet1: NNet, nnet2: NNet, num_sims):
    game = TicTacToe()

    players, player = pick_random_player(game)

    state = game.get_start_state(player)
    nnet = nnet1

    while True:
        mcts = AlphaMCTS(game)

        action = mcts.best_action(num_sims, state, nnet)
        state = game.get_next_state(state, action)

        if game.has_game_ended(state):
            if game.is_winner(state[:-1], player):
                return players[player]
            elif game.is_winner(state[:-1], game.get_next_player(player)):
                return players[game.get_next_player(player)]
            else:
                return players["*"]
        
        player = game.get_next_player(player)
        if players[player] == 1:
            nnet = nnet1
        elif players[player] == 2:
            nnet = nnet2
        else:
            raise ValueError(f"Player {player} is not recognized.")    

def pit(nnet1: NNet, nnet2: NNet, num_games = 10, num_sims = 25):
    record = {1: 0, 2: 0, 3: 0}
    for _ in range(num_games):
        winner = play_game(nnet1, nnet2, num_sims)
        record[winner] += 1
    if record[1] >= record[2]:
        return 1
    else:
        return 2

def assignRewards(examples, r):
    for x in examples:
        x[-1] = r
        r = -r
    return examples

def executeEpisode(game: TicTacToe, player: str, nnet: NNet, num_sims = 25):
    examples = []
    state = game.get_start_state(player)
    mcts = AlphaMCTS(game)

    while True:
        for _ in range(num_sims):
            mcts.search(state, game, nnet)
        examples.append([state, mcts.pi(state), None])
        pi = mcts.pi(state)
        action = np.random.choice(len(pi), p = pi)
        state = game.get_next_state(state, action)
        if game.has_game_ended(state):
            examples.append([state, mcts.pi(state), None])
            final_state = state[:-1] + player
            examples = assignRewards(examples, game.get_reward(final_state))
            return examples

def collect_data(examples, game: TicTacToe):
    inputs = examples[:, -3]
    inputs = [game.vectorize_game_state(s) for s in inputs]
    inputs = np.array(inputs, dtype=np.float32)

    policies = list(examples[:, -2])
    policies = np.array(policies, dtype=np.float32)

    rewards = list(examples[:, -1])
    rewards = np.array(rewards, dtype=np.float32)
    rewards = np.reshape(rewards, (len(rewards), 1))
    
    return inputs, [policies, rewards]

def simulate(numIters = 5, numEps = 10, numSims = 25):
    nnet = NNet()
    game = TicTacToe()
    examples = []
    players = ["X", "O"]
    for i in range(numIters):
        for _ in range(numEps):
            examples += executeEpisode(game, players[round(random.random())], nnet, numSims)
        train_x, train_y = collect_data(np.array(examples), game)
        nnet2 = nnet.clone()
        nnet2.train(train_x, train_y)
        winner = pit(nnet, nnet2)
        if winner == 2:
            nnet = nnet2
        print(f"Completed Episode {i + 1}")

    return nnet

# Model took around 2 minutes to run
def run_and_save_model():
    nnet = simulate(10, 50, 25)
    nnet.model.save("test_10_25_25")

##############
# TEST CASES #
##############

def test_executeEpisode():
    nnet = NNet()
    game = TicTacToe()
    examples = executeEpisode(game, "X", nnet , 2)
    game.print_game_state(examples[-1][0])
    print()
    for e in examples:
        print(e)

