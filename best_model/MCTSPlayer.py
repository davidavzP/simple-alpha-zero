import numpy as np

from TicTacToe import TicTacToe

class MCTSNode:
    def __init__(self, game: TicTacToe, state: str, current_player: str, parent=None, parent_action=None):
        self.game = game
        self.state = state
        self.parent = parent
        self.current_player = current_player
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = {1: 0, -1: 0, 0: 0}
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        self._untried_actions = self.game.get_legal_actions(self.state)
        return self._untried_actions

    def q(self):
        return self._results[1] - self._results[-1]

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.game.get_next_state(self.state, action)
        child_node = MCTSNode(self.game, next_state, self.current_player, self, action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.game.has_game_ended(self.state)

    def rollout(self):
        state = self.state
        while not self.game.has_game_ended(state):
            action = self.game.get_random_action(state)
            state = self.game.get_next_state(state, action)

        final_state = state[:-1] + self.current_player
        reward = self.game.get_reward(final_state)
        return reward

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, sim_num = 25):
        for _ in range(sim_num):
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(0.0).parent_action


def test_rollout(child):
    # Rollout Implementation
    state = child.state
    while not child.game.has_game_ended(state):
        action = child.game.get_random_action(state)
        state = child.game.get_next_state(state, action)

    final_state = state[:-1] + child.current_player
    reward = child.game.get_reward(final_state)
    child.game.print_game_state(state)
    print(f"Board: {state}")
    print(f"Reward: {reward}")

def test_rollout_fn():
    root = MCTSNode(TicTacToe(), "*********X", "X")
    child = root.tree_policy()
    test_rollout(child)
    print()
    root = MCTSNode(TicTacToe(), "********XO", "X")
    child = root.tree_policy()
    test_rollout(child)
    


