import random 
from Game import Game

class TicTacToe(Game):

    def __init__(self, board_size = 3, plx = "X", plo = "O", empty = "*"):
        self.size = board_size
        self.plx = plx
        self.plo = plo
        self.empty = empty

    def get_start_state(self, player: str):
        return "*********" + player
    
    def get_next_player(self, player: str):
        if player == self.plx:
            return self.plo
        elif player == self.plo:
            return self.plx
        else:
            raise ValueError(f"Player string {player} is not recognized")
    
    def is_winner(self, board: str, p: str):
        assert(len(board) == (self.size * self.size))

        return (board[0] == p and board[1] == p and board[2] == p) or \
               (board[3] == p and board[4] == p and board[5] == p) or \
               (board[6] == p and board[7] == p and board[8] == p) or \
               (board[0] == p and board[3] == p and board[6] == p) or \
               (board[1] == p and board[4] == p and board[7] == p) or \
               (board[2] == p and board[5] == p and board[8] == p) or \
               (board[0] == p and board[4] == p and board[8] == p) or \
               (board[2] == p and board[4] == p and board[6] == p)

    #########################################
    # Implementation of Game abstract class #
    #########################################

    def vectorize_game_state(self, state: str) -> list:
        state = list(state)
        for i in range(len(state)):
            if state[i] == "X":
                state[i] = 1
            elif state[i] == "O":
                state[i] = -1
            else:
                state[i] = 0
        return state

    def print_game_state(self, state: str) -> str:
        board = state[:-1]
        print("  " + " 1 " + " 2 " + " 3 ")
        for r in range(self.size):
            s = f"{r + 1} "
            for c in range(self.size):
                z = c + self.size * r
                s += f" {board[z]} "
            print(s)
        print("Current state: ", state)
    
    def get_legal_actions(self, state: str) -> list:
        board = state[:-1]
        actions = []
        for i in range(self.size * self.size):
            if board[i] == self.empty:
                actions.append(i)
        return actions
    
    def get_random_action(self, state: str) -> int:
        actions = self.get_legal_actions(state)
        return actions[random.randint(0, (len(actions) - 1))]
        

    def get_next_state(self, state: str, action: int) -> str:
        player = state[-1]
        if state[action] != self.empty:
            raise ValueError("Current Position Taken!")
        else:
            return state[:action] + player + state[(action + 1):-1] + self.get_next_player(player)

    
    def get_reward(self, state: str) -> int:
        player = state[-1]
        board = state[:-1]
        if self.is_winner(board, player):
            return 1
        elif self.is_winner(board, self.get_next_player(player)):
            return -1
        else:
            return 0


    def has_game_ended(self, state: str) -> bool:
        board = state[:-1]
        if self.is_winner(board, self.plx) or self.is_winner(board, self.plo):
            return True
        else:
            for a in board:
                if a == self.empty:
                    return False
            return True


def test():
    state = "O**OXXX**O"
    game = TicTacToe()
    game.print_game_state(state)
    print("Vectorized current state: ", game.vectorize_game_state(state))
    print("Legal actions: ", game.get_legal_actions(state))

    state = game.get_next_state(state, game.get_random_action(state))
    game.print_game_state(state)
    print("Legal actions: ", game.get_legal_actions(state))
    print()
    print("Has the game ended? ", game.has_game_ended(state))

    state = game.get_next_state(state, game.get_random_action(state))
    game.print_game_state(state)
    print("Legal actions: ", game.get_legal_actions(state))
    print()
    print("Has the game ended? ", game.has_game_ended(state))
    print("Reward: ", game.get_reward(state))
