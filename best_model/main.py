import tensorflow as tf
from tensorflow import keras

from TicTacToe import TicTacToe
from MCTSPlayer import MCTSNode
from NNet import NNet
from AlphaZeroPlayer import AlphaMCTS

player1 = "X"
player2 = "O"
num_sims = 100

a_num_sims = 25
model = keras.models.load_model("test_10_50_25")
nnet = NNet(model)


def get_move(game: TicTacToe, state: str, player: str):
    while True:
        try:
            if player == player1:
                alphamcts = AlphaMCTS(game)
                action = alphamcts.best_action(a_num_sims, state, nnet)
                print("AlphaZeroTicTacToe suggests the position: ", action)

                z = input(f"Enter Position (0-8) for player {player}: ")
        
                assert(0 <= int(z) <= 8)
                _ = game.get_next_state(state, int(z))
                return int(z)
            elif player == player2:
                print(f"Action Taken by MCTS")

                mcts = MCTSNode(game, state, player)
                z = mcts.best_action(num_sims)

                assert(isinstance(z, int))
                assert(0 <= z <= 8)
                _ = game.get_next_state(state, int(z))
                return z
            else:
                raise ValueError(f"Player string {player} is not recognized")
        except AssertionError:
            print(f"Oops, board position ({z}) out of range!")
        except ValueError:
            print(f"Invalid Input: state = {state}, action = {z}")

def main():
    # Set up game and inital position
    game = TicTacToe()
    player = player2
    state = game.get_start_state(player)

    print("Starting Board Position")
    game.print_game_state(state)
    print()

    while True:
        action = get_move(game, state, player)
        state = game.get_next_state(state, action)
        player = state[-1]
        game.print_game_state(state)
        print()

        if game.has_game_ended(state):
            reward = game.get_reward(state)
            winner = "Tie"
            if reward == 1:
                winner = state[-1]
            elif reward == -1:
                winner = game.get_next_player(state[-1])
            else:
                print("It's a tie!")
                break
            print(f"Player {winner} Wins!")
            break


if __name__ == "__main__":
    main()

# Info for Github
# Game state saved as a str with format: (0-8) positions + current players move
# NNet structure
# How to train a model
# How to load and play a model