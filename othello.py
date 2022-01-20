import os
import neat
import random
from itertools import combinations

# constants
WHITE = 1
BLACK = -1
START_BOARD = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, -1, 0, 0, 0,
    0, 0, 0, -1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]
OCT_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
WIN_BONUS = 64
TOURNAMENT_FITNESS = 0


def possible_moves(_board, _turn):
    _pboard = _board.copy()
    valid_moves = []
    for i in range(len(_pboard)):
        # get current board pieces
        if _pboard[i] == _turn:
            pos = index_to_pos(i)
            # walk in each oct direction to find possible moves
            for d in OCT_DIRS:
                last = None
                x = pos[0] + d[0]
                y = pos[1] + d[1]
                while last in [None, -_turn] and 0 <= x <= 7 and 0 <= y <= 7:
                    index = pos_to_index((x, y))
                    x += d[0]
                    y += d[1]
                    if _pboard[index] == 0 and last == -_turn:
                        # update board with possible moves, and keep track of them separately in a list
                        _pboard[index] = _turn << 1
                        valid_moves.append(index)
                    last = _pboard[index]
    return _pboard, valid_moves


def perform_move(_board, _index, _turn):
    _board[_index] = _turn
    # walk in each oct direction to flank pieces
    _pos = index_to_pos(_index)
    for d in OCT_DIRS:
        move = 1
        x = _pos[0] + d[0]
        y = _pos[1] + d[1]
        to_change = []
        while move != 0 and 0 <= x <= 7 and 0 <= y <= 7:
            i = pos_to_index((int(x), int(y)))
            if move == 1:
                if _board[i] == 0:
                    move = 0
                elif _board[i] == _turn:
                    move = -1
            elif _board[i] == -_turn:
                to_change.append(i)
            elif _board[i] == _turn:
                move = 0
            x += d[0] * move
            y += d[1] * move

        for i in to_change:
            _board[i] = _turn


def index_to_pos(_index):
    return int(_index % 8), int(_index // 8)


def pos_to_index(_pos):
    return int(_pos[0] + _pos[1] * 8)


def print_board(_board, _turn=None, _moves=None):
    if _turn is not None:
        # Print turn
        if _turn == WHITE:
            print("🔸🟠 ORANGE TURN 🟠🔸")
        elif _turn == BLACK:
            print("🔹🔵 BLUE TURN 🔵🔹")

    # Print board itself
    letters = ["⬛️", "🟠", "🔸️️", "🔹", "🔵"]
    for row in range(8):
        print(' '.join([letters[i] for i in _board[row * 8:row * 8 + 8]]))

    if _moves is not None:
        # Print possible moves
        print([index_to_pos(i) for i in _moves])


def eval_genomes(genomes, config):
    # Set all fitnesses to 0
    for id, agent in genomes:
        agent.fitness = 0

    # Battle all agents against each other in the generation
    for agents in list(combinations(genomes, 2)):
        # Pick two agents at a time
        alice_id, alice = agents[0]
        a_net = neat.nn.FeedForwardNetwork.create(alice, config)
        bob_id, bob = agents[1]
        b_net = neat.nn.FeedForwardNetwork.create(bob, config)

        # Fight them against each other twice, letting each side go first
        game_1 = eval_game(a_net, b_net)
        game_2 = eval_game(b_net, a_net)

        # Sum their total fitness for all the games
        alice.fitness += game_1[0] + game_2[1]
        bob.fitness += game_1[1] + game_2[0]


def eval_game(g1_net, g2_net):
    # Black goes first
    board = START_BOARD.copy()
    turn = BLACK
    game = True
    no_moves_available = False

    # Game loop
    while game:
        pboard, moves = possible_moves(board, turn)
        if len(moves) > 0:
            # Parse turn information from neural network and make a move
            output = g1_net.activate(pboard) if turn == BLACK else g2_net.activate(pboard)
            output = [output[i]*(i in moves) for i in range(64)]

            # Get max of possible moves and choose that
            candidate_max = min(output)
            candidate_move = 0
            for j in range(64):
                if output[j] > candidate_max:
                    candidate_max = output[j]
                    candidate_move = j
            perform_move(board, candidate_move, turn)
            no_moves_available = False
        elif no_moves_available:
            # End game if neither side can play
            game = False
        else:
            no_moves_available = True
        turn = -turn

    # End of games, evaluate fitness
    return board.count(BLACK), board.count(WHITE)


def run():
    # neural network stuff
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes)

    print("Best fitness -> {}".format(winner))


if __name__ == '__main__':
    run()
