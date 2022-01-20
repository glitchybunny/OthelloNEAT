import os
import neat
import random
import multiprocessing
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
random.seed()


def possible_moves(_board, _turn):
    _pboard = _board.copy()
    valid_moves = []
    for _i in range(len(_pboard)):
        # get current board pieces
        if _pboard[_i] == _turn:
            # walk in each oct direction to find possible moves
            for d in OCT_DIRS:
                walk = True
                can_play = False
                pos = index_to_pos(_i)
                pos[0] += d[0]
                pos[1] += d[1]
                while walk and 0 <= pos[0] <= 7 and 0 <= pos[1] <= 7:
                    _j = pos_to_index(pos)
                    if _pboard[_j] == -_turn:
                        can_play = True
                    elif _pboard[_j] == 0:
                        if can_play:
                            _pboard[_j] = 2 * _turn
                            valid_moves.append(_j)
                        walk = False
                    else:
                        walk = False
                    pos[0] += d[0]
                    pos[1] += d[1]

    return _pboard, valid_moves


def perform_move(_board, _index, _turn):
    if _board[_index] != 0:
        raise RuntimeError("Can't place a tile in an already occupied space")

    # place piece
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
    return [_index % 8, _index // 8]


def pos_to_index(_pos):
    return _pos[0] + _pos[1] * 8


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


def eval_genome(genome, config):
    # Generate network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Fight agent against RNG players
    fitness_sum = 0
    count = 50
    for _ in range(count // 2):
        # Make sure there's an equal number of games on each side
        game_1 = eval_game(net, "random")
        game_2 = eval_game("random", net)
        fitness_sum += game_1[0] + game_2[1]

    # Return fitness
    return fitness_sum


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def eval_game(p1, p2):
    # Setup game
    board, turn = game_setup(p1)

    # Play game
    a, b = game_loop(p1, p2, board, turn)

    # Determine scores
    if a == 0:
        return 0, 64
    elif b == 0:
        return 64, 0
    else:
        return a, b


def game_setup(p1):
    # Black always goes first
    board = START_BOARD.copy()
    turn = BLACK

    # But make sure first AI move is random to keep it on its toes
    if p1 not in ["random", "player"]:
        pboard, moves = possible_moves(board, turn)
        candidate_move = random.choice(moves)
        perform_move(board, candidate_move, turn)
        turn = WHITE

    return board, turn


def game_loop(p1, p2, board, turn):
    game = True
    no_moves_available = False

    # Do game loop
    while game:
        pboard, moves = possible_moves(board, turn)
        if len(moves) == 1:
            perform_move(board, moves[0], turn)
            no_moves_available = False
        elif len(moves) > 0:
            if (p1 == "random" and turn == BLACK) or (p2 == "random" and turn == WHITE):
                # Pick a random move
                candidate_move = random.choice(moves)
            else:
                # Parse output from neural network and make a move
                output = p1.activate(pboard) if turn == BLACK else p2.activate(pboard)
                output = [output[i]*(i in moves) for i in range(len(output))]
                candidate_move = output.index(max(output))
            perform_move(board, candidate_move, turn)
            no_moves_available = False
        elif no_moves_available:
            # End game if neither side can play
            game = False
        else:
            no_moves_available = True
        turn = WHITE if turn == BLACK else BLACK

    # End of game, return score
    return board.count(BLACK), board.count(WHITE)


def run():
    # neural network stuff
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create a population
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-17")

    # Add reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Train neural network
    pe = neat.ParallelEvaluator(1 + multiprocessing.cpu_count(), eval_genome, timeout=60)
    winner = p.run(pe.evaluate, 50)
    # winner = p.run(eval_genomes)

    print("Best fitness -> {}".format(winner))

    stats.save()


if __name__ == '__main__':
    run()
