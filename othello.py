# import os
import neat
import random
import multiprocessing
import pickle
from time import sleep
from copy import copy
from colorama import Fore, Back, Style

# from itertools import combinations

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

# keep running track of the best neural network to compete against
best_genome = None
best_fitness = 0


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


def perform_move(_board, _index, _turn, _debug=None):
    if _board[_index] != 0:
        # Print extra information for debug
        _pboard, _moves = possible_moves(_board, _turn)
        print_board(_pboard, _turn)
        print_board(_board, None, [_index])
        print(_debug)
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
    print()
    if _turn is not None:
        # Print turn
        if _turn == WHITE:
            print(Back.GREEN + Fore.RESET, " ░▒▓█ WHITE TURN  █▓▒░ ", Style.RESET_ALL)
        elif _turn == BLACK:
            print(Back.GREEN + Fore.BLACK, " ░▒▓█ BLACK TURN  █▓▒░ ", Style.RESET_ALL)

    # Print board
    letters = [Fore.BLACK + "░░", Fore.RESET + "██", Fore.BLACK + "▏▕", Fore.BLACK + "██"]
    counter = 0
    for row in range(8):
        chars = [letters[i] for i in _board[row * 8:row * 8 + 8]]
        for i in range(len(chars)):
            if chars[i] == Fore.BLACK + "▏▕":
                chars[i] = Fore.RESET if _turn == WHITE else Fore.BLACK
                chars[i] += str(counter).rjust(2, ' ')
                counter += 1
        print(Back.GREEN, ' '.join(chars), Style.RESET_ALL)

    if _moves is not None:
        # Print possible moves
        print([index_to_pos(i) for i in _moves])


def eval_genome(_genome, _config):
    global best_genome, best_fitness

    # Generate network
    net = neat.nn.FeedForwardNetwork.create(_genome, _config)

    # Choose agent to compete against
    if best_genome is None:
        other = "random"
        other_fitness = 0
    else:
        other = neat.nn.FeedForwardNetwork.create(best_genome, _config)
        other_fitness = best_fitness

    # Fight agent to get fitness
    fitness_sum = 0
    count = 20
    for _ in range(count // 4):
        # Play half of the games against random AI
        game_1 = eval_game(net, "random")
        game_2 = eval_game("random", net)
        fitness_sum += game_1[0] + game_2[1]

        # Fight the other half against the current champion
        game_3 = eval_game(net, other)
        game_4 = eval_game(other, net)
        fitness_sum += game_3[0] + game_4[1]
        fitness_sum += other_fitness * (game_3[0] - game_3[1]) / 100
        fitness_sum += other_fitness * (game_4[1] - game_4[0]) / 100

    fitness = fitness_sum / count

    # If fitness is better than the champion, update champion to own genome
    # Also a 1/500 chance to randomly replace it
    if fitness > best_fitness or random.random() < 0.002:
        best_genome = _genome
        best_fitness = fitness

    return fitness


def eval_genomes(_genomes, _config):
    for _id, _genome in _genomes:
        _genome.fitness = eval_genome(_genome, _config)


def eval_game(p1, p2, verbose=False):
    # Setup game
    board, turn = game_setup(p1)

    # Play game
    a, b = game_loop(p1, p2, board, turn, verbose)

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


def game_loop(p1, p2, board, turn, verbose=False):
    game = True
    no_moves_available = False

    # Do game loop
    while game:
        # Determine player and possible moves
        player = p1 if turn == BLACK else p2
        pboard, moves = possible_moves(board, turn)
        moves.sort()

        # Log
        if verbose:
            print_board(pboard, turn)

        # Choose move
        candidate_move = None
        if len(moves) > 1:  # Multiple moves, choose based on player
            if player == "random":
                # Pick a random move
                candidate_move = random.choice(moves)
            elif player == "player":
                # Let the player pick a move
                candidate_move = moves[int(input("Play: "))]
            else:
                # Parse output from neural network to pick a move
                output = player.activate(pboard)
                output = [output[i] if i in moves else float('-inf') for i in range(len(output))]
                candidate_move = output.index(max(output))
                if verbose:
                    sleep(1)
                    print("AI plays", moves.index(candidate_move))
                    sleep(1)
            no_moves_available = False
        elif len(moves) == 1:  # Only one move, forced to play it
            candidate_move = moves[0]
            no_moves_available = False
        elif no_moves_available:  # End game if neither side can play
            game = False
        else:
            no_moves_available = True

        # Perform move
        if candidate_move is not None:
            perform_move(board, candidate_move, turn)

        # Swap sides
        turn = WHITE if turn == BLACK else BLACK

    # End of game, return score
    return board.count(BLACK), board.count(WHITE)


def train(_config):
    # Create/load a population
    # p = neat.Population(_config)
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-489")
    p.config = _config

    # Add reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, 300))

    # Train neural network
    pe = neat.ParallelEvaluator(1 + multiprocessing.cpu_count(), eval_genome, timeout=30)
    winner = p.run(pe.evaluate)
    # winner = p.run(eval_genomes)

    # Save winner
    print("Best fitness -> {}".format(winner))
    stats.save()
    return winner


def get_best_genome(_config, _checkpoint):
    # Load checkpoint
    p = neat.Checkpointer.restore_checkpoint(_checkpoint)
    p.config = _config

    # Simulate once to find playable candidate
    pe = neat.ParallelEvaluator(1 + multiprocessing.cpu_count(), eval_genome, timeout=30)
    return p.run(pe.evaluate, 1)


def save_genome(_f, _genome):
    with open(_f, 'wb') as file:
        pickle.dump(_genome, file)


def load_genome(_f):
    with open(_f, 'rb') as file:
        _genome = pickle.load(file)
    return _genome


def play(_config, _genome, verbose=False):
    # Play game against best genome
    net = neat.nn.FeedForwardNetwork.create(_genome, _config)
    results = eval_game("player", net, verbose)
    print("\nFINAL SCORE\nBlack:", results[0], "\nWhite:", results[1])


if __name__ == '__main__':
    # Configure neural network
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, "config")

    # Train the neural network
    # genome = train(config)
    # save_genome('dave-winner', genome)

    # Get the best genome and save it
    # genome = get_best_genome(config, "genomes/dave/neat-checkpoint-759")
    # save_genome('genomes/dave/dave-759', genome)

    # Load a genome and play against the neural network
    genome = load_genome('genomes/dave/dave-759')
    play(config, genome, verbose=True)
