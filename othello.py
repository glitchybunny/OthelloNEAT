# import os
import neat
import random
import multiprocessing
import pickle
from time import sleep
from colorama import Fore, Back, Style

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
    '''
    if best_genome is None:
        other = "random"
        other_fitness = 0
    else:
        other = neat.nn.FeedForwardNetwork.create(best_genome, _config)
        other_fitness = best_fitness
    '''

    # Fight agent to get fitness
    fitness_sum = 0
    count = 500
    for _ in range(count // 2):
        # Play half of the games against random AI
        game_1 = eval_game(net, "random")
        game_2 = eval_game("random", net)
        fitness_sum += game_1[0] + game_2[1]

        # Fight the other half against the current champion
        '''
        game_3 = eval_game(net, other)
        game_4 = eval_game(other, net)
        fitness_sum += game_3[0] + game_4[1]
        fitness_sum += other_fitness * (game_3[0] - game_3[1]) / 64
        fitness_sum += other_fitness * (game_4[1] - game_4[0]) / 64
        '''

    # If fitness is better than the champion, update champion to own genome
    # Also a 1/500 chance to randomly replace it
    '''
    if fitness > best_fitness or (fitness > 0.5 and random.random() < 0.002):
        best_genome = _genome
        best_fitness = fitness
        
    return fitness
    '''

    # Fitness is between 0 and 1
    return ((fitness_sum / count) / 64) ** 2


def eval_genomes(_genomes, _config, _function=eval_genome):
    for _id, _genome in _genomes:
        _genome.fitness = _function(_genome, _config)


def train(_config, _checkpoint=None, _function=eval_genome):
    # Create/load a population
    if _checkpoint is None:
        p = neat.Population(_config)
    else:
        p = neat.Checkpointer.restore_checkpoint(_checkpoint)
    p.config = _config

    # Add reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, 300))

    # Train neural network in parallel
    pe = neat.ParallelEvaluator(1 + multiprocessing.cpu_count(), _function, timeout=30)
    winner = p.run(pe.evaluate)

    # Train neural network
    # winner = p.run(eval_genomes)

    # Save winner
    print("Best fitness -> {}".format(winner))
    stats.save()
    return winner


def eval_game(p1, p2, verbose=False):
    # Setup game
    board, turn = game_setup(p1, p2)

    # Play game
    a, b = game_loop(p1, p2, board, turn, verbose)

    # Determine scores
    if a == 0:
        return 0, 64
    elif b == 0:
        return 64, 0
    else:
        return a, b


def game_setup(p1=None, p2=None):
    # Black always goes first
    board = START_BOARD.copy()
    turn = BLACK

    # But make sure first AI move is random to keep it on its toes
    if p1 not in ["random", "player"]:
        pboard, moves = get_possible_moves(board, turn)
        perform_move(board, random.choice(moves), turn)
        turn = WHITE

        if p2 not in ["random", "player"]:
            pboard, moves = get_possible_moves(board, turn)
            perform_move(board, random.choice(moves), turn)
            turn = BLACK

    return board, turn


def game_loop(p1, p2, board, turn, verbose=False):
    game = True
    skips = 0

    # Do game loop
    while game:
        # Determine player and let them pick a move
        player = p1 if turn == BLACK else p2
        move = game_pick_move(player, board, turn, verbose)

        if move is None:
            # No move available, skip turn
            skips += 1
            if skips >= 2:
                # If no moves available for two turns, end game
                game = False
        else:
            # Otherwise, perform move
            perform_move(board, move, turn)
            skips = 0

        # Swap sides
        turn = WHITE if turn == BLACK else BLACK

    # End of game, return score
    return board.count(BLACK), board.count(WHITE)


def game_pick_move(player, board, turn, _verbose=False):
    # Generate possible moves
    pboard, moves = get_possible_moves(board, turn)
    moves.sort()

    # Print out board
    if _verbose:
        print_board(pboard, turn)

    # Choose which move to play
    move = None
    if len(moves) > 0:
        if player == "player":
            # Let the player pick a move
            move = moves[int(input("Play: "))]
        elif player == "random":
            # Pick a random move
            move = random.choice(moves)
        else:
            # AI picks a move
            # Invert pboard if they're playing as black so the AI always receives consistent values
            # -1 = other player pieces, 1 = ai pieces, 2 = possible moves
            pboard = [i * turn for i in pboard]

            # Parse output from neural network to pick a move
            output = player.activate(pboard)
            output = [output[i] if i in moves else float('-inf') for i in range(len(output))]
            move = output.index(max(output))

    # Print out chosen move
    if _verbose and player != "player":
        sleep(.5)
        if move is None:
            print("No available moves, skipping turn")
        elif player == "random":
            print("Random plays", moves.index(move))
        else:
            print("AI plays", moves.index(move))
        sleep(1.5)

    return move


def get_possible_moves(_board, _turn):
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
        _pboard, _moves = get_possible_moves(_board, _turn)
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


def get_best_genome(_config, _checkpoint, _function=eval_genome):
    # Load checkpoint
    p = neat.Checkpointer.restore_checkpoint(_checkpoint)
    p.config = _config

    # Simulate once to find playable candidate
    pe = neat.ParallelEvaluator(1 + multiprocessing.cpu_count(), _function, timeout=30)
    return p.run(pe.evaluate, 1)


def save_genome(_f, _genome):
    with open(_f, 'wb') as file:
        pickle.dump(_genome, file)


def load_genome(_f):
    with open(_f, 'rb') as file:
        _genome = pickle.load(file)
    return _genome


def play(_config, _p1, _p2, verbose=False):
    # Play game against best genome
    if type(_p1) == neat.genome.DefaultGenome:
        _p1 = neat.nn.FeedForwardNetwork.create(_p1, _config)
    if type(_p2) == neat.genome.DefaultGenome:
        _p2 = neat.nn.FeedForwardNetwork.create(_p2, _config)

    results = eval_game(_p1, _p2, verbose)
    if verbose:
        print("\nFINAL SCORE\nBlack:", results[0], "\nWhite:", results[1])
    return results


if __name__ == '__main__':
    # Load network configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, "config")

    # Train the neural network
    '''
    genome = train(config, 'neat-checkpoint-2569', eval_genome)
    save_genome('elbertson-winner', genome)
    '''

    # Get the best genome and save it
    '''
    genome = get_best_genome(config, "genomes/elbertson/neat-checkpoint-3039")
    save_genome('genomes/e-3039', genome)
    '''

    # Measure a genome's ability by matching it against randoms 1000 times
    #'''
    path = 'genomes/e-3039'
    genome = load_genome(path)
    num_games = 1000
    score_black = 0
    score_white = 0
    for _ in range(num_games):
        if _ % 100 == 0:
            print(_)
        results_black = play(config, genome, "random")
        results_white = play(config, "random", genome)
        score_black += results_black[0]
        score_white += results_white[1]
    score_black /= num_games
    score_white /= num_games
    print(path, "scored\nblack:", score_black, "\nwhite:", score_white,
          "\naverage:", (score_black + score_white)/2)
    #'''

    # Play against a genome
    '''
    genome = load_genome('genomes/e-1733')
    play(config, "player", genome, verbose=True)
    '''

    # Fight two genomes against each other
    '''
    genome1 = load_genome('genomes/dave/d-759')
    genome2 = load_genome('genomes/elbertson/e-1733')
    play(config, genome1, genome2)
    '''
