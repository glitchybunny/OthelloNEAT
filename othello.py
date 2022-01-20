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
                while last in [None, -1 * _turn] and 0 <= x <= 7 and 0 <= y <= 7:
                    index = pos_to_index((x, y))
                    x += d[0]
                    y += d[1]
                    if _pboard[index] == 0 and last == -1 * _turn:
                        # update board with possible moves, and keep track of them separately in a list
                        _pboard[index] = 2 * _turn
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
            i = pos_to_index((x, y))
            if move == 1:
                if _board[i] == 0:
                    move = 0
                elif _board[i] == _turn:
                    move = -1
            elif move == -1:
                if _board[i] == -1 * turn:
                    to_change.append(i)
                elif _board[i] == _turn:
                    move = 0
            x += d[0] * move
            y += d[1] * move

        for i in to_change:
            _board[i] = _turn


def index_to_pos(_index):
    return _index % 8, _index // 8


def pos_to_index(_pos):
    return _pos[0] + _pos[1] * 8


def print_board(_board, _turn=None, _moves=None):
    # Print turn
    if _turn == WHITE:
        print("🔸🟠 ORANGE TURN 🟠🔸")
    elif _turn == BLACK:
        print("🔹🔵 BLUE TURN 🔵🔹")

    # Print board itself
    letters = ["⬛️", "🟠", "🔸️️", "🔹", "🔵"]
    for row in range(8):
        print(' '.join([letters[i] for i in _board[row * 8:row * 8 + 8]]))

    # Print possible moves
    print([index_to_pos(i) for i in _moves])


if __name__ == '__main__':
    board = START_BOARD
    turn = BLACK  # black always goes first
    game = True
    cant_play = False

    while game:
        # get possible moves
        pboard, moves = possible_moves(board, turn)
        moves.sort()
        print_board(pboard, turn, moves)

        if len(moves) != 0:
            cant_play = False

            # get player input and play move
            player_move = moves[int(input("Play: "))]
            perform_move(board, player_move, turn)
        else:
            if not cant_play:
                print("Can't play")
                cant_play = True
            else:
                print("Both players can't make a move, game over")
                white_score = board.count(1)
                black_score = board.count(-1)
                if white_score > black_score:
                    print("Orange wins!")
                    # white_score += WIN_BONUS
                elif black_score > white_score:
                    print("Blue wins!")
                    # black_score += WIN_BONUS
                else:
                    print("Orange and blue tied!")
                    # white_score += WIN_BONUS/2
                    # black_score += WIN_BONUS/2
                print("Orange:", white_score)
                print("Blue:", black_score)
                game = False

        # change to other player
        turn *= -1
