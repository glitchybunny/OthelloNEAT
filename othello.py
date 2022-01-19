import os

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


def possible_moves(gameboard, turn):
    pboard = gameboard.copy()
    valid_moves = []
    for i in range(len(pboard)):
        if pboard[i] == turn:
            # get current board pieces
            pos = index_to_pos(i)

            # walk in each oct direction to find possible moves
            for d in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
                valid = True
                last = None
                x = pos[0] + d[0]
                y = pos[1] + d[1]
                while valid and 0 <= x <= 7 and 0 <= y <= 7:
                    index = pos_to_index((x, y))
                    x += d[0]
                    y += d[1]
                    if pboard[index] == -1 * turn:
                        last = pboard[index]
                    elif pboard[index] == 0 and last == -1 * turn:
                        # update board with possible moves, and keep track of them separately in a list
                        pboard[index] = 2 * turn
                        valid_moves.append(index)
                        valid = False
                    else:
                        valid = False
    return pboard, valid_moves


def index_to_pos(i):
    return i % 8, i // 8


def pos_to_index(pos):
    return pos[0] + pos[1] * 8


def print_board(gameboard, _turn=None, _moves=None):
    print('\n\n')

    # Print turn
    if _turn == WHITE:
        print("🔸🟠 ORANGE TURN 🟠🔸")
    elif _turn == BLACK:
        print("🔹🔵 BLUE TURN 🔵🔹")

    # Print board itself
    letters = ["⬛️", "🟠", "🔸️️", "🔹", "🔵"]
    for row in range(8):
        print(' '.join([letters[i] for i in gameboard[row * 8:row * 8 + 8]]))

    # Print possible moves
    print([index_to_pos(i) for i in _moves])


if __name__ == '__main__':
    board = START_BOARD
    turn = BLACK

    while True:
        pboard, moves = possible_moves(board, turn)
        moves.sort()

        print_board(pboard, turn, moves)
        board[moves[int(input("Play: "))]] = turn

        turn *= -1



