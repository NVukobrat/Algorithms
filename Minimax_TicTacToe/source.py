import random
import numpy as np

# Variables
FIRST_PLAYER = "X"
SECOND_PLAYER = "O"
EMPTY_FIELD = "_"


def create_board():
    return (np.array([[EMPTY_FIELD, EMPTY_FIELD, EMPTY_FIELD],
                      [EMPTY_FIELD, EMPTY_FIELD, EMPTY_FIELD],
                      [EMPTY_FIELD, EMPTY_FIELD, EMPTY_FIELD]]))


def print_board(board):
    for i in range(len(board)):
        for j in range(len(board)):
            print(board[i][j], end=" ")
        print()
    print()


def make_move(board, player, rnd=False):
    if rnd:
        return random_place(board, player)
    else:
        return smart_place(board, player)


def random_place(board, player):
    possible_moves = possibilities(board)
    next_move = random.choice(possible_moves)
    board[next_move] = player

    return board


def possibilities(board):
    left_empty_fields = []

    for i in range(len(board)):
        for j in range(len(board)):

            if board[i][j] == EMPTY_FIELD:
                left_empty_fields.append((i, j))

    return left_empty_fields


def smart_place(board, player):
    possible_moves = possibilities(board)

    best_score = -2187
    best_move = None
    depth = 0
    for move in possible_moves:
        board[move] = player
        score = minimax(board, player, depth)
        board[move] = EMPTY_FIELD

        if score > best_score:
            best_score = score
            best_move = move

    board[best_move] = player

    return board


def minimax(board, player, depth):
    depth += 3

    winner = evaluate(board)
    if winner != 0:
        if winner == FIRST_PLAYER:
            return depth - 243
        elif winner == SECOND_PLAYER:
            return 243 - depth
        else:
            return 0

    possible_moves = possibilities(board)
    if player == FIRST_PLAYER:
        best_score = 2187
        for move in possible_moves:
            board[move] = player
            best_score = min(best_score, minimax(board, change_player(player), depth))
            board[move] = EMPTY_FIELD

        return best_score

    if player == SECOND_PLAYER:
        best_score = -2187
        for move in possible_moves:
            board[move] = player
            best_score = max(best_score, minimax(board, change_player(player), depth))
            board[move] = EMPTY_FIELD

        return best_score


def change_player(player):
    if player == FIRST_PLAYER:
        return SECOND_PLAYER
    else:
        return FIRST_PLAYER


def remove_next_move(next_move, possible_moves):
    index = None
    for i, move in enumerate(possible_moves):
        if move == next_move:
            index = i
            break

    possible_moves.pop(index)

    return possible_moves


def evaluate(board):
    winner = 0

    for player in [FIRST_PLAYER, SECOND_PLAYER]:
        if row_win(board, player) or \
                col_win(board, player) or \
                main_diag_win(board, player) or \
                anti_diag_win(board, player):
            winner = player

    if np.all(board != EMPTY_FIELD) and winner == 0:
        winner = "Nobody"

    return winner


def row_win(board, player):
    for x in range(len(board)):
        win = True

        for y in range(len(board)):
            if board[x, y] != player:
                win = False
                continue

        if win:
            return win

    return win


def col_win(board, player):
    for x in range(len(board)):
        win = True

        for y in range(len(board)):
            if board[y][x] != player:
                win = False
                continue

        if win:
            return win

    return win


def main_diag_win(board, player):
    win = True

    for x in range(len(board)):
        if board[x, x] != player:
            win = False
            break

    return win


def anti_diag_win(board, player):
    win = True

    for x in range(len(board)):
        for y in range(len(board)):
            if x == len(board) - y - 1:
                if board[x, y] != player:
                    win = False
                    break

    return win


def main():
    """
    Minimax belongs to the family of the backtracking algorithms.
    Backtracking is an algorithmic-technique for solving problems
    recursively by trying to build a solution incrementally, one
    piece at a time, removing those solutions that fail to satisfy
    the constraints of the problem at any point of time. It is mostly
    used in two player games like Tic-Tac-Toe, Backgammon, Mancala,
    Chess, etc.

    In Minimax the two players are called maximizer and minimizer.
    The maximizer tries to get the highest score possible while
    the minimizer tries to do the opposite and get the lowest score
    possible. This is done by some heuristics, which are unique for
    every type of game.
    """
    rounds = 100
    first = 0
    second = 0
    nobody = 0
    for i in range(rounds):
        board, winner = create_board(), 0
        # print_board(board)

        while winner == 0:
            for player in [FIRST_PLAYER, SECOND_PLAYER]:
                if player == FIRST_PLAYER:
                    board = make_move(board, player, rnd=True)
                elif player == SECOND_PLAYER:
                    board = make_move(board, player, rnd=False)
                # print_board(board)

                winner = evaluate(board)
                if winner != 0:
                    break

        # print("Winner is: " + winner)
        if winner == FIRST_PLAYER:
            first += 1
        elif winner == SECOND_PLAYER:
            second += 1
        else:
            nobody += 1

        print("Played {0}/{1}".format(i, rounds))
        print_board(board)

    print("Results: ")
    print("First won {:f}".format((first / float(rounds)) * 100))
    print("Second won {:f}".format((second / float(rounds)) * 100))
    print("Nobody won {:f}".format((nobody / float(rounds)) * 100))


if __name__ == '__main__':
    main()
