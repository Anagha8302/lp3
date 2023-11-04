def solve_n_queens(N):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    board = [-1] * N
    
    def solve(row):
        if row == N:
            print_solution(board)
            return
        for col in range(N):
            if is_safe(board, row, col):
                board[row] = col
                solve(row + 1)
            
    def print_solution(board):
        for row in board:
            line = ["Q" if i == row else "." for i in range(N)]
            print(" ".join(line))
        print()
    solve(0)
    
N = int(input("Enter the size of board(N): "))
solve_n_queens(N)