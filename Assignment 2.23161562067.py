import heapq
import time
from typing import List, Tuple, Optional, Set

State = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]

GOAL_STATE: State = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)

def get_blank_position(state: State) -> Tuple[int, int]:
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return -1, -1

def generate_moves(state: State) -> List[State]:
    moves = []
    x, y = get_blank_position(state)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_board = [list(row) for row in state]
            new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
            moves.append(tuple(tuple(row) for row in new_board))

    return moves

def heuristic(state: State) -> int:
    return sum(
        1 for i in range(3) for j in range(3)
        if state[i][j] != 0 and state[i][j] != GOAL_STATE[i][j]
    )

def greedy_best_first_search(start: State):
    visited: Set[State] = set()
    queue = [(heuristic(start), start, [])]
    nodes_expanded = 0
    start_time = time.time()

    while queue:
        _, current, path = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == GOAL_STATE:
            return path + [current], time.time() - start_time, nodes_expanded

        for move in generate_moves(current):
            if move not in visited:
                heapq.heappush(queue, (heuristic(move), move, path + [current]))

    return None, time.time() - start_time, nodes_expanded

def a_star_search(start: State):
    visited: Set[State] = set()
    queue = [(heuristic(start), 0, start, [])]
    nodes_expanded = 0
    start_time = time.time()

    while queue:
        f, g, current, path = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == GOAL_STATE:
            return path + [current], g, time.time() - start_time, nodes_expanded

        for move in generate_moves(current):
            if move not in visited:
                new_g = g + 1
                new_f = new_g + heuristic(move)
                heapq.heappush(queue, (new_f, new_g, move, path + [current]))

    return None, float('inf'), time.time() - start_time, nodes_expanded

def print_path(path: List[State]):
    for state in path:
        for row in state:
            print(row)
        print("-" * 10)

def compare_algorithms(initial_states: List[State]):   
    for idx, start in enumerate(initial_states):
        print(f"\n====== Test Case {idx + 1} ======")

        print(">>> Greedy Best-First Search:")
        gbfs_path, gbfs_time, gbfs_nodes = greedy_best_first_search(start)
        print(f"Steps: {len(gbfs_path) - 1}")
        print(f"Time: {gbfs_time:.4f}s")
        print(f"Nodes expanded: {gbfs_nodes}")

        print(">>> A* Search:")
        a_star_path, cost, a_star_time, a_star_nodes = a_star_search(start)
        print(f"Steps: {len(a_star_path) - 1}")
        print(f"Time: {a_star_time:.4f}s")
        print(f"Cost: {cost}")
        print(f"Nodes expanded: {a_star_nodes}")

if __name__ == "__main__":
    test_states = [
        ((1, 2, 3), (4, 0, 6), (7, 5, 8)),
        ((1, 2, 3), (0, 4, 6), (7, 5, 8)),
        ((7, 2, 4), (5, 0, 6), (8, 3, 1))
    ]
    compare_algorithms(test_states)
