# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta


class BattleshipEnv(ta.Env):
    """
    Battleship is a two-player turn-based strategy game played on hidden grids,
    where players aim to locate and sink the opposing fleet.
    """

    def __init__(self, grid_size: Optional[int] = 10):
        """
        Initialize the Battleship game.

        Args:
            grid_size (int): Size of the game grid (default: 10x10)
        """
        super().__init__()
        self.grid_size = grid_size
        self.ships = {
            "Aircraft Carrier": 5,
            "Battleship": 4,
            "Submarine": 3,
            "Destroyer": 3,
            "Patrol Boat": 2
        }
        # Regular expression to match coordinates like [A4], [C5]
        self.coord_pattern = re.compile(r"\[([A-Z])(\d+)\]", re.IGNORECASE)

    def get_board_str(self):
        """Return a string representation of the current game state."""
        return self._render_board()

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the game to its initial state."""
        self.state = ta.State(
            num_players=2,
            min_players=2,
            max_players=2,
            max_turns=None,  # No turn limit
            check_truncated=False,
        )

        # Generate boards and ship placements
        board, tracking_board, ship_placements = self._generate_boards()
        
        game_state = {
            "board": board,
            "tracking_board": tracking_board,
            "ship_placements": ship_placements
        }

        self.state.reset(
            seed=seed,
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )

        # Show initial board state to both players
        for player_id in range(2):
            initial_view = self._render_player_view(player_id)
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=player_id,
                message=f"Here is your initial board:\n{initial_view}",
                for_logging=True,
            )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the initial prompt for a player."""
        return (
            f"You are Player {player_id}. You are playing the Battleship game.\n"
            f"Your goal is to sink all of your opponent's ships before they sink yours.\n"
            f"On your turn, you can fire missiles at specific coordinates in the following format: '[A4]'. "
            f"If the missile hits a ship, it is marked with 'X'. If it misses, it is marked with 'O'.\n"
            f"In either scenario, the game environment will inform you of your hits. "
            f"If you have sunk a ship, the game environment will tell you!\n"
            f"The game ends when all of one player's ships have been sunk.\n"
            f"Your board shows your ships and your opponent's hits on you, "
            f"and your hits and misses on your opponent's board without showing your opponent's ships.\n"
        )

    def _generate_boards(self) -> Tuple[Dict[int, List[List[str]]], Dict[int, List[List[str]]], Dict[int, Dict[str, List[Tuple[int, int]]]]]:
        """Generate boards and place ships for both players."""
        # Initialize grids
        board = {
            0: [['~'] * self.grid_size for _ in range(self.grid_size)],
            1: [['~'] * self.grid_size for _ in range(self.grid_size)]
        }
        tracking_board = {
            0: [['~'] * self.grid_size for _ in range(self.grid_size)],
            1: [['~'] * self.grid_size for _ in range(self.grid_size)]
        }
        ship_placements = {0: {}, 1: {}}

        # Place ships for both players
        for player_id in range(2):
            for ship_name, length in self.ships.items():
                placement = self._place_ship_on_board(board[player_id], ship_name, length)
                ship_placements[player_id][ship_name] = placement

        return board, tracking_board, ship_placements

    def _place_ship_on_board(self, grid: List[List[str]], ship_name: str, length: int) -> List[Tuple[int, int]]:
        """Place a ship on the board randomly."""
        placed = False
        directions = ["horizontal", "vertical"]
        max_attempts = 100  # Prevent infinite loops
        attempts = 0

        while not placed and attempts < max_attempts:
            attempts += 1
            direction = random.choice(directions)
            
            if direction == "horizontal":
                row = random.randint(0, self.grid_size - 1)
                col = random.randint(0, self.grid_size - length)
                
                # Check if space is available
                if all(grid[row][col + i] == '~' for i in range(length)):
                    # Place the ship
                    placement = []
                    for i in range(length):
                        grid[row][col + i] = ship_name[0]
                        placement.append((row, col + i))
                    placed = True
                    
            else:  # vertical
                row = random.randint(0, self.grid_size - length)
                col = random.randint(0, self.grid_size - 1)
                
                # Check if space is available
                if all(grid[row + i][col] == '~' for i in range(length)):
                    # Place the ship
                    placement = []
                    for i in range(length):
                        grid[row + i][col] = ship_name[0]
                        placement.append((row + i, col))
                    placed = True

        if not placed:
            raise RuntimeError(f"Could not place ship {ship_name} after {max_attempts} attempts")
        
        return placement

    def _render_board(self) -> str:
        """Render the complete board view (for debugging/admin)."""
        view = []
        view.append("   " + "Player 0's Ships".center(self.grid_size * 3) + "        " + "Player 1's Ships".center(self.grid_size * 3))
        view.append("   " + " ".join([f"{i:2}" for i in range(self.grid_size)]) + "      " + "   " + " ".join([f"{i:2}" for i in range(self.grid_size)]))
        
        for i in range(self.grid_size):
            row_label = chr(i + ord('A'))
            row_player0 = " ".join(f"{cell:2}" for cell in self.state.game_state['board'][0][i])
            row_player1 = " ".join(f"{cell:2}" for cell in self.state.game_state['board'][1][i])
            view.append(f"{row_label}   {row_player0}     {row_label}   {row_player1}")
        
        return "\n".join(view)

    def _render_player_view(self, player_id: int) -> str:
        """Render the player's view of the game."""
        own_grid = self.state.game_state['board'][player_id]
        tracking_grid = self.state.game_state['tracking_board'][player_id]
        player_label = f"Player {player_id}"
        
        view = []
        view.append(f"\n{player_label}'s View".center(self.grid_size * 4 + 15))
        view.append("   " + "Your Ships".center(self.grid_size * 3) + "        " + "Your Hits on Opponent".center(self.grid_size * 3))
        view.append("   " + " ".join([f"{i:2}" for i in range(self.grid_size)]) + "      " + "   " + " ".join([f"{i:2}" for i in range(self.grid_size)]))
        
        for i in range(self.grid_size):
            row_label = chr(i + ord('A'))
            row_own_grid = " ".join(f"{cell:2}" for cell in own_grid[i])
            row_tracking_grid = " ".join(f"{cell:2}" for cell in tracking_grid[i])
            view.append(f"{row_label}   {row_own_grid}     {row_label}   {row_tracking_grid}")
        
        return "\n".join(view)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process the player's action."""
        player_id = self.state.current_player_id
        
        # Log the action
        self.state.add_observation(
            from_id=player_id,
            to_id=-1,
            message=action,
            for_logging=True,
        )

        # Parse the coordinate
        match = self.coord_pattern.search(action)
        if not match:
            self.state.set_invalid_move(
                player_id=player_id,
                reason="The player did not respond with a valid coordinate in square brackets (e.g., [A4])."
            )
            return self.state.step()

        # Convert coordinate to row/col indices
        row = ord(match.group(1).upper()) - ord('A')
        col = int(match.group(2))

        # Validate coordinate bounds
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"The coordinate {match.group(1).upper()}{match.group(2)} is outside the board."
            )
            return self.state.step()

        # Check if already fired at this location
        tracking_board = self.state.game_state['tracking_board'][player_id]
        if tracking_board[row][col] != '~':
            self.state.set_invalid_move(
                player_id=player_id,
                reason=f"The coordinate {match.group(1).upper()}{match.group(2)} has already been fired upon."
            )
            return self.state.step()

        # Process the shot
        opponent_id = 1 - player_id
        opponent_board = self.state.game_state['board'][opponent_id]
        coord_str = f"{match.group(1).upper()}{match.group(2)}"

        if opponent_board[row][col] != '~':
            # Hit!
            ship_initial = opponent_board[row][col]
            tracking_board[row][col] = 'X'
            opponent_board[row][col] = 'X'
            
            # Check if ship is sunk
            ship_sunk = not any(ship_initial in row_cells for row_cells in opponent_board)
            
            if ship_sunk:
                # Ship sunk
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=player_id,
                    message=f"Sunk! You sunk a ship at {coord_str}!\n{self._render_player_view(player_id)}",
                    for_logging=True,
                )
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=opponent_id,
                    message=f"Opponent sunk your ship at {coord_str}!\n{self._render_player_view(opponent_id)}",
                    for_logging=True,
                )
            else:
                # Hit but not sunk
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=player_id,
                    message=f"Hit! You hit a ship at {coord_str}!\n{self._render_player_view(player_id)}",
                    for_logging=True,
                )
                self.state.add_observation(
                    from_id=ta.GAME_ID,
                    to_id=opponent_id,
                    message=f"Opponent hit your ship at {coord_str}!\n{self._render_player_view(opponent_id)}",
                    for_logging=True,
                )
        else:
            # Miss
            tracking_board[row][col] = 'O'
            opponent_board[row][col] = 'O'
            
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=player_id,
                message=f"Miss! You missed at {coord_str}!\n{self._render_player_view(player_id)}",
                for_logging=True,
            )
            self.state.add_observation(
                from_id=ta.GAME_ID,
                to_id=opponent_id,
                message=f"Opponent missed at {coord_str}!\n{self._render_player_view(opponent_id)}",
                for_logging=True,
            )

        # Check for win condition
        if self._check_win(player_id):
            self.state.set_winners(
                player_ids=[player_id],
                reason=f"Player {player_id} has sunk all of their opponent's ships!"
            )
            return self.state.step()

        return self.state.step()

    def _check_win(self, player_id: int) -> bool:
        """Check if the game is over."""
        opponent_board = self.state.game_state['board'][1 - player_id]
        ship_initials = {name[0] for name in self.ships.keys()}
        
        # Check if any ship parts remain on opponent's board
        return not any(any(cell in ship_initials for cell in row) for row in opponent_board)