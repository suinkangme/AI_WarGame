from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests
#import sys
import sys
# import time
import time

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 3.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None
    heuristic: int = 0
##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0
        

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True
    
    ##how many each unit attacker and defender have? - to calculate heuristics
    num_units_attacker = {'Virus': 2, 'Firewall': 1, 'Program': 2, 'AI': 1}
    num_units_defender = {'Tech': 2, 'Firewall': 2, 'Program': 1, 'AI': 1}
    
    ##check if illegal move by computer was attempted
    comp_illegal_move: bool = False
        
    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)       
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            
            ##control the number of the unit for the correct heuristic calculation
            if unit.type == UnitType.Virus:
                self.num_units_attacker["Virus"] -= 1
                
            if unit.type == UnitType.Tech:
                self.num_units_defender["Tech"] -= 1
                
            if unit.type == UnitType.Program:
                if unit.player == Player.Attacker:
                    self.num_units_attacker["Program"] -= 1
                else:
                    self.num_units_defender["Program"] -= 1
                    
            if unit.type == UnitType.Firewall:
                if unit.player == Player.Attacker:
                    self.num_units_attacker["Firewall"] -= 1
                else:
                    self.num_units_defender["Firewall"] -= 1
                    
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        
        unit = self.get(coords.src)
        
        ##Player in turn must play its own unit, not the opponent's.
        if unit is None or unit.player != self.next_player:
            return False

        unit_dst = self.get(coords.dst)
        
        ## Check if the destination is free
        if self.is_empty(coords.dst):
            
            ## check if destination is adjacent
            adjacent_list = list(coords.src.iter_adjacent())
            if coords.dst not in adjacent_list:
                return False
            
            ## Check if the units(AI, Firewall, Program) are engaged in combat
            for adjacent_coord in adjacent_list:
                adjacent_unit = self.get(adjacent_coord)
                if adjacent_unit is not None and unit.player != adjacent_unit.player:
                    if unit.type in [UnitType.AI, UnitType.Firewall, UnitType.Program]:
                        return False

             ## Check movement restrictions based on player and unit type
            if self.next_player == Player.Attacker:
                if unit.type in [UnitType.AI, UnitType.Firewall, UnitType.Program]:
                    ## Attacker's AI, Firewall, and Program can only move up or left
                    if coords.dst.row > coords.src.row or coords.dst.col > coords.src.col:
                        return False
                else:
                     ## Attacker's Tech and Virus can move left, top, right, bottom
                    if abs(coords.dst.row - coords.src.row) > 1 or abs(coords.dst.col - coords.src.col) > 1:
                        return False
                
            elif self.next_player == Player.Defender:
                if unit.type in [UnitType.AI, UnitType.Firewall, UnitType.Program]:
                    ## Defender's AI, Firewall, and Program can only move down or right
                    if coords.dst.row < coords.src.row or coords.dst.col < coords.src.col:
                        return False
                else:
                     ## Defender's Tech and Virus can move left, top, right, bottom
                    if abs(coords.dst.row - coords.src.row) > 1 or abs(coords.dst.col - coords.src.col) > 1:
                        return False
        
        else:
            ## To perform Attack Action, check if T and S are belong to different players 
            if unit_dst.player != unit.player:       
                ## T must be adjacent to S in any of the 4 directions
                adjacent_list = list(coords.src.iter_adjacent())
                if coords.dst in adjacent_list:
                    return True
        
            ## To perform Repair Action, check if T and S are belong to the same player
            if unit_dst.player == unit.player:
                ## T must be adjacent to S in any of the 4 directions
                adjacent_list = list(coords.src.iter_adjacent())
                if coords.dst in adjacent_list:
                    if unit.type == UnitType.Tech and unit_dst.type == UnitType.Virus:
                        return False
                    elif unit_dst.health == 9:
                        return False 
                    else: 
                        return True                   
         
        return True
    
    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if self.is_valid_move(coords):
                     
            action = f"Move from {coords.src} to {coords.dst}"
            
            ##self destruct
            if coords.src.row == coords.dst.row and coords.src.col == coords.dst.col:
                action += "\n**Self Destruct**\n"
                self.mod_health(coords.src, -9)
                for coord in list(coords.src.iter_range(1)): 
                    target = self.get(coord)
                    if target is not None:
                        self.mod_health(coord, -2)
            else:                 
                ##in case of moving
                target = self.get(coords.dst)
                source = self.get(coords.src)
                
                if target is None:
                    action += "\n**Moving**\n"
                    self.set(coords.dst,source)
                    self.set(coords.src, None)
                else:
                    ##in case of attack
                    if self.get(coords.src).player != target.player:
                        action += "\n**Attack each other**\n"
                        self.mod_health(coords.dst, -source.damage_amount(target))
                        self.mod_health(coords.src, -target.damage_amount(source))
               
                    ##in cases of repair
                    else:
                        action += "\n**Repair**\n"  
                        self.mod_health(coords.dst, source.repair_amount(target))         
            return (True, action)
        
        return (False,"invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self, output):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            # human player can have a chance without penalty
            retry = 1 
            while retry > 0:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    print(result, file = output)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")
                    retry -= 1
            if retry == 0:
                print("No more attempt allowed. Proceeding to the next turn.")

    def computer_turn(self, output) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move(output)
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                print(f"Computer {self.next_player.name}: "+result, file = output)
                self.next_turn()
            else:
                self.comp_illegal_move = True
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.comp_illegal_move:
            if self.options.game_type == GameType.AttackerVsComp:
                return Player.Attacker
            if self.options.game_type == GameType.CompVsDefender:
                return Player.Defender
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)
        
    def minimax(self, start_time, stats_dict, depth, maximize, coord = CoordPair | None) -> Tuple[int, CoordPair | None, float]:
         
        time_limit_searching = (self.options.max_time * 0.6)
        
        if (datetime.now() - start_time).total_seconds() >= time_limit_searching or self.move_candidates() is None or self.turns_played >= self.options.max_turns or depth >= self.options.max_depth:
            return (self.options.heuristic, coord, depth)
        
        game_simul = self.clone()
        move_candidates = list(self.move_candidates())
        best_move = CoordPair()
        value = float('-inf') if maximize else float('inf')

        for child_coord in move_candidates:
            game_simul.perform_move(child_coord)
            maximize = not maximize
            h_score, move, result_depth = game_simul.minimax(start_time, stats_dict, depth + 1, maximize, child_coord)
                
            keys = stats_dict.keys()
            if result_depth not in keys:
                stats_dict.update({result_depth : 1})
            else:
                stats_dict.update({result_depth : stats_dict[result_depth]+1})

            if maximize:
                if h_score > value:
                    best_move = child_coord
                    value = h_score
            else:
                if h_score < value:
                    best_move = child_coord
                    value = h_score

        return (value, best_move, depth)
    
    def minmax_alphabeta(self, start_time, stats_dict, depth, alpha, beta, maximize : bool = False, coord = CoordPair | None)-> Tuple[int, CoordPair | None, float]:
        
        time_check = datetime.now()
        time_duration = (time_check - start_time).total_seconds()
        
        if time_duration >= 0.7*self.options.max_time or list(self.move_candidates()) is None  or self.turns_played >= self.options.max_turns or depth >= self.options.max_depth: 
            return (self.options.heuristic, coord, depth)
        
        else:
            move_candidates = list(self.move_candidates())
            game_simul = self.clone()
            best_move = CoordPair()
            
            if maximize:
                value = alpha
                for children in move_candidates:
                    
                    ## check time and do not go if it is gonna be too late
                    if time_duration >= 0.7*self.options.max_time:
                        return (value, best_move, depth)
                    else:
                        game_simul.perform_move(children)
                        h_score, move, result_depth = game_simul.minmax_alphabeta(start_time, stats_dict, depth+1, alpha, beta, False, children)
                        ##update game stat dictionary
                        keys = stats_dict.keys()
                        if result_depth not in keys:
                            stats_dict.update({result_depth : 1})
                        else:
                            stats_dict.update({result_depth : stats_dict[result_depth]+1})
                                            
                        if(h_score > value):
                            best_move = children
                        value = max(value, h_score)
                        alpha = max(alpha, value)
                        if beta <= alpha:
                            break
                return (value, best_move, depth)
            
            else:
                value = beta      
                for children in move_candidates:
                    ##time limit
                    if time_duration >= 0.7*self.options.max_time:
                        return (value, best_move, depth)
                    ##if you have more time go check more children
                    else:
                        game_simul.perform_move(children)
                        h_score, move, result_depth = game_simul.minmax_alphabeta(start_time, stats_dict, depth+1, alpha, beta, True, children)
                
                        ##update game stat dictionary
                        keys = stats_dict.keys()
                        if result_depth not in keys:
                            stats_dict.update({result_depth : 1})
                        else:
                            stats_dict.update({result_depth : stats_dict[result_depth]+1})
                    
                        ##pruning
                        if(h_score < value):
                            best_move = children
                        value = min (value, h_score)
                        beta = min (beta, value)
                        if beta <= alpha:
                            break
                            
                return (value, best_move, depth)  
        
    
    def suggest_move(self, output) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
               
        #This is a string to be printed in the output file
        report =""
        
        old_non_root_node = 0
        old_non_leaf_node = 0
        
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            old_non_root_node += self.stats.evaluations_per_depth[k]
            if k != len(self.stats.evaluations_per_depth.keys()) - 1:
                old_non_leaf_node += self.stats.evaluations_per_depth[k]
        old_non_root_node +=1 
       
        start_time = datetime.now()
        #(score, move, avg_depth) = self.random_move()
        
        if self.options.alpha_beta:
            if self.next_player == Player.Attacker:
                result = self.minmax_alphabeta(start_time, self.stats.evaluations_per_depth, 0, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE, True)                    
            else:
                result = self.minmax_alphabeta(start_time, self.stats.evaluations_per_depth, 0, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE, False)
        else: 
            if self.next_player == Player.Attacker:
                result = self.minimax(start_time, self.stats.evaluations_per_depth, 0, True)    
            else:
                result = self.minimax(start_time, self.stats.evaluations_per_depth, 0, False)
     
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        
        #depth is only useful inside the minimax method.
        score, move, depth = result
        
        ##if elapsed_seconds is over the max_time, AI lose
        if elapsed_seconds > self.options.max_time:
            move = None
            return move
               
        report += f"Heuristic score: {score}\n"
        
        print(f"Heuristic score: {score}")
        
        total_evals = sum(self.stats.evaluations_per_depth.values())
        report += f"Cumulative evals: {total_evals}\n"
        print(f"Cumulative evals: {total_evals}")
        
        new_non_root_node = 0
        new_non_leaf_node = 0
        
        print(f"Evals per depth: ",end='')
        report += f"Evals per depth: \n" 
        keys = sorted(self.stats.evaluations_per_depth.keys())
        for k in keys:
            #print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
            new_non_root_node += self.stats.evaluations_per_depth[k]
            if k != keys[-1]:
                new_non_leaf_node += self.stats.evaluations_per_depth[k]
                report += f"{k}:{self.stats.evaluations_per_depth[k]} "
        new_non_root_node += 1
        print()
        report += "\n"
        
        #print(f"Evals per depth percentage: ", end='')
        report += f"Evals per depth percentage: \n"
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            calcul = (self.stats.evaluations_per_depth[k]/total_evals)*100
            #print(f"{k}: {calcul:0.1f}% ", end='')
            report += f"{k}: {calcul:0.1f}% "
        print()
        report += "\n"

        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")   
      
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        report += f"Elapsed time: {elapsed_seconds:0.1f}s\n"
        print(f"Branching Factor: {abs(new_non_root_node - old_non_root_node) / abs(new_non_leaf_node - old_non_leaf_node): 0.1f}")
        report += f"Branching Factor: {abs(new_non_root_node - old_non_root_node) / abs(new_non_leaf_node - old_non_leaf_node): 0.1f}"
        
        print(report, file = output)
        print(move)
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    
    ## method added to print initial config
    def print_initial(self, output, options : Options):
        if options.game_type == GameType.AttackerVsDefender:
            play_mode = 'Player 1: Human vs Player 2: Human'
        elif options.game_type == GameType.AttackerVsComp:
            play_mode = 'Player 1: Human vs Player 2: AI'
        elif options.game_type == GameType.CompVsDefender:
            play_mode = 'Player 1: AI vs Player 2: Human'
        else:
            play_mode = 'Player 1: AI vs Player 2: AI'
        game_param = f'Game Mode\n{play_mode}\nTimeout in seconds: {options.max_time}\nMax # of turns: {options.max_turns}\n'
        print(game_param, file = output)
        print(self, file = output)       
    
    ## method added for the heuristic e0
    def e0(self):
        return (((3*self.num_units_attacker["Virus"])
                 +(3*self.num_units_attacker["Firewall"])
                 +(3*self.num_units_attacker["Program"])
                 +(9999*self.num_units_attacker["AI"]))-
                ((3*self.num_units_defender["Tech"])
                 +(3*self.num_units_defender["Firewall"])
                 +(3*self.num_units_defender["Program"])
                 +(9999*self.num_units_defender["AI"])))
    
    
    # heuristic e1 : add penalties based on the health level of the AI units
    def e1(self):
        
        attacker_ai_unit = Unit(player=Player.Attacker, type=UnitType.AI)
        defender_ai_unit = Unit(player=Player.Defender, type=UnitType.AI)

        attacker_ai_health = self.unit_health_penalty(attacker_ai_unit.health)
        defender_ai_health = self.unit_health_penalty(defender_ai_unit.health)

        return (((10*self.num_units_attacker["Virus"])
                 +(5*self.num_units_attacker["Firewall"])
                 +(0.1*self.num_units_attacker["Program"])
                 +(9999*self.num_units_attacker["AI"]) * attacker_ai_health)-
                 
                ((10*self.num_units_defender["Tech"])
                 +(5*self.num_units_defender["Firewall"])
                 +(0.1*self.num_units_defender["Program"])
                 +(9999*self.num_units_defender["AI"]) * defender_ai_health))
    
    # give health penalty based on remaining health level of the units.
    def unit_health_penalty(self, health):
        if health>= 9:
            return 0.1
        elif health >= 6:
            return 3
        elif health >= 3:
            return 5
        else:
            return 10
        
    
    # heuristic e2 : less weight for the program unit 
    def e2(self):

        attacker_ai_coord = None
        penalty_e2 = 0

        # find Attacker's AI coord
        for (coord, unit) in self.player_units(Player.Attacker):
            if unit.type == UnitType.AI:
                attacker_ai_coord = coord
                break      

        # iterate over Coords inside a rectangle centered on Attacker's AI
        surrounding_coords = list(attacker_ai_coord.iter_range(1))      
        for coord in surrounding_coords:
            unit = self.get(coord)
            if unit is not None:
                player_name = unit.player.name
                unit_type = unit.type.name

                # identify the unit belongs to which player
                # calculate penalty for attacker's unit
                if player_name == Player.Attacker:
                    if unit_type == UnitType.Virus:
                        penalty_e2 += 1000
                    elif unit_type == UnitType.Program:
                        penalty_e2 += 50
                    elif unit_type == UnitType.Firewall:
                        penalty_e2 += 1

                # calculate penalty for attacker's unit
                elif player_name == Player.Defender:
                    if unit_type == UnitType.Tech:
                        penalty_e2 -= 500
                    elif unit_type == UnitType.AI:
                        penalty_e2 -= 250
                    elif unit_type == UnitType.Program:
                        penalty_e2 -= 100
                    elif unit_type == UnitType.Firewall:
                        penalty_e2 -= 1

        return penalty_e2

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    
    ##input max num of turns
    parser.add_argument('--max_turns', type = int, help = 'maximum number of turns')
    
    # select the heuristic function among e0,e1, and e2
    parser.add_argument('--heuristic', type=int, default=0, help='heuristic function: (0: e0, 1: e1, 2: e2)')

     # input minimax algorithm
    parser.add_argument('--minimax', type=str, default = "on", help = 'turn on/off minimax mode, on|off')

    ##input alpha-beta search mode
    parser.add_argument('--alpha_beta', type=str, default = "on", help = 'turn on/off alpha-beta search mode, on|off')
    
    args = parser.parse_args()
    
    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
        
    ## set up max num of turns as input value
    if args.max_turns is not None:
        options.max_turns = args.max_turns
        options.minimax = False    

    # create a new game
    game = Game(options=options)

    # determine which heuristic algorithm will be used
    if args.game_type != "manual" and (game_type == GameType.AttackerVsComp or game_type == GameType.CompVsDefender or game_type ==GameType.CompVsComp):
        if args.heuristic == 0:
            options.heuristic = game.e0()
        elif args.heuristic == 1:
            options.heuristic = game.e1()
        elif args.heuristic == 2:
            options.heuristic = game.e2()
        else:
            print("Invalid choice. Using the default heuristic (e0).")
            options.heuristic = game.e0()

    ##if game_type is human vs human or alpha_beta was asked to be off, turn off alpha beta
    if args.game_type == "manual" or args.alpha_beta == "off":
        options.alpha_beta = False        
    
    ## open file and print initial configuration
    title = f'gameTrace--{options.alpha_beta}-{options.max_time}-{options.max_turns}'
    outputFile = open(f'{title}.txt', 'w')
    game.print_initial(outputFile, options)
    
    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            
            ##print the winner to the output file
            print(f"{winner.name} wins in {game.turns_played} turn!", file = outputFile)
            outputFile.close()
            
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn(outputFile)
            print(game, file = outputFile)
            
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn(outputFile)
            print(game, file = outputFile)
            
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn(outputFile)
            print(game, file = outputFile)
            
        else:
            player = game.next_player
            move = game.computer_turn(outputFile)
            print(game, file = outputFile)
            
            ##computer illegal move - AI automatically lose
            if game.comp_illegal_move:
                auto_lose = "Computer attempted illegal move. Game Over.\n ***Human player wins!***"
                print(auto_lose)
                print(auto_lose, file = outputFile)
                winner = game.has_winner()
                print(f"{winner.name} wins!")
                print(f"{winner.name} wins in {game.turns_played} turn!", file = outputFile)
                outputFile.close()
                exit(1)
                
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                outputFile.close()
                exit(1)

##############################################################################################################

if __name__ == '__main__':
    main()