# AI_WarGame

## COMP 472 Fall 2023 Project ##

Instructor: Leila Kosseim, Section NN <br/>
Team: Superintelligence

Student Id | Name
-----------|-------
40129337   | Suin Kang
40174913   | Hyun Soo Kim


***Deliverable 1***

For this deliverable, 

  * we completed the validation and execution logic in the is_valid_move() and perform_move() methods in the Game class. 

  * The implementation of these methods enables players to engage in manual mode, playing against each other (human vs. human).

  * Players have the option to disable the alpha-beta algorithm by activating the manual mode of the game. 

  * Furthermore, they can configure the maximum number of turns and the maximum time allowed for each turn.

  * An output file will be generated, documenting the details of each action and the winner.

  * To play manual mode and set up the max time for each turn as 3 seconds, maximum number of turns as 50, type like this:<br/>
    `python ai_wargame.py --game_type "manual" --max_turns 50 --max_time 3.0`
  
***Deliverable 2***

For this deliverable,

 * we added minimax and alpha beta pruning features.
 
 * Now, game type other than manual is also playable (Computer vs. Human, Human vs. Computer, Computer vs. Computer)

 * For the evaluation, e0, e1, and e2 functions are also added.

 * By passing integer parameter 0 1 2, user can choose which evalulation function for the game. By default, e0 is used.

 * For example, to play auto mode (Computer vs Computer mode) with evaluation function e1, type like this in the terminal:<br/>
  `python ai_wargame.py --game_type "auto" --heuristic 1`
