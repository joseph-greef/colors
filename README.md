
#### General Controls 
Key     | Effect
------- | ------
escape  | Quit
f       | Toggle Cuda Kernel
p       | Print the current ruleset
[       | Save a screenshot

#### LifeLike Controls
Key     | Effect
------- | ------
c       | Toggle drawing color
e       | Initialize a square in the center of the board of width 2 $radius
i       | Randomly initialize the board, cells have a $density % chance of being alive
r       | Randomize the ruleset
w       | Initialize a diamond in the center of the board with a long exis of 2 $radius
x       | Initialize a cross in the center of the board with width 2 $radius and line width 2 $density
a + NUM | Set the number of refractory generations after a cell dies
d + NUM | Set the $density
m + NUM | Set the dead color scheme
n + NUM | Set the alive color scheme
s + NUM | Set the $radius

<!---
------------------------------- Simulation Controls -------------------------------

q:                      Quits simulation

d:                      randomizes the rule set and randomizes the starting colors

r:                      randomizes the rule set with non-deterministic behaviour and randomizes the 
                        starting colors
                        
j:                      randomizes the rule set for smooth life
                        
f:                      toggle between GPU and CPU calculations       

i:                      reinitialize the rules to their starting versions

x:                      randomizes only the starting colors
                        
c:                      toggles whether the sim draws black and white or colors

v:                      toggles whether to have the background change color if it has not yet been
                        interacted with yet
                        
p:                      prints the ruleset to the console
                        
Left Shift:             pauses the simulation but keeps changing the colors

----------------------------- Change Simulations -----------------------------

b:                      sets the simulation to non-deterministic mode

h:                      sets the simulation to hodge mode

comma:                  sets the simulation to 1D mode

n:                      sets the simulation to normal automata mode

m:                      sets the simulation to smooth mode

l:                      sets the simulation to larger than life mode

                        
----------------------------- Board Initializations -----------------------------

space:                  randomly generates a new board with live cell density $density

g:                      generates a board of gliders

e:                      inits $num_gliders/4 squares of side length $density/10 in each quadrant
                        with vertical and horizontal symmetry

a:                      initializes a board with a center square of side length $density/10

z:                      clears the board

o:                      places a random circle on the board without clearing it

k:                      initializes a random board for smooth life

s:                      initializes a square in the center of the board

t:                      initializes a $num_gliders-gon at the center of the screen

w:                      initializes a circle at the center of the screen

y:                      initializes a board for a 1D cellular automata

------------------------------- Parameter Changes -------------------------------

0 -> 9:                 changes &density from low to high

F1 -> F12:              changes how fast the colors change from fast to slow

Arrow Up/Down:          increases/decreases the number of gliders to generate by 1
                        also changes the number of quadrant dots to generate by 1/4
                        also changes the number of refractory states to have by 1
                      
Arrow Right/Left:       increases/decreases the number of gliders to generate by 4
                        also changes the number of quadrant dots to generate by 1
                        also changes the number of refractory states to have by 4



------------------------------- Genetic Controls -------------------------------

+/=:                    Current ruleset is pretty, add it to the seeds and generate new rules

-/_:                    Current ruleset is not pretty, destroy it and generate new rules

[:			Same as - but for nondeterministic life

]: 			Same as + but for nondeterministic life
--->
