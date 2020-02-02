02/02/2020 1:30am 
## Q-Learning

<pre>
                                                    Possible wins for state 25
   Rows  _________________________________    ________________________________________
         |                               |    |                                      |
    6    |   O   O   O   O   O   O   O   |    |                                      |
         |                               |    |                                      |
    5    |   O   O   O   O   O   O   O   |    |                                      |
         |                               |    |                                      |
    4    |   O   O   O   O   O   O   O   |    |   22   23   24   25   26   27   28   |
         |                               |    |                                      |
    3    |   O   O   O   O   O   O   O   |    |             17   18   19             |
         |                               |    |                                      |
    2    |   O   O   O   O   O   O   O   |    |         9        11        13        |
         |                               |    |                                      |
    1    |   O   O   O   O   O   O   O   |    |    1              4              7   |                               
         |_______________________________|    |______________________________________|    
            
   Cols -->> 1   2   3   4   5   6   7             1    2    3    4    5    6    7

</pre>

<br><br>

Model:

<br>
6 rows x 7 cols = 42 total states

<br>
Possible actions ---> 7 (action = selected column)

<br>
Reward given only if the robot wins the game

<br>
Possible winning directions from a given state ---> 5 <br>
Check for a win from given state (row and col):

+ function parameters state row and col
+ Loop for (r) ---> -3 to 0
+ loop for (c) ---> -3 to 3
+ count++ if (r and c both are NOT = 0) and (row and col are within model boundries) and (row + r and col + c == true)

<br>
current state to be defined in full (current) sequence of the game in either 1 to 42 states or row and col (eg 1,1;4,2;7,1)

<br>

