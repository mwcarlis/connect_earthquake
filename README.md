# connect_earthquake
Connect Four variant with random chance AI.

Here is an example transcript of a connect_earthquake game:

$ python connect_earthquake.py

Would you like to go first (y/n)? y
Should I tell you when earthquake happen (y/n)?
(If not, each round, I will ask you if an earthquake happened)y
| | | | | | | |
|1|2|3|4|5|6|7|
Please enter a slot from 1 to 7 for your move:4
Board After Your Move:
| | | | | | | |
| | | |X| | | |
|1|2|3|4|5|6|7|
Board After My Move:
| | | | | | | |
| | |O|X| | | |
|1|2|3|4|5|6|7|
Board After Earthquake Check:
| | | | | | | |
| | |O|X| | | |
|1|2|3|4|5|6|7|
Please enter a slot from 1 to 7 for your move:4
Board After Your Move:
| | | | | | | |
| | | |X| | | |
| | |O|X| | | |
|1|2|3|4|5|6|7|
Board After My Move:
| | | | | | | |
| | | |X| | | |
| | |O|X|O| | |
|1|2|3|4|5|6|7|
Board After Earthquake Check:
| | | | | | | |
| | | |X| | | |
|1|2|3|4|5|6|7|
.... (many moves omitted)
Board After My Move:
| | | | | | | |
| | | | |O|O| |
| | | |O|X|X| |
| | |O|X|X|O|X|
| |O|X|X|O|O|X|
|1|2|3|4|5|6|7|
I win!!! Game over!
