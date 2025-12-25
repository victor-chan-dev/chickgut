# Summary
This package models the intestinal tract of a chicken

It's main purpose is to determine which feeds are most optimal by determining how much protein
is left after a chicken poops.

# Todo List:
* Restructure the python code
* Move critical code out of main.py and into its own separate area
* Write tests to verify existing functionality
* Write a "How do you run this" guide in the ReadMe
* Time each part to understand bottle necks
* Replace existing differential evolution solution with scipy to use pymoo
    * Add checkpointing via pymoo + pickles
* Re-time and add estimates on how to handle different feeds
* *Bonus* - Find out how to implement a Digital Twin-like system where we can have multiple computers computing at once

# Personal Notes
* Loop through all of the different feeds
* optimize_params
* investigate plug flow
* Apparent digestability of protein
    * The less at the end, the more digestable
        * This tells us how useful the food 

* Is it better to loop through each one or do 1 at a time?
    * one at a time
* Saving progress is great
* Digital twin?