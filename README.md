# Summary
This package models the intestinal tract of a chicken

It's main purpose is to determine which feeds are most optimal by determining how much protein
is left after a chicken poops.

# How to run
* Run the following in bash/terminal
```
python PATH_TO_PACKAGE/chickgut/main.py
```

# Todo List:
* Move critical code out of main.py and into its own separate area 
* Write tests to verify existing functionality
    * Add a code coverage report so I can see what's covered and what's not
    * Use AI to write the tests. Very light, but necessary since I don't understand all the calculations
* !Bottlenecks are caused by the `solve_ivp` function. Ideally move to JAX to use GPU instead
* !Replace existing differential evolution solution with scipy to use pymoo
    * Add checkpointing via pymoo + pickles

## Done
* Restructure the python code ✅
* Write a "How do you run this" guide in the ReadMe ✅
* Time each part to understand bottle necks ✅

## Maybe
* *Bonus* - Find out how to implement a Digital Twin-like system where we can have multiple computers computing at once 
    * V: This is not necessary since this would require 2 computers. It's a nice-to-have for something more complex

# Personal Notes
* Rough calculations show that for the ideal popsize of 100 & maxiter of 1000 it'll be 100K evaluations. 
    * At 2 seconds per eval, this will take 2.3 days to complete
    * At the current pace of 20 seconds per eval, it takes 23 days
    * So instead, for the 2.3 days, have it run each of the feeds in parallel
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

# Performance Notes
* Function HindGIT took 37.265013s
    * 20 seconds in DuoDenum
    * 9 seconds in ileum
    * 7 seconds in jejunum

```
##Evaluation 22
Function solving_Unode0 took 0.062765s
Solving duodenum undigestible protein equations took 0.888347 seconds
Solving duodenum slowly-digested protein equations took 17.913205 seconds
Function solving_duo_USl took 18.801604s
Function flatten_result_duo_CPsl took 0.071319s
Function SlP_for_RP_d took 0.071729s
Function solving_duo_R took 1.926259s
Function solving_duo_feed took 0.040284s
Solving duodenum equations took 20.830977 seconds
Function flatten_result_duo_CPsl took 0.070310s
Flattening duodenum results took 0.234086 seconds
finished duodenum

~~~Calculating duodenum properties took 21.065130 seconds
Function solving_jej_USl took 0.543761s
Function solving_jej_R took 6.664935s
Function solving_jej_feed took 0.013195s
Solving jejunum equations took 7.221953 seconds
Flattening jejunum results took 0.212703 seconds
finished jejunum

~~~Calculating jejunum properties took 7.434720 seconds
Function solving_il_USl took 0.280376s
Function solving_il_R took 5.375315s
Function solving_il_feed took 0.005631s
Solving ileum equations took 5.661380 seconds
Flattening ileum results took 0.194890 seconds
finished ileum

~~~Calculating ileum properties took 5.886197 seconds
Function HindGIT took 34.386133s
```