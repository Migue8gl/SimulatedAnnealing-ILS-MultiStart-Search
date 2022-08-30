# SimulatedAnnealing-ILS-MultiStart-Search
Trajectory techniques for the Weight Learning Problem.

- This practice was done in the Metaheuristic (Metaheurísticas) subject at the UGR. It consists in study, understand and execute different trajectory based algorithms such as Simulated annealing, iterated local search or multistart search to solve the weight learning problem in three different data sets.

### Data Sets:
  * Parkinsons – data used to distinguish presence or absence
of Parkinson's disease. It consists of 195 examples, 23 attributes and 2 classes.

  * Spectf-heart – contains attributes calculated from images
medical computed tomography. The task is to find out if the
physiology of the heart is correct. Consists of 267 examples, 45 attributes
and 2 classes.

  * Ionosphere – radar data collected at Goose Bay. Its objective is to
classify electrons as "good" and "bad" depending on whether they have
some kind of structure in the ionosphere. It consists of 352 examples, 34
attributes and 2 classes.

### Algorithms:
- Simulated Annealing: A way to prevent local-search from ending at local optima,
fact that usually occurs with traditional local-search algorithms, is to allow some moves to be towards worse solutions. SA achieves this because of of a probability function that will
decrease the probability of moving towards worse solutions while the
search is moving along.

- Iterated local-search (ILS): It is based on applying local-search to an initial solution repeatedly.

- Multistart search: This search applies several local-searchs to random solutions. The best of them or the one that satisfies a certain condition will be return.

- ILS-SA hybridization: SA is used instead local-search.

-----HOW TO RUN IT-----

1 - To compile run "make" in the terminal.

2 - To run the program go to the terminal and run "./bin/p3 <seed> <parameter>" where
<seed> is the seed we choose and
<parameter> can be:
		-> 1: spectf-heart
		-> 2: parkinsons
		-> 3: ionosphere

--- Other things will get an error ---

[USE] -> A script is attached for a more comfortable execution of the results in text files, the script is "script_acp.sh".
All data sets can be executed concurrently (which makes the individual times of each program greater, but it is faster when it comes to obtain results).
If, on the contrary, it is desired, it can be executed sequentially. All of this is specified in the script itself.
