The Perceptron Algorithm implement concurently with MPI + CUDA + OMP
====================================================================

---------------------------------------------------------------------
MPI

I choose to use MPI staticly.
at first the Master read all points from the file.
Then he takes the tMax time and split it according to the number of process
(Include the master itself - which mean no one is resting while other is working!) (Social Democrat program :) )

Then, Each process has its own "Time Zone" for making the perceptron alogrithm.
The first process who will find the result we the shorest time - will win and that paricular process result will be
written to the file only by the MASTER

Why I chose to make MPI staticly and not dynamic?
Well That is a good question!
In our course we learned that comunication between process takes A LOT of time,
if the communication happened not in the same computer but in DIFFRENT computer,
the time of comminucaion will be ASTRONOMIC!
In the dynamic approach, each process that finished the job need to ask the MASTER again
"What to do next?"
LOT OF WASTE OF TIME WHILE COMUNICATE!

this is why I chose the static why. which maybe says that each process will work a  little bit harder,
but if we look at the big picture it will be much more worth it!

Complexety - O(Tmax/ dt)


-----------------------------------------------------------------------
CUDA

One of the most best thing about Cuda is that he have an infinat resource of threads
(well not infinat... but definitly alot!)
This is why Cuda is an important tool of my algorithm.

In the perceptorn algorithm there are 2 main things that need to check all points and update some data if neccesary.
The first one - calculate if a point is in the right place according to the line we draw
The second one - update the location on a point according to the pyhsic formula.

Those two scanrios I mentiond now require to pass ALL of the points and make an update.
If we had 2 or 3 points that is not a problem at all...
but imagen we have 500000, Well not so little anymore.
with Cuda we can check every point and update it concurencly!

this is why I chose to use Cuda on this 2 pariculare scanrios.

Complexety - O(k) when k is the dimension (Both in calcnMissWithKernel and updateLocationOfAllPointsWithKernel

We call calcnMissWithKernel LIMIT times and therefore the total Complexety will be

O(k * LIMIT)


-------------------------------------------------------------------------
OMP

with omp I used only in one but very imporatnt part of my algorithm.

After Cuda will check and its first scnario that every point is its place, it will return an array
that will be in the size of  the number of points.
each cell in the array will tell if the point ws in her right place or not.

like we said above - if the case was 2 or 3 ponts, that not a problem
but when we have 500000 points, going over the array will be pretty long.

in addition, when we pass that array we need to sum the nMISS according to the results.

OMP will definitly will do the trick.
instaed of passing 500000 points one by one, we use OMP that will use the threads of the process and will
sum nMiss concurrently

Complexety = O(n / numOfThreads)

We call sumNmissLIMIT times and therefore the total Complexety will be

O((n / numOfThreads)* LIMIT) 

-----------------------------------------------------------------------------


Complexety of the program - O(Tmax/ (dt/numOfProc)) * ((O((n / numOfThreads)* LIMIT) + O(k * LIMIT)))














