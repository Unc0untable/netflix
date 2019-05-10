Recommender Systems Project

By Anthony Harris

5/10/2019

----------
Command Line Arguments And Program Operation
----------
**********************************************************************************************************
1.Make correct file executable: chmod +x Netflix_Probe_Project_2.py
2.Check the path location of the netflix dataset: write/type it down
3.check the path location of the probe dataset: write/type it down
4.In the directory of the most recent executable type: "vim Netflix_Probe_Project.py"
5.Within the program update the two paths accordingly(lines 46 & 48)
	-netflix path should be incomplete like so: Ex.  path = "/home/AnthonyHarris830/anaconda3/training_set/mv_"
	-probe path should be complete like so: Ex.  probe_path = "/home/AnthonyHarris830/anaconda3/probe.txt"

Arguments
python Netflix_Probe_Project_2 <int: first k movies> <boolean: examines entire probe file> <int: examines first j lines in probe file> <boolean:verbose output

argv[1]--> number_of_movies = 17770 (default)
argv[2]--> breaker = False (default)
argv[3]--> exit_threshold = 100 (default, this will not activate when breaker is off)
argv[4]--> verbose = False (default)

example_1: I want to examine the entire probe dataset that relates to the first 5 movies
arguments: python Netflix_Probe_Project_2 5 False 0 False
output: A text file called "Output_5_movies", containing relevant analytics


example_2: I want to examine the first 200 lines of the probe dataset that relates to the first 10 movies
arguements: python Netflix_Probe_Project_2 10 True 200 False
output: A text file called "Output_10_movies"

example_3: I want to examine all the lines of the probe dataset for every movie in the dataset
arguements python Netflix_Probe_Project_2 17700 False 0 False
output: A text file called "Output_17770_movies"

*Warning exmining the entire probe dataset is slow, if you want quicker feedback examine the first x lines of the probe dataset*

*******************************************************************************************************

-----------
Background:
----------

*********************************************************************************************************
This project was an attempt to replicate the results published by McSherry and Mironov about a decade ago
which related to differentially private recommender-systems. The paper itself was a relatively quick read.
As I began working on the project, my first objective was to get a better intution of recommender systsems.
I wanted to know what recommender systems are, as well as how and why do they work. Once I watched and
read enough information on the topic, I began investigating the paper with at least a better idea of the
topic at hand. Essentially, by replicating these results we would have a higher confidence that the 
dataset being analyzed is differentially private. Datasets with this property preserves data utility
for data mining, while mitigating "linking attacks" where an anonymity of the user is broken. Once this
type of dataset is aquired, then one could explore potential vulnerabilities of differential-privacy  


********************************************************************************************************
----------
Program Structure
----------

The program is a single class with over 30 attributes for each netflix object created. These attributes correspond
to a discrete state of on an array or value. For instance, when a subset of a netflix array is process it will be
stored as an attribute to be invoked at will. Overall, here is a basic flow scheme of the program

1.choose k amount of movies
2.read netflix file containng first k movies-rating columns  
3.construct netflix array with first k movies-rating columns
4.read in the probe file (you can read up to the jth line of text)
5.after reading the probe file, recover the user-moives pairs
6.given the user-movie pair, find the corresponding movie rating from the netflix array
7.after aquiring (user,movie,movie-rating), store that triple in a probe array
8.train the probe array based on the algortihms in the paper
9.training the data will eventually yield an equally-sized prediction array called "hat(r_ui)" 
10.use rsme(root square mean estimate) to determine the accuracy between the probe array and hat(r_ui) 


----------
Issues:
----------

**********************************************************************************************************
The objective of the project was not met, so this section discusses some of the challenges encountered.

1.) Inexperience

Prior to this project I did not deal with recommender systems or large datasets. The former was more significant
as the intuition differed from my expectations. Given the fact that I was reading a technical paper, the authors
assumed a high level of compentency of the subject from the reader's perspective. For example, I assumed that all
the values I saw on the Netflix dataset were ratings from the users. This turned out to be false, as the dataset
was incredibly sparse. In this situation, the array data structure is either filled with "missing values" or zeros,
which are normally interpreted as missing. Luckily, the operations involved in the paper does not require
multiplication between elements within the data structure, with other elements within the same data sture. Such an
operation would cascade into legitimate values being assign "zero" or potential divisions by zero. Besides that,
it took while to build a relatively complete intuition of what was going per operation. Because of this lag of intition
production debugging and program structure/development was slow.

2.) Computational Resources

For those that eventually dive into the code you will notice that the program is designed to deal with 
datasets of arbitary size. This is due to memory restriction, where my machine throws a memory error during execution.
.Because of this, I kept my dataset small, usually between 10-15 movies or so which represented between 10,000-20,000
users. Keep in mind that there are over 17,700 movies to take into consideration and at least 500,000 users to account
for. When you're dealing with these datasets be cautious with nested loops.

4.) Time Complexity

This program is incredibly slow. Based on experimentation, a heavy process involves parsing the probe file. But I am sure there are other bottlenecks

3.) Lack of specificity in the paper

One of the more important things that I realized later on, which was quite crucial, was that you need to invoke the
probe dataset. The porbe dataset is essentially a random subset of the raw data. The probe data is essentially
the benchmark of how well the algorithm you built works. Once you extrated the probe data in an array you will need
to make a copy of it so that you have a dataset to train your algorithms on. Once the "clone" dataset was trained by
the various algorithms, you compute an rsme on the trained cloned dataset and the original dataset
to assess the accuracy of the prediction. The thing was, everything that mentioned above was not mentioned at all in 
the paper. Also, the fact that normalization was one of the last algorithms the mentioned made it very debatable
on whether it is an optional feature or a requirement. Since the details on how they actually achieved their results 
are sparse, it was hard to find papers or resources that concisely commented on the objectives and goal of the paper.

*******************************************************************************************************

