#!/usr/bin/env python
# coding: utf-8

# In[245]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pylab
from numpy import array
from numpy.linalg import norm
import time
import datetime
import os.path
import sys
#get_ipython().run_line_magic('matplotlib', 'inline')

#-------------------------------------------------------------------#
#------------------ Begin Readme -----------------------------------#
#-------------------------------------------------------------------#

#This code used my machines path to access the netflix files.
#Use your path accordingly
#In other words use the following template:
#Be sure to use double backslash when writing path
#path = "*path_of_netflix_files_here*\\mv_"

#-------------------------------------------------------------------#
#------------------ End Readme -------------------------------------#
#-------------------------------------------------------------------#

class Netflix_Dataset(object):

    #Within my local onyx/piret directory the netflix dataset path is...
    #... path = "/home/AnthonyHarris830/project/training_set/mv_"
    
    #Within my local onyx/piret directory, the probe dataset path is ...
    #... path = "/home/AnthonyHarris830/project/probe.txt"

    #object initialization
    def __init__(self,path ="/home/AnthonyHarris830/anaconda3/training_set/mv_" ,number_of_movies = 10):
        self.path = path
        self.probe_path = "/home/AnthonyHarris830/anaconda3/probe.txt"
        self.number_of_movies = number_of_movies
        privacy_budget = self.privacy_budget = 1
        self.noise_1 = .02 * privacy_budget
        self.noise_2 = .19 * privacy_budget
        self.noise_3 = .79 * privacy_budget
        self.alpha = 4
        self.k = 20
        self.beta = 10
        self.beta_m = 15
        self.beta_p = 20
        self.B = 1.0
        self.netflix_list = np.array([])
        self.netflix_array = np.array([])
        self.netflix_array_sum = 0
        self.user_amount = 0
        self.binary_array = np.array([])
        self.netflix_vertical_vector = np.array([])
        self.netflix_horizontal_vector = np.array([])
        self.binary_vertical_vector = np.array([])
        self.binary_horizontal_vector = np.array([])
        self.gsum = 0
        self.gcnt = 0
        self.g = 0
        self.msum = np.array([])
        self.mcnt = np.array([])
        self.mavg = np.array([])
        self.rui_hat = np.array([])
        self.r_bar =np.array([])
        self.r_hat = np.array([])
        self.cov = np.array([])
        self.w = np.array([])
        self.wgt = np.array([])
        self.cov_bar = np.array([])
        self.avg_wgt = 0
        self.avg_cov = 0
        self.score_array = np.array([])
        self.rsme = 0
        self.normalized_ru = np.array([])
        self.normalized_eu = np.array([])
        self.normalized_ru_hat = np.array([])
        self.probe_array = np.array([])
        self.user_vector = np.array([])
        self.concatenated_netflix_array = np.array([])
        self.probe_netflix_array = np.array([])
    
    #shows the attributes of the netflix object in varying degrees
    def info(self,verbosity = 5):
        
        if(self.netflix_list.shape[0] == 0):
            print("***Use the files_processed() method on object***")
            return
        
        if(self.user_amount == 0):
            print("***Use the user_count() method on object***")
            return
        
        if(verbosity < 2):
            print("Path: " + self.path)
        elif(verbosity < 3):
            print("Path: " + self.path)
            print("Number of movies: " + str(self.number_of_movies))
        elif(verbosity < 4):
            print("Path: " + self.path)
            print("Number of movies: " + str(self.number_of_movies))
            print(self.noise)
        elif(verbosity < 5):
            print("Path: " + self.path)
            print("Number of movies: " + str(self.number_of_movies))
            print("Noise: " + str(self.noise))
            print("Netflix list has dimension: " + str(self.netflix_list.shape))
        elif(verbosity < 6):
            print("Path: " + self.path)
            print("Number of movies: " + str(self.number_of_movies))
            print("Noise: " + str(self.noise))
            print("Netflix list has dimension: " + str(self.netflix_list.shape))
            print("Netflix array has dimension: " + str(self.netflix_array.shape))        
        elif(verbosity < 7):
            print("Path: " + self.path)
            print("Number of movies: " + str(self.number_of_movies))
            print("Noise: " + str(self.noise))
            print("Netflix list has dimension: " + str(self.netflix_list.shape))
            print("Netflix array has dimension: " + str(self.netflix_array.shape))
            print("Number of users: " + str(self.user_amount))
        else:
            print("Path: " + self.path)
            print("Number of movies: " + str(self.number_of_movies))
            print("Noise: " + str(self.noise))
            print("Netflix list has dimension: " + str(self.netflix_list.shape))
            print("Netflix array has dimension: " + str(self.netflix_array.shape))
            print("Netflix array sum: " +  str(np.sum(self.netflix_array)))
            print("Number of users: " + str(self.user_amount))
            print("Binary array dimensions: " + str(netflix.binary_array.shape))
            print("Sum of binary dimensions: " + str(np.sum(netflix.binary_array)))
            print("Dimension of netflix vertical vector: " + str(self.netflix_vertical_vector.shape))
            print("Dimension of netflix horizontal vector: " + str(self.netflix_horizontal_vector.shape))
            print("Dimension of binary vertical vector: " + str(self.binary_vertical_vector.shape))
            print("gsum has value: " + str(self.gsum))
            print("gcnt has value: " + str(self.gcnt))
            print("G has value: " + str(self.g))
            print("Dimension of msum: " + str(self.msum.shape))
            print("Dimension of mcnt: " + str(self.mcnt.shape))
            print("Dimension of mavg: " + str(self.mcnt.shape))
            print("Dimension of r_bar: " + str(self.r_bar.shape))
            print("Dimension of r_hat: " + str(self.r_hat.shape))
            print("Dimension of cov: " + str(self.cov.shape))
            print("Sum of cov elements: " + str(np.sum(self.cov)))
            print("Dimension of wgt: " + str(self.wgt.shape))
            print("Sum of wgt elements: " + str(np.sum(self.wgt)))
            print("Dimension of r_norm: " + str(self.r_norm.shape))
            print("Dimension of r_hat_norm: " + str(self.r_hat_norm.shape))
            print("Dimension of e_norm: " + str(self.e_norm.shape))
            print("Dimension of e_norm_2: " + str(self.e_norm_2.shape))
            print("Dimension of cov_bar: " + str(self.cov_bar.shape))
            print("Sum of cov_bar elements: " + str(np.sum(self.cov_bar)))
            #print("Dimension of wgt_bar: " + str(self.wgt_bar.shape))
            #print("Sum of wgt_bar elements: " + str(np.sum(self.wgt_bar)))         
            print("avg_wgt has value: " + str(self.avg_wgt))
            print("avg_cov has value: " + str(self.avg_cov))
            print("Dimension of score: " + str(self.score_array.shape))
            print("rsme has value " + str(self.rsme))
            
#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next five methods can be executed with the begin() method ------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    
        

    #Can process the first "n" many movie files. "n" was defined in the constructor        
    def files_processed_method(self):#1 to 17770
    
        number_of_files = self.number_of_movies
        array_list = []
        string_of_zeros = ""
        dataframe = ""
        array = ""
        filename = self.path
        filename_2 = self.path

        for i in range(number_of_files):
            
            i += 1

            if i < 10:
                string_of_zeros = "000000" #six zeros
            elif i < 100:
                string_of_zeros = "00000" #five zeros
            elif i < 1000:
                string_of_zeros = "0000" #four zeros
            elif i < 17700:
                string_of_zeros = "000" #three zeros

            filename = filename + str(string_of_zeros) + str(i) + ".txt"

            dataframe = "df_"+str(i)

            dataframe =  pd.read_csv(filename, index_col = False,skiprows = [0], header = None, names =["User_ID","Movie_Rating","Date_Of_Rating"])

            dataframe=dataframe.iloc[:,0:2] #slices out the "Date_Of_Rating" column

            dataframe=dataframe.sort_values(["User_ID"],ascending =[True])

            filename = filename_2

            array = "array_" + str(i)

            #converting datatframe into numpy array
            array = dataframe.values 

            array_list.append(array)

        array_list = np.array(array_list)

        self.netflix_list = array_list

        #each file should be converted into an array and stored in this list sequentially

    def user_count_method(self): 
        #new_list is a list of multiple arrays

        user_amount = 0

        new_list = self.netflix_list

        for array in new_list:

            array = np.asarray(array)

            #acquires the row dimension of each array and computes a running sum
            user_amount += array.shape[0] 

        self.user_amount = user_amount      
        
    #Populates the netflix array with user ratings
    #Each individual row uniquely coreespond to a user_id
    #Each individual column uniquely correspond to a rating for a specific movie 
    def create_netflix_array_method(self):
        
        array_list = self.netflix_list
        
        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        netflix_array = np.zeros((number_of_users,number_of_movies))
        
        original_user_id_array = np.zeros((1,number_of_users),int)
        
        original_user_id = 0
        
        id_counter = 0
        
        location_counter = 0
        
        ith_movie = 0
        
        #The goal of the double for-loop is to convert the original id's to standard id's and correctly assign user ratings
        #We store the original id's in an array before standardizing them
        #From there, we assign the rating of the user (w.r.t the ith movie) to the netflix array
        #If the same original id appears more than once we find its corresponding standardized id before assigning a rating (cont.)
        # ...to the ith movie in the netflix array
        for array in array_list:   
            
            for j in range(array.shape[0]):
                
                #This conditional checks if the original user id has been used before
                if array[j][0] not in original_user_id_array:
                    
                    #This conditional prevents out-of-bounds exceptions
                    #if id_counter < number_of_users:
                    if id_counter < number_of_users and j < array.shape[0]:
                        
                        original_user_id = array[j][0]
                        
                        original_user_id_array[0][id_counter] = original_user_id
                        
                        netflix_array[id_counter][ith_movie] = array[j][1]
                        
                        id_counter += 1
                #The user id has been used before
                else:
                    
                    #This conditional prevents out-of-bounds exceptions
                    if id_counter < number_of_users and j < array.shape[0]:
                        
                        #This conditional checks where the user id exists in the id array
                        while array[j][0] != original_user_id_array[0][location_counter]:
                            
                            location_counter +=1
                            
                        netflix_array[location_counter][ith_movie] = array[j][1]
                        
                        location_counter = 0

            ith_movie += 1
            
        netflix_array = np.array(netflix_array)
        
        self.netflix_array = netflix_array
        
        self.user_vector = original_user_id_array.T
        
        
    def remove_empty_rows_method(self):

        netflix_array = self.netflix_array 
        
        number_of_users = self.user_amount

        #if there exist a zero row vector within the netflix_array, we will remove it and update the array and number_of_users
        netflix_vertical_vector = np.sum(netflix_array, axis = 1)[:,None]

        counter = 0

        #removes all of the zero rows in the netflix_array
        #By design the netflix_array will almost always have leftover zero rows
        while 0 in netflix_vertical_vector[:,0]:
            
            if netflix_vertical_vector[counter][0] == 0:
                
                netflix_array = np.delete(netflix_array, (counter), axis=0)
                
                netflix_vertical_vector = np.sum(netflix_array, axis = 1)[:,None]
                
                counter -=3
                
            counter += 1
            
            if counter < 0:
                
                counter = 0
        
        self.netflix_array = np.array(netflix_array)
        
        self.user_amount = netflix_array.shape[0]

    def remove_empty_rows_method_2(self,array):

        number_of_users = array.shape[0]

        #if there exist a zero row vector within the netflix_array, we will remove it and update the array and number_of_users
        netflix_vertical_vector = np.sum(array, axis = 1)[:,None]
        
        counter = 0

        #removes all of the zero rows in the netflix_array
        #By design the netflix_array will almost always have leftover zero rows
        while 0 in netflix_vertical_vector[:,0]:
            
            if netflix_vertical_vector[counter][0] == 0:
                
                array = np.delete(array, (counter), axis=0)
                
                netflix_vertical_vector = np.sum(array, axis = 1)[:,None]
                
                counter -=3
                
            counter += 1
            
            if counter < 0:
                
                counter = 0

        array = np.array(array)

        return array       
        
    def user_vector_method(self):
        
        self.user_vector = self.remove_empty_rows_method_2(self.user_vector)

    def concatenate_netflix_array_method(self):
        
        self.concatenated_netflix_array = np.concatenate((self.user_vector,self.netflix_array),axis = 1)
    
    def binary_array_method(self):

        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        adjusted_netflix_array = self.netflix_array
        
        #hopefully this doesnt break the overall code
        self.user_amount = adjusted_netflix_array.shape[0]
        
        self.number_of_movies = adjusted_netflix_array.shape[1]
        
        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        #An array that indicates if a user rated a movie or not, using a binary scheme.
        # 1 indicates a user rated the specified movie, 0 indicates that the user did not rate the movie
        binary_rating_array = np.zeros((number_of_users,number_of_movies))

        #todo use numpy methods to optimize algotrithm
        #populates number_of_ratings and binary_rating_array
        for i in range(number_of_users):
            
            for j in range(number_of_movies):
                
                if adjusted_netflix_array[i][j] > 0:
                    
                    binary_rating_array[i][j]=1   

        self.binary_array = np.array(binary_rating_array)
        
    def begin(self):
        
        self.files_processed_method()
        
        self.user_count_method()
        
        self.create_netflix_array_method()
        
        self.remove_empty_rows_method()
        
        self.binary_array_method()
        
        self.user_vector_method()
        
        self.concatenate_netflix_array_method()
           
    def begin_2(self,number_of_movies,privacy_budget):
        
        self.number_of_movies = number_of_movies
        
        self.privacy_budget = privacy_budget
        
        self.noise_1 = .02 * privacy_budget
        
        self.noise_2 = .19 * privacy_budget
        
        self.noise_3 = .79 * privacy_budget
        
        self.begin()
        
    def begin_probe_dataset_computation(self,number_of_movies = 17770,privacy_budget = 1):
        
        self.begin_2(self.number_of_movies,self.privacy_budget)
        
        self.probe_dataset_method(breaker = True, exit_threshold = 100, verbose = False)
        
        self.remove_empty_rows_method()
        
        self.user_amount = self.netflix_array.shape[0]
        
        self.user_count_method()
        
        self.binary_array_method()
        
#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- End begin() method -------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    
        
        
#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next four methods can be executed with the shortcut_1() method--------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    
        
        #ux1 column vector
    def binary_vertical_vector_method(self):
        
        binary_array = self.binary_array
        
        binary_vertical_vector = np.sum(binary_array,axis = 1)[np.newaxis].T
        
        self.binary_vertical_vector = binary_vertical_vector
        
        
        #1xi row vector
    def binary_horizontal_vector_method(self):

        binary_array = self.binary_array

        binary_horizontal_vector = np.sum(binary_array,axis = 0)[np.newaxis]

        self.binary_horizontal_vector = binary_horizontal_vector


        #ux1 column vector
        #vector that tracks the total ratings designated by  a user for some specific movie
        #A dx1 vector
        #Each row corresponds to a specific user
    def netflix_vertical_vector_method(self):
        
        adjusted_netflix_array = self.netflix_array

        vertical_vector = np.sum(adjusted_netflix_array, axis = 1)[np.newaxis].T
        
        self.netflix_vertical_vector =  vertical_vector

        #1xi row vector
        #vector that tracks the total ratings designated by all users for some specific movie
        #A 1xd vector, information about how each user rated each individual movie is lost
        #Each column corresponds to a specific movie
    def netflix_horizontal_vector_method(self):
        
        adjusted_netflix_array = self.netflix_array

        horizontal_vector = np.sum(adjusted_netflix_array, axis = 0)[np.newaxis]
        
        self.netflix_horizontal_vector =  horizontal_vector   
           
    def normalize_method_1(self):
        
        number_of_users = self.user_amount
        
        #number_of_movies = self.number_of_movies
        
        adjusted_netflix_array = self.netflix_array
        
        binary_netflix_array = self.binary_array
        
        #warning this adjust netflix_array unexpectedly
        for i in range(number_of_users):
        
            L2_norm_binary = norm(binary_netflix_array[i,:])
            
            adjusted_netflix_array[i,:] = np.divide(adjusted_netflix_array[i,:],L2_norm_binary)
            
            binary_netflix_array[i,:] = np.divide(binary_netflix_array[i,:],L2_norm_binary)           

        self.normalized_ru = adjusted_netflix_array         
        
        self.normalized_eu = binary_netflix_array
        
    def w_vector_method(self):
        
        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        binary_netflix_array = self.binary_array
        
        w_vector = np.zeros((number_of_users,1))
        
        for i in range(number_of_users):
            
            L2_norm_binary = norm(binary_netflix_array[i,:])
            
            inverse_value = np.power(L2_norm_binary,-1)
            
            w_vector[i][0] = inverse_value
        
        self.w = w_vector          
        
    def shortcut_1(self):
        
        self.binary_vertical_vector_method()
        
        self.binary_horizontal_vector_method()
        
        self.netflix_vertical_vector_method()
        
        self.netflix_horizontal_vector_method()
        
        self.w_vector_method()
        
    def normalized_shortcut_1(self):
        
        self.normalize_method_1()
        
        self.binary_vertical_vector_method()
        
        self.binary_horizontal_vector_method()
        
        self.netflix_vertical_vector_method()
        
        self.netflix_horizontal_vector_method()
        
        self.w_vector_method()

#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------End shortcut_1()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    


#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next three methods can be executed with the shortcut_2() method ------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   

    def gsum_method(self):
        
        netflix_array = self.netflix_array
        
        gsum = np.sum(netflix_array)
        
        noise_1 = self.noise_1
        
        self.gsum = gsum + noise_1
        
    def gcnt_method(self):
        
        binary_array = self.binary_array
        
        gcnt = np.sum(binary_array)
        
        noise_1 = self.noise_1
        
        self.gcnt = gcnt + noise_1
        
    def g_method(self):
        
        self.gsum_method()
        
        self.gcnt_method()
        
        self.g = self.gsum /self.gcnt
        
    def shortcut_2(self):
        
        self.g_method()

#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------End shortcut_2()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    

#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next three methods can be executed with the shortcut_3() method-------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   

#the method shortcut_1() must be implemented prior to these methods

    def msum_method(self):
        
        #self.netflix_horizontal_vector should have been populated through the shortcut_1() method
        #1xi matrix, where i correspond to the amount of columns for each item
        horizontal_vector = self.netflix_horizontal_vector
        
        msum = horizontal_vector
        
        noise_2 = self.noise_2
        
        msum = msum + noise_2
        
        self.msum = msum
        
    def mcnt_method(self):
        
        #self.netflix_horizontal_vector should have been populated through the shortcut_1()
        #1xi matrix, where i correspond to the amount of columns for each item
        horizontal_vector = self.binary_horizontal_vector
        
        mcnt = horizontal_vector
        
        noise_2 = self.noise_2
        
        mcnt = mcnt + noise_2
        
        self.mcnt = mcnt
        
    def mavg_method(self):
        
        mavg = (self.msum + (self.beta_m * self.g))/(self.mcnt +(self.beta_m))
        
        self.mavg = mavg
        
    def shortcut_3(self):
        
        self.msum_method()
        
        self.mcnt_method()
        
        self.mavg_method()
        

#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------End shortcut_3()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    


#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next few methods can be executed with the shortcut_4()method ---------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   

    def r_bar_method(self):
        
        sum_mavg = np.sum(self.mavg)
        
        beta_p = self.beta_p
        
        g = self.g
        
        number_of_users = self.user_amount
        
        netflix_vertical_vector = self.netflix_vertical_vector
        
        binary_vertical_vector = self.binary_vertical_vector

        numerator = (netflix_vertical_vector - sum_mavg) + (beta_p * g)
        
        denominator = binary_vertical_vector + beta_p

        r_bar = numerator / denominator
        
        #if you dont reshape you get an akward tensor array
        r_bar = np.reshape(r_bar,(number_of_users,1))

        self.r_bar = r_bar
    
    def r_hat_method(self):
     
        #r_bar_method() need to be implement before calling r_hat_method()
    
        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        r_bar = self.r_bar
        
        adjusted_netflix_array = self.netflix_array
        
        B = self.B

        rui_hat = np.zeros((number_of_users,number_of_movies))

        for i in range(number_of_users):
            
            for j in range(number_of_movies):
                
                difference = adjusted_netflix_array[i][j] - r_bar[i][0]
                
                if difference < -B:
                    
                    rui_hat[i][j] = -B
                    
                elif difference >= -B and difference < B:
                    
                    rui_hat[i][j] = difference
                    
                else:
                    rui_hat[i][j] = B
                    
        self.r_hat = rui_hat
        
    def normalize_method_2(self):
        
        number_of_users = self.user_amount
        
        #number_of_movies = self.number_of_movies
        
        binary_netflix_array = self.binary_array
        
        r_hat_array = self.r_hat
        
        for i in range(number_of_users):
            
            L2_norm_binary = norm(binary_netflix_array[i,:])
                
            r_hat_array[i,:] = np.divide(r_hat_array[i,:],L2_norm_binary)
        
        self.normalized_ru_hat = r_hat_array 
    
    def shortcut_4(self):
        
        self.r_bar_method()
        
        self.r_hat_method()
        
    def normalized_shortcut_4(self):
        
        self.r_bar_method()
        
        self.r_hat_method()
        
        self.normalize_method_2()

#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------End shortcut_4()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    


#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next few methods can be executed with the shortcut_5()method ---------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   


    def cov_array_method(self):
    
        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        noise_3 = self.noise_3
        
        rui_hat = self.r_hat

        noise_array = np.zeros([number_of_movies,number_of_movies], dtype = float)
        
        noise_array = noise_array + noise_3

        w_vector = self.w

        cov =  np.zeros([number_of_movies,number_of_movies], dtype = float)

        for i in range(number_of_users):
            
            row = np.reshape(rui_hat[i,:],(1,number_of_movies))
            
            column = np.reshape(rui_hat[i,:].T,(number_of_movies,1))
            
            cov = cov + w_vector[i][0]*np.multiply(column,row)

        cov = cov + noise_array

        self.cov = cov

    def wgt_array_method(self):

        #number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        noise_3 = self.noise_3
        
        rui_hat = self.r_hat
        
        binary_array = self.binary_array
        
        netflix_vertical_vector = self.netflix_vertical_vector

        noise_array = np.zeros([number_of_movies,number_of_movies], dtype = float)
        
        noise_array = noise_array + noise_3
        
        w_vector = self.w

        wgt =  np.zeros([number_of_movies,number_of_movies], dtype = float)

        for i in range(number_of_users):
            
            row = np.reshape(binary_array[i,:],(1,number_of_movies))
            
            column = np.reshape(binary_array[i,:].T,(number_of_movies,1))
            
            wgt = wgt + w_vector[i][0]*np.multiply(column,row)

        wgt = wgt + noise_array

        self.wgt = wgt

    def shortcut_5(self):
        
        self.cov_array_method()
        
        self.wgt_array_method()


#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------End shortcut_5()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    



#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- The next few methods can be executed with the shortcut_6()method ---------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   


    def cov_bar_method(self):
        
        #number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        cov = self.cov
        
        wgt = self.wgt
        
        beta = self.beta
        
        avg_cov = np.sum(cov) / number_of_movies
        
        avg_wgt = np.sum(wgt) / number_of_movies
        
        self.avg_cov = avg_cov
        
        self.avg_wgt = avg_wgt
        
        numerator = cov + (beta * avg_cov)
        
        denominator = wgt + (beta * avg_wgt)
        
        cov_bar = np.divide(numerator,denominator)

        self.cov_bar = cov_bar

    #code is pending
    def score_method(self):

        number_of_users = self.user_amount
        
        number_of_movies = self.number_of_movies
        
        cov = self.cov
        vertical_cov = np.sum(cov,axis = 1)[np.newaxis].T
        
        netflix_array = self.netflix_array

        ri_bar = mavg = self.mavg

        subtraction_array = np.zeros([number_of_users,number_of_movies])
        
        abs_cov = np.absolute(cov)
        
        inverse_abs_cov = np.power(abs_cov,-1)
        
        vertical_inverse_abs_cov = np.sum(inverse_abs_cov,axis=1)

        for i in range(number_of_users):
            
            for j in range(number_of_movies):
                
                subtraction_array[i][j] = netflix_array[i][j] - mavg[0][j]
        
        numerator = np.multiply(vertical_cov.T,subtraction_array) 
        
        right_array = np.multiply(vertical_inverse_abs_cov,numerator)
    
        score_array = np.add(mavg,right_array)
      
        self.score_array = score_array
            
    def shortcut_6(self):

        self.cov_bar_method()
        
        self.score_method()
#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------End shortcut_6()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    


#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- ----------------------------super_shortcut--------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   
    def super_shortcut(self, number_of_movies = 10, privacy_budget = 1):
        
        self.number_of_movies = number_of_movies
        
        self.privacy_budget = privacy_budget
        
        self.begin_2(self.number_of_movies,self.privacy_budget)
        
        self.shortcut_1()
        
        self.shortcut_2()
        
        self.shortcut_3()
        
        self.shortcut_4()
        
        self.shortcut_5()
        
        self.shortcut_6()
    
    #this method must be computed after super_shortcut in order to work properly
    def super_shortcut_probe_dataset_computation(self,number_of_movies = 17770, privacy_budget = 1):
        self.number_of_movies = number_of_movies
        
        self.privacy_budget = privacy_budget     
        
        self.begin_probe_dataset_computation(self.number_of_movies,self.privacy_budget)
        
        self.shortcut_1()
        
        self.shortcut_2()
        
        self.shortcut_3()
        
        self.shortcut_4()
        
        self.shortcut_5()
        
        self.shortcut_6()
        

#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------supershortcut()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    

    def root_square_mean(self,array_1,array_2): #array_1 and array_2 are both same dimension

        array_1 = np.array(array_1)
                           
        array_2 = np.array(array_2)

        dimension = array_1.shape[0] * array_1.shape[1]

        difference_array = np.subtract(array_1,array_2)

        square_array = np.power(difference_array,2)

        output = np.sum(square_array)

        output = (output + 0.0) / (dimension)

        output = np.power(output,0.5)

        self.rsme = output
        

#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------------- ----------------------------ultra_shortcut--------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#   
    #This method does not use the probe dataset 
    def final_shortcut(self,number_of_movies,privacy_budget):
        
        self.begin_2(number_of_movies,privacy_budget)
                           
        self.shortcut_1()
                           
        self.shortcut_2()
                           
        self.shortcut_3()
                           
        self.shortcut_4()
                           
        self.shortcut_5()
                           
        self.shortcut_6()
                           
        self.root_square_mean_2(self.netflix_array,self.score_array)
    
    #This method does not use the probe dataset 
    def normalized_final_shortcut(self,number_of_movies,privacy_budget):

        self.begin_2(number_of_movies,privacy_budget)
                           
        self.normalized_shortcut_1()
                           
        self.shortcut_2()
                           
        self.shortcut_3()
                           
        self.normalized_shortcut_4()
                           
        self.shortcut_5()
                           
        self.shortcut_6()
                           
        self.root_square_mean_2(self.netflix_array,self.score_array)
        
    def final_shortcut_probe_dataset_computation(self, number_of_movies =17770 , privacy_budget = 1, breaker = False, exit_threshold = 100, verbose = False):
        
        #--------------------Initialize Original Netflix Dataset ---------------------------------------------#
        self.begin_2(number_of_movies,privacy_budget)
        
        #----Now that the Netflix Dataset is constructed, we construct the Probe Dataset ---------------------#
        #----Probe Dataset was constructed, which depends on the original dataset ----------------------------#
        self.probe_dataset_method(breaker = breaker, exit_threshold = exit_threshold, verbose = verbose)
        #-----------------------------------------------------------------------------------------------------#
        
        probe_netflix_array = self.probe_netflix_array
        
        self.netflix_array = self.probe_netflix_array
        
        self.user_count_method()
        
        self.remove_empty_rows_method()
        
        self.binary_array_method()
        
        probe_netflix_array = self.remove_empty_rows_method_2(probe_netflix_array)
          
        #----Shorcut_1: binary_vertical, binary_horizontal, netflix_vertical, netflix_horizontal, w_vector----#   
        self.shortcut_1()
                          
        #----Shorcut_2: G_method------------------------------------------------------------------------------#
        self.shortcut_2()
                          
        #----Shortcut_3: Msum, Mcnt, Mavg methods-------------------------------------------------------------#    
        self.shortcut_3()
                         
        #----Shortcut_4: r_bar, r_hat methods-----------------------------------------------------------------#
        self.shortcut_4()
            
        #----Shortcut_5: cov, wgt methods --------------------------------------------------------------------#
        #self.shortcut_5()
                           
        #----Shortcut_6: rsme methods ------------------------------------------------------------------------#    
        #self.shortcut_6()
        
        #-----------------------------------------------------------------------------------------------------#
        
        self.root_square_mean(self.r_hat,probe_netflix_array)      

#-------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------ultra_shortcut()-------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------#    

    def probe_dataset_method(self, breaker = False, exit_threshold = 250, verbose = False):
        
        #the netflix_array should have already been populated before using this method
        
        number_of_rows = self.netflix_array.shape[0]
        
        number_of_columns = self.netflix_array.shape[1]
        
        probe_netflix_array = np.zeros([number_of_rows,number_of_columns], dtype = float)
        
        missing_user_list_set = set([])
                           
        user_list_set = set([])
                           
        missing_movie_list_set = set([])
                           
        movie_list_set = set([])
        
        user_vector = self.user_vector
        
        concatented_netflix_array = self.concatenated_netflix_array
        
        movie_index = 0
        
        user_id = 0
        
        user_index = 0
        
        user_counter = 0
                           
        missed_user_counter = 0
        
        movie_counter = 0
                           
        missed_movie_counter = 0
                           
        exit_counter = 0
        
        #probe_txt = open("probe.txt", "r")
        probe_txt = open(self.probe_path,"r")

        for line in probe_txt:
            
            if breaker == True:
                
                exit_counter += 1
            
            if exit_counter > exit_threshold:
                
                break
                
            
            value = line.split()
            
            if ":" in value[0]:
                
                movie_index = int(value[0].replace(":",""))
                
            else:
                
                user_id = int(value[0])
                
                #if user_index_checker is zero, then the user in question is not in our sample dataest
                #Without this component, using a sample dataset will cause exceptions
                #If we used the complete dataset then we should be fine
                user_index_checker = np.array(np.where(user_vector == user_id)).size
                
                #helps test toy datasets
                if user_index_checker != 0:
                    
                    user_counter += 1
                    
                    user_index =  np.where(user_vector == user_id)[0][0]
                    
                    user_list_set.add(user_id)
                    
                else:
                
                    missing_user_list_set.add(user_id)
                    
                    missed_user_counter += 1
                
                #helps test toy datasets
                #This verifies that the movie in question is actually in our sample dataset
                #For the complete dataset there should be no problems
                if movie_index > self.number_of_movies:
                    
                    missed_movie_counter += 1
                    
                    missing_movie_list_set.add(movie_index)
                    
                    continue
                    
                else:
                    
                    movie_counter += 1
                    
                    movie_list_set.add(movie_index)
                
                probe_netflix_array[user_index][movie_index - 1] = self.concatenated_netflix_array[user_index][movie_index]
        
        probe_txt.close()
        
        self.probe_netflix_array = probe_netflix_array

        if verbose == True:
            print("Missing user list: ")
            print(missing_user_list_set)
            print("Total missing user")
            print(len(missing_user_list_set))
            print()
            print("Missing movie list: ")
            print(missing_movie_list_set)
            print("Missing movie size: ")
            print(len(missing_movie_list_set))
            print()
            print("probe netflix array: ")
            print()
            print(self.probe_netflix_array)
            print("Probe netflix array shape: \n")
            print(self.probe_netflix_array.shape)
            print("sum of the probe array elements: ")
            print(np.sum(self.probe_netflix_array))
            print()
            print("Total amount of movies evaluated: ")
            print(missed_movie_counter + movie_counter)
            print("Missed movie counter: ")
            print(missed_movie_counter)
            print("Movie counter: ")
            print(movie_counter)
            print("Total amount of users evaluated: ")
            print(missed_user_counter + user_counter)
            print("Missed user counter: ")
            print(missed_user_counter)
            print("User counter: ")
            print(user_counter)
            
            
        


# In[246]:


netflix = Netflix_Dataset()


# In[247]:


privacy_list = []

rsme_list = []

date_list = []

today = datetime.datetime.now()

date_list.append(today)

start = time.time()

entered_amount_of_movies = 0

entered_boolean_1 = True

entered_iteration_amount = 0

entered_boolean_2 = True

print("First argument:: " + sys.argv[1] + "\n")
print("Second argument: " + sys.argv[2] + "\n")
print("Third argument: " + sys.argv[3] + "\n")
print("Fourth argument: " + sys.argv[4] + "\n")

if int(sys.argv[1]) > 0:

	entered_amount_of_movies = int(sys.argv[1])
else: 
	entered_amount_of_movies = 10

if type(str(sys.argv[2])) ==  str:

	if sys.argv[2] == "True":

		entered_boolean_1 = True


	elif sys.argv[2] == "False":

		entered_boolean_1 = False

	else:

		entered_boolean_1 = -1
	

else:

	entered_boolean_1 = True
	
if int(sys.argv[3]) > 0:

	entered_iteration_amount = int(sys.argv[3])

else:
	entered_iteration_amount = 100

if type(str(sys.argv[4])) == str:

	entered_boolean_2 = str(sys.argv[4])

else:

	entered_boolean_2 = False

	
#for host machine netflix_path = "<insert_path_here>\\probe.txt"
#probe_path = "C:\\Users\\DareDevil\\Documents\\download\\probe.txt"

for i in range(5):
    
    privacy = i * 0.25
    
    privacy_list.append(privacy)
    
    netflix.final_shortcut_probe_dataset_computation(entered_amount_of_movies,privacy,entered_boolean_1,entered_iteration_amount,entered_boolean_2)
    
    rsme = netflix.rsme
    
    rsme_list.append(rsme)
    
end = time.time()

elapsed_time = end - start

Output = "Output"

counter = 0


Output = Output + "_" +  str(netflix.probe_netflix_array.shape[1]) + "_movies"

if os.path.isfile(Output) == False:

	write_text_file = open(Output,"w")

	write_text_file.write("Execute at: " + str(date_list[0]) + "\n")

	write_text_file.write("Progrma's time to execute: " + str(elapsed_time) + "\n")

	write_text_file.write("This averages: " + str(elapsed_time / 5.0) + " seconds per iteration \n")

	write_text_file.write("The number of users: " + str(netflix.probe_netflix_array.shape[0]) + "\n")

	write_text_file.write("The number of movies: " + str(netflix.probe_netflix_array.shape[1]) + "\n")

	write_text_file.write("Number of elements in the probe array: " + str(netflix.probe_netflix_array.shape[0] * netflix.probe_netflix_array.shape[1]) + "\n")

	write_text_file.write("Privacy List: " + str(privacy_list) + " \n")

	write_text_file.write("RSME List: " + str(rsme_list) + " \n")

	write_text_file.close()

print()
print("Executed at: " + str(date_list[0]))
print()
print("Program's time to execute: " + str(elapsed_time) + " seconds" )
print()
print("This averages: " + str(elapsed_time / 5.0) + " seconds per iteration")
print()
print("The number of users: " + str(netflix.probe_netflix_array.shape[0]))
print()
print("The number of movies: " + str(netflix.probe_netflix_array.shape[1]))
print()
print("Number of elements in the probe array: " + str(netflix.probe_netflix_array.shape[0] * netflix.probe_netflix_array.shape[1]))
print()
print("Privacy List: \n")
print(privacy_list)
print()
print("RSME List: \n")
print(rsme_list)
print()

#X = privacy_list
#Y = rsme_list
#scatter plot
#plt.scatter(X, Y, s=110, c='red', marker='^')

#add title
#plt.title('Global Effects Graph For RSME')

#add x and y labels
#plt.xlabel('Privacy budget: Theta')
#plt.ylabel('RSME')

#show plot
#plt.show()


# In[ ]:




