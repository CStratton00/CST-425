Objective: Examine the records of existing patients and use that information 
to predict whether a particular patient is likely to suffer from heart disease or not.

# Install the packages needed for this code file
# install.packages("e1071")

#----------------------------------------------------------------
#' #1. Load the Data
#----------------------------------------------------------------

# Load the data and use the `col_types` argument to specify type of 
# all columns('n' stands for numerical and 'f' stands for factor).


# Get a glimpse of the data.

#----------------------------------------------------------------
#' #2. Explore and Prepare the Data
#----------------------------------------------------------------
# Show a set of descriptive statistics for every variable in the data.


Note that there are some missing values in the dataset. Also, some of the numeric features have a wider range of values than others. 
How do you handle these missing values?
Do you have to normalize data to overcome the wide range in values?

Do a little research about how is naïve Bayes affected by these issues. 

# Using the sample() function, let's create our training and test datasets with a 75% to 25% split.
# The set.seed() function ensures to get the same result every time we run a random sampling process.


# Check the proportions for the class between all 3 sets.


#----------------------------------------------------------------
#' #3. Build the Model
#----------------------------------------------------------------


# Train a new model using the naiveBayes() function.


#----------------------------------------------------------------
#' #4. Evaluate the Model's Performance
#----------------------------------------------------------------
# Use the model to predict the class of the test instances.


# Create confusion matrix of our results.


# What is the accuracy of our prediction?

The results show that the predictive accuracy of our model is .... 
Is this good?
How important is prediction accuracy?
Peek at near future concepts: resampling and k-fold cross-validation

#----------------------------------------------------------------
#' #5. Interpret the results
#----------------------------------------------------------------

