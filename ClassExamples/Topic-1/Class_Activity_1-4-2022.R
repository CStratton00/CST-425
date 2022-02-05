# Class Activity
# Collin Stratton
# Isac Artzi

# Create a Simple Notebook and Perform Initial Descriptive Analysis
# sources: https://r-lang.com/

# create the vector
x <- c(1, 5, 6, 3, 6, 8, 3, 1, 9, 2)

# mean
xmean <- mean(x)
print(xmean)

# mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

xmode <- getmode(x)
print(xmode)

# standard deviation
xstd <- sd(x)
print(xstd)

# interquartile range
xiqr <- IQR(x)
print(xiqr)

# scatter plot
y <- c(3, 6, 4, 3, 8, 5, 4, 2, 8, 3)
plot(x, y)