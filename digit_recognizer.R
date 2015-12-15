install.packages("class")
library(class)

#########################
#DEFINE CUSTOM FUNCTIONS#
#########################

#Function to plot an inputted row
digit_plot <- function(row, digits){
  digit_mat <- matrix(data = 0, nrow = 28, ncol = 28) #Initialize pixel matrix
  mat_col = 1
  mat_row = 1
  for (col in 2:ncol(digits)){  #Column 2 is pixel 0
    
    digit_mat[mat_row, mat_col] <- digits[row, col]
    
    if ((col - 1) %% 28 == 0){
      mat_row = mat_row + 1
      mat_col = 1 #reset columns for new row
    }
    
    if (mat_col < 28) mat_col<- mat_col + 1
  }
  #Plot Image
  x <- 1:28
  y <- 1:28
  image(x, y, digit_mat, col = gray((0:255)/255))
}

#Function to plot principal components
PC_plot <- function(pc_input, pc_num){
  digit_mat <- pc_to_mat(pc_input, pc_num)
  #Plot Image
  x <- 1:28
  y <- 1:28
  image(x, y, digit_mat, col = gray((0:255)/255))
}

pc_to_mat <- function(pc_comp_struc, pc_num) {
  matrix <- matrix(0, ncol =28, nrow = 28)
  mat_col = 1
  mat_row = 1
  for(i in 1:length(pc_comp_struc$rotation[,pc_num])) {
    matrix[mat_row, mat_col] <- pc_comp_struc$rotation[i, pc_num]  
    if ((i) %% 28 == 0){
      mat_row = mat_row + 1
      mat_col = 1 #reset columns for new row
    }
    
    if (mat_col < 28) mat_col<- mat_col + 1
    i=i+1 
  }
  return(matrix)
}


############
#READ DATA##
############

digits_train <- read.csv("C:\\Users\\grazim\\Desktop\\Kaggle\\Digit Recognizer\\Train.csv")
digits_test <- read.csv("C:\\Users\\grazim\\Desktop\\Kaggle\\Digit Recognizer\\Test.csv")

############
#SCALE DATA#
############

digits <- digits_train
means <- list()
#scale the images
#Take the mean of each column and subtract the mean from the initial values (Set mean of each to 0)
 for(i in 2:ncol(digits)){
   means[i-1] = mean(digits[,i])
   digits[,i] = as.numeric(digits[,i]) - as.numeric(means[i-1])
   digits_test[,i-1] = as.numeric(digits_test[,i-1]) - as.numeric(means[i-1]) #digits_test does not have a number indicator
 }
digits[,2:ncol(digits)] = digits[,2:ncol(digits)]/255 #255 is max value.  Scale all values
digits_test = digits_test/255 #255 is max value.  Scale all values


##########
#ANALYSIS#
##########

#Plot Some Training data
digit_plot(5627, digits_train)

#Run PCA Analysis
digits_pc <- prcomp(digits_train[2:ncol(digits_train)])

#Determine the Percent of variation capture by each PC
variation <- (digits_pc$sdev)^2 / sum(digits_pc$sdev^2) 
sum(variation[1:30]*100) # % variation of 30 PC is greater than 70%, sum 87 is .9006116
plot(digits_pc, type = "l")
summary(digits_pc)
screeplot(digits_pc, npcs = 100)

#Plot the first 30 Principal Components
par(mfrow = c(1, 1))
for (j in 1:30) PC_plot(digits_pc,j)

#Keep the 30 top PC
PC_top <- digits_pc$rotation[,1:30]

#Multiply the Top PC to my original data to get the transformed data
digits_train_PC <- t(as.matrix(PC_top)) %*% t(as.matrix(digits_train[,2:length(digits_train)]))
#Transpose the data so features are columns again. 
digits_train_PC <- t(digits_train_PC)
#digits_train_PC <- cbind(digits_train[,1],digits_train_PC ) #Add back in the classifyer column from original data

#Apply PC to test data to run KNN
digits_test_PC <- t(as.matrix(PC_top)) %*% t(as.matrix(digits_test))
#Transpose
digits_test_PC <- t(digits_test_PC)

#Run KNN
KNN <- knn(digits_train_PC, digits_test_PC, digits_train[,1])

write.csv(KNN, file = "C:\\Users\\grazim\\Desktop\\Kaggle\\Digit Recognizer\\KNN_Results.csv")




#How does converting original data back look when plotted?
digit_train_approx <- t(as.matrix(PC_top) %*% t(as.matrix(digits_train_PC)))

#test <- function(digit_train_approx, means){
for(i in 1:ncol(digit_train_approx)){
  digit_train_approx[,i] = 255*(digit_train_approx[,i]) + as.numeric(unlist(means[i]))
}
#}



row = 21
digit_plot(row, digits_train)
digit_plot(row, digit_train_approx)









digit_plot_test <- function(row, digits){
  
  digit_mat <- matrix(data = 0, nrow = 28, ncol = 28) #Initialize pixel matrix
  mat_col = 1
  mat_row = 1
  for (col in 1:ncol(digits)){  #Column 2 is pixel 0
    
    digit_mat[mat_row, mat_col] <- digits[row, col]
    
    if ((col - 1) %% 28 == 0){
      mat_row = mat_row + 1
      mat_col = 1 #reset columns for new row
    }
    
    if (mat_col < 28) mat_col<- mat_col + 1
  }
  #Plot Image
  x <- 1:28
  y <- 1:28
  image(x, y, digit_mat, col = gray((0:255)/255))
}
