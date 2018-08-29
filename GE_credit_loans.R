# Load all of the packages required for the analysis. Read in the data and convert the data to factors as necessary. Remname columns 7 and 17 because the original names break some algorithms.

library(rattle)
library(tibble)
library(randomForest)
library(UBL)
library(readxl)
library(magrittr)
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(corrplot)
library(Hmisc)
library(onehot)
library(caret)
Credit_Data <- read_excel("~/Credit Data.xls")

# Convert to factors where needed
Credit_Data$CHK_ACCT <- as.factor(Credit_Data$CHK_ACCT)
Credit_Data[,4:10] <- lapply(Credit_Data[,4:10], as.factor)
Credit_Data[,12:13] <- lapply(Credit_Data[,12:13], as.factor)
Credit_Data[,15:22] <- lapply(Credit_Data[,15:22], as.factor)
Credit_Data[,24:26] <- lapply(Credit_Data[,24:26], as.factor)
Credit_Data$JOB <- as.factor(Credit_Data$JOB)
Credit_Data[,30:32] <- lapply(Credit_Data[,30:32], as.factor)
creditdf <- column_to_rownames(Credit_Data, var = "OBS#")

# Rename to remove bad characters
colnames(creditdf)[7] <- "Radio_TV"
colnames(creditdf)[17] <- "Co_Applicant"
str(creditdf)

# Check the distributions of the data frame. Ensure no N/As are present. Verify no extreme values (i.e. 9999) we used in placed of N/A.

describe(creditdf)

# Split the data into training, validation, and test data sets.
set.seed(456)
samp.train <- sample(1000, 0.7 * 1000)
train <- creditdf[samp.train, ]
val <- creditdf[-samp.train,]
test <- creditdf[-samp.train,]
set.seed(456)
samp.val <- sample(300, 150)
val <- val[samp.val,]
test <- test[-samp.val,]

# Recreate pilot model for comparison to new models. Cutoff parameter was used to weight predictions.
set.seed(456)
pilot.model <- randomForest(DEFAULT ~ ., data = train, cutoff = c(.7, .3))
pilot.model

results.pilot.val <- predict(pilot.model, val)
table(val$DEFAULT, results.pilot.val)
results.pilot.test <- predict(pilot.model, test)
table(test$DEFAULT, results.pilot.test)

# Specificity for the validation set is 0.8049, and it is 0.8286 for the test data. This is good, as the goal is to minimalize the false negative rate and increase the true negative rate.

# Next, we try feature reduction methods. The first method is FAMD - Factor Analysis of Mixed Data. It is a dimension reduction technique for data that contains both numerical and categorical data.

results.famd <- FAMD(train, ncp = 45, sup.var = 31, graph = TRUE)

# Explore variances in FAMD dimensions
eig.val <- get_eigenvalue(results.famd)
eig.val

# Plot the percentage of variances by dimension
fviz_screeplot(results.famd, ncp = 45)

# Dimensions do not explain much variance on their own. At least 70% of the variance is explained at 23 dimensions, so we will train a model with this.

train.famd <- as.data.frame(results.famd$ind)
train.famd <- train.famd[,1:23]
train.famd$DEFAULT <- train$DEFAULT
set.seed(123)
famd.rf.model <- randomForest(DEFAULT ~ ., data = train.famd, cutoff = c(.7, .3))
famd.rf.model

# Apply tramsformations to validation and test data. Predict values and measure accuracy.
val.famd <- as.data.frame(predict(results.famd, newdata = val))
results.famd.rf.val <-predict(famd.rf.model, val.famd)
table(test$DEFAULT, results.famd.rf.val)

test.famd <- as.data.frame(predict(results.famd, newdata = test))
results.famd.rf.test <-predict(famd.rf.model, test.famd)
table(test$DEFAULT, results.famd.rf.test)

# Accuracy has not improved as a result of the FAMD model. Next, we try PCA with one-hot-encoding.

# One hot encode categorical variables
onehot.transformations <- onehot(creditdf)
onehot.creditdf <- as.data.frame(predict(onehot.transformations, creditdf))

# Split the one-hot-encoded data into train, val, and test sets
train.onehot <- onehot.creditdf[samp.train, ]
val.onehot <- onehot.creditdf[-samp.train, ]
test.onehot <- onehot.creditdf[-samp.train, ]

val.onehot <- val.onehot[samp.val, ]
test.onehot <- test.onehot[-samp.val, ]

# Remove DEFAULT variable from the model and perform PCA
pca <- prcomp(train.onehot[,-c(70,71)], scale. = TRUE)

# Variance
pca.var <- (pca$sdev)^2 

# % of variance
pca.var.per <- pca.var / sum(pca.var)

# Plot
plot(pca.var.per, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", type = "b" )

# Scree Plot
plot(cumsum(pca.var.per), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", type = "b" )

# Identify components that explain 70% or higher
cumsum(pca.var.per)

# Variacnce breaks 70% after the 21st variable. The new random forest model will be made using these components.

# Create new dataframe for the RF model with the orginal DEFAULT variable and PCA transformations
train.pca <- data.frame(DEFAULT = train$DEFAULT, pca$x[,1:21])

# Create model using new data frame
set.seed(123)
pca.rf.model <- randomForest(DEFAULT ~ ., data = train.pca, cutoff = c(.7, .3))
pca.rf.model

# Transform val into PCA
val.pca <- as.data.frame(predict(pca, newdata = val.onehot))
# Keep only first 21 components
val.pca <- val.pca[,1:21]

# Predict new test data, compare to original DEFAULT variables
results.pca.rf.val <-predict(pca.rf.model, val.pca)
table(test$DEFAULT, results.pca.rf.val)

# Transform test into PCA
test.pca <- as.data.frame(predict(pca, newdata = test.onehot))
# Keep only first 21 components
test.pca <- test.pca[,1:21]

# Predict new test data, compare to original DEFAULT variables
results.pca.rf.test <-predict(pca.rf.model, test.pca)
table(test$DEFAULT, results.pca.rf.test)

# The PCA results are slightly worse than the FAMD model. This is expected as PCA is intended for only numerical data, and transforming categorical data to numerical through one-hot-encoding is more of a workaround than a solution. Next, we will attempt two reduce features based on the Mean Decrease in Gini from the pilot model.

# Get variable importance from pilot model and resort by MeanDecreaseGini
pilot.var.imp <- as.data.frame(pilot.model$importance)
pilot.var.imp <- rownames_to_column(pilot.var.imp, "variable")

# Create new data sets using only the variables with a MeanDecreaseGini greater than 10
pilot.var.subset <- as.vector(pilot.var.imp[pilot.var.imp$MeanDecreaseGini >= 10,1])
train.mdg <- subset(train, select = c(pilot.var.subset, "DEFAULT"))
val.mdg <- subset(val, select = c(pilot.var.subset, "DEFAULT"))
test.mdg <- subset(test, select = c(pilot.var.subset, "DEFAULT"))

# Create model using new training data
set.seed(789)
mdg.rf.model <- randomForest(DEFAULT ~ ., data = train.mdg, cutoff = c(.7, .3))
mdg.rf.model

# Predict val data, test data, and measure accuracy
results.mdg.rf.val <- predict(mdg.rf.model, val.mdg)
table(val.mdg$DEFAULT, results.mdg.rf.val)

results.mdg.rf.test <- predict(mdg.rf.model, test.mdg)
table(test.mdg$DEFAULT, results.mdg.rf.test)

# These results are better than PCA and FAMD, but still not as good as the pilot model. The final attempt is to use recursive feature elimination.

# RFE using caret package
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
set.seed(123)
results.rfe <- rfe(creditdf[,1:30], y = creditdf$DEFAULT, sizes = c(1:30), rfeControl = control)
print(results.rfe)

# Create new data sets using only the predictors from the RFE algorithm
rfe.var.subset <- as.vector(predictors(results.rfe))
train.rfe <- subset(train, select = c(rfe.var.subset, "DEFAULT"))
val.rfe <- subset(val, select = c(rfe.var.subset, "DEFAULT"))
test.rfe <- subset(test, select = c(rfe.var.subset, "DEFAULT"))

# Create model using new RFE data
set.seed(456)
rfe.rf.model <- randomForest(DEFAULT ~ ., data = train.rfe, cutoff = c(.7, .3))
rfe.rf.model

# Predict val data, test data, and measure accuracy
results.rfe.rf.val <- predict(rfe.rf.model, val.rfe)
table(val.rfe$DEFAULT, results.rfe.rf.val)

results.rfe.rf.test <- predict(rfe.rf.model, test.rfe)
table(test.rfe$DEFAULT, results.rfe.rf.test)

# The RFE model has the most accurate predictions for the validation and test data. This will be the model submitted to GE.
