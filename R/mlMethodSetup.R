#' This function uses the LireNLPSystem's rule-based functions to classify each report for the finding of interest
#' @param df input dataframe
#' @param imageid column indicating the id of each report, default set to "imageid"
#' @param seed indicates the seed, default 187
#' @keywords mlMethodSetup
#' @import caret
#' @export
#' @return returns a list of the MLMETHOD, the metric to select probability cutoff, MyControl which is needed for caret, and
#'         the train and test IDs
#' @examples
#' mlMethodSetup(df = outcome.df, imageid = "imageid", seed = 187)

mlMethodSetup = function(df,
                         imageid = "imageid",
                         seed = 187) {
  
  ### Set up ML parameters
  MLMETHOD = "glmnet" # elastic net logistic regression
  METRIC = "auc" # Can also use "f1" for optimization using F1-score
  myControl = trainControl(method = "cv", 
                           number = 10,
                           search = "random",
                           verboseIter = TRUE, # Change to FALSE if don't want output training log
                           returnData = TRUE,
                           returnResamp = "final",
                           savePredictions = "final",
                           classProbs = TRUE,
                           summaryFunction = aucSummary, # custom AUC loss function instead of default misclassification error
                           selectionFunction = "best",
                           preProcOptions = c("center", "scale"),
                           predictionBounds = rep(FALSE, 2),
                           seeds = NA,
                           trim = TRUE,
                           allowParallel = TRUE)
  
  ### Split to training and testing
  set.seed(seed) 
  n = nrow(df)
  testSample = sample(1:n, 0.2*n) # 20% held out for testing
  devSample = setdiff(1:n, testSample)
  trainID = df$imageid[devSample]
  testID = df$imageid[testSample]
  
  return(list(mlmethod = MLMETHOD,
              metric = METRIC,
              myControl = myControl,
              trainID = trainID,
              testID = testID))
}