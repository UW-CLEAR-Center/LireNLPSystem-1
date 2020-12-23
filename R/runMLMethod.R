#' This function elastic net logistic regression for a given finding and feature matrix
#' @param finding string indicating finding of interest
#' @param featureMatrix feature matrix
#' @param outcome dataframe indicating the labels for each report
#' @param trainID vector indicating the imageid's for the training data
#' @param testID vector indicating the imageid's for the test data
#' @param metric string indicating which metric to maximize (eg. "auc" or "fscore")
#' @param mlmethod string indicating the ml method (eg. "glmnet")
#' @param myControl caret myControl hyperparameters
#' @param outpath string indicating outpath to write results to to
#' @keywords runMLMethod
#' @import caret
#' @import ROCR
#' @import PRROC
#' @import lattice
#' @export
#' @return a list of 3 dataframes: metrics (performance metrics for train and test), predictions (predictions on train and test), model's features and coefficients
#' @examples
#' runMLMethod(finding = "fracture",
#'             featureMatrix = featureMatrix,
#'             outcome = outcome.df,
#'             trainID = c(1,2,3),
#'             testID = c(4,5,6),
#'             metric = "auc",
#'             mlmethod = "glmnet",
#'             outpath = "Results/")

runMLMethod = function(finding,
                       featureMatrix,
                       outcome,
                       trainID,
                       testID,
                       metric,
                       mlmethod,
                       myControl,
                       outpath) {
  
  ### SET UP DATASETS
  
  #### Match imageid ordering of both feature matrix and outcome dataframe
  featureMatrix = featureMatrix[order(match(featureMatrix$imageid, outcome$imageid)), ]
  outcome = outcome[order(match(featureMatrix$imageid, outcome$imageid)), ]
  
  #### REMOVE REGEX AND NEGEX FOR NON-FINDINGS OF INTEREST
  colsRemove = names(featureMatrix)[which(grepl(paste("(?<!",finding,")_(r|n)egex", sep="")
                                                , names(featureMatrix), perl = TRUE))]
  
  ##### the only identifier is imageid
  if (length(colsRemove) > 0) {
    X = featureMatrix[ , -which(names(featureMatrix) %in% colsRemove)]
  } else {
    X = featureMatrix
  }
  Y = outcome %>%
    dplyr::select(c("imageid", "siteID", "imageTypeID", finding)) %>%
    dplyr::mutate(trueClass = factor(make.names(get(finding))))
  
  
  ##### caret function needs "X0" and "X1" as levels (corresponding to 0 and 1)
  CONTROL_LEVEL = levels(Y$trueClass)[1]
  CASE_LEVEL = levels(Y$trueClass)[2]
  
  ### PERFORM TRAINING
  trainModel = train(x = X %>%
                       dplyr::filter(imageid %in% trainID) %>%
                       dplyr::select(-imageid) %>%
                       as.matrix(.),
                     y = Y %>%
                       dplyr::filter(imageid %in% trainID) %>%
                       dplyr::select(trueClass) %>%
                       unlist(.),
                     method = mlmethod,
                     metric = ifelse(metric == "auc","ROC", metric), # Loss function to optimize on
                     maximize = TRUE,
                     trControl = myControl,
                     tuneLength = 10
  )
  print(trainModel)
  
  #### PREDICTION ON TRAINING DATASET
  trainOutput = data.frame(imageid = Y$imageid[trainModel$pred[,"rowIndex"]], 
                           siteID = Y$siteID[trainModel$pred[,"rowIndex"]],
                           imageTypeID = Y$imageTypeID[trainModel$pred[,"rowIndex"]],
                           trueClass = trainModel$pred[,"obs"],
                           predProb = trainModel$pred[,CASE_LEVEL],
                           fold = "train"
  )
  trainROC = prediction(predictions = trainOutput$predProb, 
                        labels = trainOutput$trueClass, 
                        label.ordering = c(CONTROL_LEVEL, CASE_LEVEL))
  
  #### PLOT ROC AND PRECISION-RECALL CURVES
  
  case = trainOutput$predProb[trainOutput$trueClass == "X1"]
  control = trainOutput$predProb[trainOutput$trueClass == "X0"]
  
  ##### ROC
  roc = roc.curve(scores.class0 = case, scores.class1 = control, curve = T)
  try(png(paste0(outpath,"/trainROC.png")))
  try(plot(roc))
  try(dev.off())
  
  ##### Precision-Recall
  #try(png(paste0(outpath, "/trainPrecisionRecall.png")))
  #pr = pr.curve(scores.class0 = case, scores.class1 = control, curve = T)
  #try(plot(pr))
  #try(dev.off())
  
  #### GET OPTIMAL CUT-OFF
  if(metric == "f"){ # For F1, find cut off that gives highest F1 score
    perfObject = performance(trainROC,
                             measure = metric,
                             x.measure = "cutoff")
    k_optimal = perfObject@"x.values"[[1]][which.max(perfObject@"y.values"[[1]])]
  }
  if(metric == "auc"){ # For AUC, find cut off that gives topleft corner of ROC curve
    perfObject = performance(trainROC,
                             measure = "tpr",
                             x.measure = "fpr")
    # false positive rate
    x = perfObject@"x.values"[[1]]
    # true positive rate
    y = perfObject@"y.values"[[1]]
    # identify probability cut that leads to the topleft corner of the ROC curve
    d = (x - 0)^2 + (y - 1)^2
    ind = which(d == min(d))
    k_optimal = perfObject@"alpha.values"[[1]][ind]
  }
  
  #### add the prediction based on the optimal cut, anything above the cut is case and anything
  trainOutput$predClass = ifelse(trainOutput$predProb >= k_optimal, 
                                 CASE_LEVEL, 
                                 CONTROL_LEVEL)
  
  ### PERFORM TESTING
  
  testOutput = data.frame(imageid = Y$imageid[which(Y$imageid %in% testID)],
                          siteID = Y$siteID[which(Y$imageid %in% testID)],
                          imageTypeID = Y$imageTypeID[which(Y$imageid %in% testID)],
                          trueClass = Y %>%
                            dplyr::filter(imageid %in% testID) %>%
                            dplyr::select(trueClass) %>%
                            unlist(.), 
                          predProb = predict(trainModel, 
                                             newdata = X %>%
                                               dplyr::filter(imageid %in% testID) %>%
                                               dplyr::select(-imageid) %>%
                                               as.matrix(.),
                                             type = "prob")[,levels(Y$trueClass)[2]],
                          fold = "test"
  )
  testOutput$predClass = ifelse(testOutput$predProb >= k_optimal, 
                                CASE_LEVEL, 
                                CONTROL_LEVEL)
  testROC = prediction(predictions=testOutput$predProb, 
                       labels=testOutput$trueClass, 
                       label.ordering = c(CONTROL_LEVEL,CASE_LEVEL))
  
  #### PLOT ROC AND PRECISION-RECALL CURVES
  
  case = testOutput$predProb[testOutput$trueClass == "X1"]
  control = testOutput$predProb[testOutput$trueClass == "X0"]
  
  ##### ROC
  roc = roc.curve(scores.class0 = case, scores.class1 = control, curve = T)
  try(png(paste0(outpath,"/testROC.png")))
  try(plot(roc))
  try(dev.off())
  
  ##### Precision-Recall
  #try(png(paste0(outpath, "/testPrecisionRecall.png")))
  #pr = pr.curve(scores.class0 = case, scores.class1 = control, curve = T)
  #try(plot(pr))
  #try(dev.off())
  
  ### EVALUATION METRICS
  trainMetric = CalcMetrics(test = trainOutput$predClass, 
                            truth = trainOutput$trueClass,
                            N = length(trainID),
                            control = CONTROL_LEVEL, 
                            case = CASE_LEVEL)
  testMetric = CalcMetrics(test=testOutput$predClass, 
                           truth=testOutput$trueClass, 
                           N=length(testID),
                           control=CONTROL_LEVEL,
                           case=CASE_LEVEL)
  
  metrics = rbind(train = c(trainMetric[-length(trainMetric)], auc=as.numeric(performance(trainROC,"auc")@y.values)),
                  test = c(testMetric[-length(testMetric)], auc=as.numeric(performance(testROC,"auc")@y.values))
  )
  
  metrics = as.data.frame(metrics)
  metrics$Finding = finding
  metrics$optim = metric # May be auc or f1
  metrics$N = c(length(trainID), length(testID))
  metrics$Partition <- rownames(metrics) # either "train" or "test" depending on which partition subject is in
  
  ### GET FEATURES
  
  ### model predictors/features
  myFeat = coef(trainModel$finalModel, trainModel$bestTune$lambda)
  myFeat = as.data.frame(as.matrix(myFeat))
  
  # the last row is the optimal cut identified from the training data
  myFeat = data.frame(predictors = c(row.names(myFeat),"cutoff"),
                      coef = c(myFeat[,1],k_optimal))
  myFeat = myFeat[myFeat$coef != 0,]
  
  myFeat$predictors = gsub("\\(|\\)","", myFeat$predictors, perl = TRUE) %>%
    gsub("IMPRESSION", "IMP", .)
  
  myFeat = myFeat %>% dplyr::arrange(desc(abs(coef)))
  myFeat = rbind(myFeat %>% dplyr::filter(predictors != "cutoff"), myFeat %>% dplyr::filter(predictors == "cutoff"))
  
  # output results
  write.csv(metrics, paste(outpath, "metrics.csv", sep = "/"), row.names = FALSE)
  write.csv(data.frame(rbind(trainOutput, testOutput)), paste(outpath, "prediction.csv", sep = "/"), row.names = FALSE)
  write.csv(myFeat, paste(outpath, "features.csv", sep = "/"), row.names = FALSE)
  
  return(list(metrics = metrics,
              predictions = data.frame(rbind(trainOutput, testOutput)), 
              features = myFeat))
}