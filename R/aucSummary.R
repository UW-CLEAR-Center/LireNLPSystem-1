#' custom AUC loss function instead of default misclassification error.
#' @param data
#' @param lev
#' @param model
#' @keywords ML, machine-learning
#' @export
#' @return A list
#' @examples
#' aucSummary(data)

aucSummary = function(data, 
                      lev = NULL, 
                      model = NULL) {
  # AUC loss function (top left corner of ROC curve)
  if (length(levels(data$obs)) > 2) 
    stop(paste("Your outcome has", length(levels(data$obs)), 
               "levels. The aucSummary() function isn't appropriate."))
  
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  
  aucObject <- try(performance(prediction(predictions = data[,lev[2]], 
                                          labels = data[,"obs"], 
                                          label.ordering=c(lev[1], lev[2])),
                               measure = "auc",
                               x.measure = "cutoff"), silent = TRUE)
  
  aucOpt <- if(class(aucObject)[1] == "try-error") NA
  else aucObject@"y.values"[[1]][which.max(aucObject@"y.values"[[1]])]
  
  out <- c(aucOpt, sensitivity(data = data[,"pred"], 
                               reference = data[,"obs"], 
                               positive = lev[2]), 
           specificity(data = data[,"pred"], 
                       reference = data[,"obs"], 
                       negative = lev[1]))
  names(out) <- c("ROC", "sens", "spec")
  return(out)
}