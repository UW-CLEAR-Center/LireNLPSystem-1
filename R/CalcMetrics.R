#' calculate auc using trapezoid method.
#' @param test
#' @param truth
#' @param N
#' @param control
#' @param case
#' @keywords ML, machine-learning
#' @return a list
#' @examples
#' CalcMetrics(test, truth, N, control, case)
CalcMetrics = function(test,truth,N,control,case){
  ### Function to get error rates: prev, sens, spec, ppv, npv, fscore, auc of a prediction algorithm
  test <- ifelse(test==control,0,1)
  truth <- ifelse(truth==control,0,1)
  
  truth.pos <- which(truth==1)
  test.pos <- which(test==1)
  tp <- length(which(truth.pos %in% test.pos))
  fp <- length(test.pos) - tp
  fn <- length(truth.pos) - tp
  tn <- N - tp - fp - fn
  
  prev <- length(truth.pos)/N
  sens <- tp/(tp+fn)
  spec <- tn/(tn+fp)
  ppv <- tp/(tp+fp)
  npv <- tn/(tn+fn)
  fscore <- 2*sens*ppv/(sens+ppv)
  auc <- CalcTrapezoidalAUC(sens,spec)
  
  result <- c(prev=prev,sens=sens,spec=spec,ppv=ppv,npv=npv,fscore=fscore,auc=auc)
  return(result)
}