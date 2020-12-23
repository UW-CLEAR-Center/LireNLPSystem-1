#' calculate auc using trapezoid method.
#' @param sens
#' @param spec
#' @keywords ML, machine-learning
#' @return value that represents the AUC
#' @examples
#' CalcTrapezoidalAUC(sens, spec)
CalcTrapezoidalAUC <- function(sens,spec){
  ### Function to calculate AUC of a binary variable, using trapezoidal rule
  return(1/2*(1-spec)*(sens) + (spec)*(sens+1)/2)
}