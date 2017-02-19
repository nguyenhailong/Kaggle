loglossSummary <- function(data, lev = NULL, model = NULL)  
{  
  #print(paste(class(data$pred))) # factor  
  data$pred <- as.numeric(data$pred)-1 # otherwise I get an error as these are factors  
  data$obs <- as.numeric(data$obs)-1 # otherwise I get an error as these are factors  
  epsilon      <- 1e-15  
  yhat           <- pmin(pmax(data$pred, rep(epsilon)), 1-rep(epsilon))  
  logloss      <- -mean(data$obs*log(yhat) + (1-data$obs)*log(1 - yhat))  
  names(logloss) <- "LOGLOSS"  
  logloss  
} 