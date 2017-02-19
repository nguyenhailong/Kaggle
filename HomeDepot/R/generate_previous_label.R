temp1 = unique(log_feature$id)
# temp2 = unique(event_type$id)
# temp3 = unique(resource_type$id)
# temp4 = unique(severity_type$id)
# sum(temp1-temp2)
# sum(temp1-temp3)
# sum(temp1-temp4)
# find previous label
loc_time_order = c()
loc_time_order$id = unique(log_feature$id)
test$fault_severity = -1
temp = rbind(train,test)
loc_time_order = merge(loc_time_order, temp,'id', sort = FALSE)
sum(temp1 - loc_time_order$id)
#Find previous label
## immdediate previous label
loc_time_order$preLbl = c(-1,loc_time_order$fault_severity[1:18551])

## find the closest label (not-1) with same location
previousLabel = -1
for(i in 2:nrow(loc_time_order)){
  j = i-1
  while(j>1 & loc_time_order$fault_severity[j] == -1){
    j = j-1
  }
  previousLabel = c(previousLabel,loc_time_order$fault_severity[j])
}
loc_time_order$preLblNotEmpt = previousLabel

## find the closest label (not-1) with same location
previousLabel = -1
for(i in 2:nrow(loc_time_order)){
  j = i-1
  while(j>1 & loc_time_order$fault_severity[j] == -1){
    j = j-1
  }
  if(loc_time_order$location[i] == loc_time_order$location[j]){
    previousLabel = c(previousLabel,loc_time_order$fault_severity[j])
  }else{
    previousLabel = c(previousLabel,-1)
  }
}
loc_time_order$preLblSameLoc = previousLabel

# Given an ID, find previous ID base on the above order
previousLabelTrain = merge(train, loc_time_order,'id', sort = FALSE)
length(which(previousLabelTrain$fault_severity.y == previousLabelTrain$preLblSameLoc))
length(which(previousLabelTrain$fault_severity.y == previousLabelTrain$preLblNotEmpt))
length(which(previousLabelTrain$fault_severity.y == previousLabelTrain$preLbl))


#previousLabelTrain = previousLabelTrain[,c(1,6:ncol(previousLabelTrain))]
previousLabelTrain = previousLabelTrain[,c('id','preLblNotEmpt')]
previousLabelTest  = merge(test, loc_time_order,'id', sort = FALSE)
#previousLabelTest = previousLabelTest[,c(1,6:ncol(previousLabelTest))]
previousLabelTest = previousLabelTest[,c('id','preLblNotEmpt')]

#table(previousLabelTrain$previousLabel)
#table(previousLabelTest$previousLabel)
save(previousLabelTrain,previousLabelTest, file = '../data/previousLabel.Rdata')


