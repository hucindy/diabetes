# Early Detection of Diabetes Using Classification

# load libraries
library(dplyr)
library(ggplot2)
library(ggExtra)
library(gridExtra)
library(jtools)
library(ROCR)
library(party)
library(randomForest)
library(caret)

# load data
diabetes <- read.csv("diabetes.csv")
# convert character to factor
diabetes <- mutate_if(diabetes, is.character, as.factor)

# EDA
# class percent
diabetes %>% 
  group_by(class) %>% 
  summarize(count=n()) %>% 
  mutate(percent = prop.table(count)*100) %>%
  ggplot(aes(class,count, fill=class)) + 
  geom_col() +
  theme_minimal()+
  labs(x="", y="") +
  ggtitle("Diabetes") +
  theme(legend.position = "none")+
  geom_text(aes(label=paste0(round(percent,1),"%")), vjust=-.8, size=4) +
  ylim(0,350)

# age density and boxplot
p <- ggplot(diabetes, aes(x = Age, fill = class, colour = class)) +
  geom_point(aes(y = 0.03), alpha = 0) + # add an invisible scatterplot geom as the first layer
  geom_density(alpha = 0.8) +
  theme_minimal()+
  ggtitle("Age")+
  labs(x="Age", y="Density")

p %>% ggMarginal(type = "boxplot", 
                 margins = "x", 
                 size = 5,
                 groupColour = TRUE,
                 groupFill = TRUE)

# plot of proportion responses by class
grid.arrange(
  
  ggplot(diabetes, aes(x=Gender,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Gender")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Polyuria,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Polyuria")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Polydipsia,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Polydipsia")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=sudden.weight.loss,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Sudden weight loss")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=weakness,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Weakness")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Polyphagia,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Polyphagia")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Genital.thrush,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Genital thrush")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=visual.blurring,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Visual blurring")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Itching,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Itching")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Irritability,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Irritability")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=delayed.healing,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Delayed healing")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=partial.paresis,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Partial paresis")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=muscle.stiffness,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Muscle stiffness")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Alopecia,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Alopecia")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ggplot(diabetes, aes(x=Obesity,fill=class))+ 
    geom_bar(position = 'fill')+
    theme_bw() +
    ggtitle("Obesity")+
    theme(legend.position="none")+
    labs(x="",y="Proportion"),
  
  ncol=5
)

# prepare data for classification
# generate random number
set.seed(123) 
# split data into 80% for training, 20% for testing
ind <- sample(2, nrow(diabetes), replace=TRUE, prob=c(0.8,0.2))
train <- diabetes[ind==1,]
test <- diabetes[ind==2,]

# 1) Logistic regression model
# start with full model, then drop insignificant variables
#m1 <- glm(class~., data=train, family="binomial")
#m2 <- glm(class~Age+Gender+Polyuria+Polydipsia+weakness+Polyphagia+Genital.thrush+Itching+Irritability+partial.paresis, data=train, family="binomial")
#m3 <- glm(class~Gender+Polyuria+Polydipsia+weakness+Genital.thrush+Itching+Irritability+partial.paresis, data=train, family="binomial")
final_model <- glm(class~Gender+Polyuria+Polydipsia+Itching+Irritability, data=train, family="binomial")
summ(final_model)

# odds ratio plot
plot_summs(final_model, exp = TRUE) + labs(x="Odds Ratio")

# function to plot confusion matrix
draw_confusion_matrix <- function(cm, stitle) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(stitle, cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6,  col='black')
  text(195, 335, res[2], cex=1.6,  col='black')
  text(295, 400, res[3], cex=1.6,  col='black')
  text(295, 335, res[4], cex=1.6,  col='black')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}

# misclassification on test data
final_model_predict <- predict(final_model, test, type="response")
final_model_predict <- ifelse(final_model_predict>0.5, "Positive", "Negative")
# confusion matrix
cm <- confusionMatrix(as.factor(final_model_predict), test$class, positive ="Positive")
# plot confusion matrix
draw_confusion_matrix(cm, "Confusion Matrix - Logistic Regression")


# ROC curve
final_model_predict <- predict(final_model, test, type="response")
final_model_prediction <- prediction(final_model_predict, test$class)
roc <- performance(final_model_prediction, "tpr", "fpr")
plot(roc, colorize=TRUE, 
     main = "ROC Curve - Logistic Regression Model")
abline(a=0,b=1)
auc <- performance(final_model_prediction, "auc")
auc <- unlist(slot(auc, "y.values"))
legend(.8,.2, round(auc,4),title="AUC")

# 2) Random Forest

# plot decision tree
tree <- ctree(class ~ ., data=train)
plot(tree)

# random forest model
rf <- randomForest(class~. , data=train)
rf

# error rate of random forest model
plot(rf, main = "Error vs Number of trees")

# tune mtry
t <- tuneRF(train[,-17], train[,17],
            stepFactor = 0.5,
            plot=TRUE,
            ntreeTry=200,
            trace=TRUE,
            improve=0.05,
            main="OOB Error vs mtry")

# random forest ntree=200, mtry=4
rf <- randomForest(class~., data=train,
                   ntree=200,
                   mtry=4,
                   importance=TRUE,
                   proximity=TRUE)
rf

# after tuning hyperparameters
# prediction for testing data
rf_predict <- predict(rf, test)
rf_cm <- confusionMatrix(rf_predict, test$class, positive = "Positive")
# confusion matrix
draw_confusion_matrix (rf_cm, "Confusion Matrix - Random Forest (ntrees = 200, mtry = 4)")

# variable importance
varImpPlot(rf, sort=TRUE, main="Variable Importance - Random Forest (ntrees = 200, mtry = 4)")

# make prediction
rf_predict <- predict(rf, test,type='prob')
rf_prediction <- prediction(rf_predict[,2], test$class)

# ROC curve
roc <- performance(rf_prediction, "tpr", "fpr")
plot(roc, colorize=TRUE, 
     main = "ROC Curve - Random Forest")
abline(a=0,b=1)
auc <- performance(rf_prediction, "auc")
auc <- unlist(slot(auc, "y.values"))
legend(.8,.2, round(auc,4),title="AUC")
