library(glmnet)
install.packages("glmnet")
library(randomForest)
install.packages("randomForest")
install.packages("repr")
install.packages("coefplot")
install.packages("gridExtra")
library(glmnetUtils)me
install.packages("caret")
library(grid)
library(gridExtra)


hotel <- read.csv("Desktop/hotel data.csv")
hotel = hotel[,c(28,29,30,34,36,37,38,39,40,41)]
summary(hotel)

n = dim(hotel)[1]
p = dim(hotel)[2]
y = hotel[,7]
x=data.matrix(hotel[,-7])

#split data into train and test#
ntrain = floor(0.8*n)
ntest = n - ntrain

M = 100

#R square
#elastic net
rsq.test.ela      =     rep(0,M) 
rsq.train.ela     =     rep(0,M)
#ridge
rsq.test.ridge    =     rep(0,M)  
rsq.train.ridge   =     rep(0,M)
#lasso 
rsq.test.lasso    =     rep(0,M)  
rsq.train.lasso   =     rep(0,M)
#random forest
rsq.test.rf       =     rep(0,M) 
rsq.train.rf      =     rep(0,M)


# Lambda
lam.las = c(seq(1e-3,0.1,length=100),seq(0.12,2.5,length=100)) 
lam.rid = lam.las*1000

for(m in c(1:M)){
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:ntrain]
  test = shuffled_indexes[(1+ntrain):n]
  xtrain = x[train,]
  ytrain = y[train]
  xtest = x[test,]
  ytest = y[test]
  
#elastic a=0.5
  a =0.5
  elafit_cv = cv.glmnet(xtrain, ytrain, alpha=a,nfolds = 10)
  elafit = glmnet(xtrain, ytrain, alpha=a, lambda = elafit_cv$lambda.min)
  ytrain_hatela = predict(elafit,newx=xtrain, type="response")
  ytest_hatela = predict(elafit,newx=x, type="response")
  rsq.test.ela[m] = 1-mean((ytest-ytest_hatela)^2)/mean((y-mean(y))^2)
  rsq.train.ela[m] = 1-mean((ytrain-ytrain_hatela)^2)/mean((y-mean(y))^2)
  #ridge#
  b=0
  ridge.fit_cv        = cv.glmnet(xtrain, ytrain, alpha = b,lambda = lam.rid, nfolds = 10)
  optimal_lambda      = ridge.fit_cv$lambda.min
  ridge.fit           = glmnet(xtrain, ytrain, alpha = b, lambda = optimal_lambda)
  ytrain.hat_ridge    =     predict(ridge.fit, newx = xtrain, type = "response") 
  ytest.hat_ridge     =     predict(ridge.fit, newx = xtest, type = "response") 
  rsq.test.ridge[m]   =     1-mean((ytest - ytest.hat_ridge)^2)/mean((y - mean(y))^2)
  rsq.train.ridge[m]  =     1-mean((ytrain - ytrain.hat_ridge)^2)/mean((y - mean(y))^2) 
  #lasso#
  c=1 
  lasso.fit_cv        =     cv.glmnet(xtrain, ytrain, alpha = c, lambda = lam.las, nfolds = 10)
  optimal_lambda2 <- lasso.fit_cv$lambda.min
  lasso.fit           =     glmnet(xtrain, ytrain, alpha = c, lambda = optimal_lambda2)
  ytrain.hat_lasso    =     predict(lasso.fit, newx = xtrain, type = "response") 
  ytest.hat_lasso     =     predict(lasso.fit, newx = xtest, type = "response") 
  rsq.test.lasso[m]   =     1-mean((ytest - ytest.hat_lasso)^2)/mean((y - mean(y))^2)
  rsq.train.lasso[m]  =     1-mean((ytrain - ytrain.hat_lasso)^2)/mean((y - mean(y))^2)
  #randomForest#
  randomf           =     randomForest(xtrain, ytrain, mtry = sqrt(p), importance = TRUE, ntree = 100)
  ytest.hat_rf      =     predict(randomf, xtest)
  ytrain.hat_rf     =     predict(randomf, xtrain)
  rsq.test.rf[m]    =     1-mean((ytest - ytest.hat_rf)^2)/mean((y - mean(y))^2)
  rsq.train.rf[m]   =     1-mean((ytrain - ytrain.hat_rf)^2)/mean((y - mean(y))^2)

  cat(sprintf("m=%3.f| ,  rsq.test.ela=%.2f|,  rsq.train.ela=%.2f|, 
              rsq.test.ridge=%.2f|,  rsq.train.ridge=%.2f|,
              rsq.test.lasso=%.2f|,  rsq.train.lasso=%.2f|,
            rsq.test.rf=%.2f|,  rsq.train.rf=%.2f|\n", 
              m, rsq.test.ela[m], rsq.train.ela[m],
              rsq.test.ridge[m], rsq.train.ridge[m],
              rsq.test.lasso[m], rsq.train.lasso[m],
              rsq.test.rf[m], rsq.train.rf[m]
  ))
}

boxplot(rsq.train.rf, rsq.train.ela, rsq.train.lasso, rsq.train.ridge,
        main  = "Boxplot for R Square in Train Data",
        names = c("RF", "ELASTIC", "LASSO", "RIDGE"),
        col   = c("orange","red", "blue", "yellow"))

boxplot(rsq.test.rf, rsq.test.ela, rsq.test.lasso, rsq.test.ridge,
        main  = "Boxplot for R Square in Test Data",
        names = c("RF", "ELASTIC", "LASSO", "RIDGE"),
        col   = c("orange","red", "blue", "yellow"))


# boxplot for  10-fold CV cruves EN
plot(elafit_cv , 
     main="10-fold CV curves for Elastic")

# boxplot for  10-fold CV cruves Ridge
plot(ridge.fit_cv , 
     main="10-fold CV curves for Ridge")

# boxplot for  10-fold CV cruves Lasso
plot(lasso.fit_cv, 
     main="10-fold CV curves for Lasso")


# side by side residual boxplots

shuffled_indexes = sample(n)
train = shuffled_indexes[1:ntrain]
test = shuffled_indexes[(1+ntrain):n]
xtrain = x[train,]
ytrain = y[train]
xtest = x[test,]
ytest = y[test]
#elastic
a =0.5
elafit_cv = cv.glmnet(xtrain, ytrain, alpha=a,nfolds = 10)
elafit = glmnet(xtrain, ytrain, alpha=a, lambda = elafit_cv$lambda.min)
ytrain_hatela = predict(elafit,newx=xtrain, type="response")
ytest_hatela = predict(elafit,newx=x, type="response")
residu.test.ela = ytest - ytest_hatela
residu.train.ela = ytrain - ytrain_hatela

#ridge
b=0
ridge.fit_cv        = cv.glmnet(xtrain, ytrain, alpha = b,lambda = lam.rid, nfolds = 10)
optimal_lambda      = ridge.fit_cv$lambda.min
ridge.fit           = glmnet(xtrain, ytrain, alpha = b, lambda = optimal_lambda)
ytrain.hat_ridge    =     predict(ridge.fit, newx = xtrain, type = "response") 
ytest.hat_ridge     =     predict(ridge.fit, newx = xtest, type = "response") 
residu.test.rid = ytest - ytest.hat_ridge
residu.train.rid = ytrain - ytrain.hat_ridge


#lasso 
c=1 
lasso.fit_cv        =     cv.glmnet(xtrain, ytrain, alpha = c, lambda = lam.las, nfolds = 10)
optimal_lambda2 <- lasso.fit_cv$lambda.min
lasso.fit           =     glmnet(xtrain, ytrain, alpha = c, lambda = optimal_lambda2)
ytrain.hat_lasso    =     predict(lasso.fit, newx = xtrain, type = "response") 
ytest.hat_lasso     =     predict(lasso.fit, newx = xtest, type = "response") 
residu.test.las = ytest - ytest.hat_lasso
residu.train.las = ytrain - ytrain.hat_lasso

#random 
randomf           =     randomForest(xtrain, ytrain, mtry = sqrt(p), importance = TRUE, ntree = 100)
ytest.hat_rf      =     predict(randomf, xtest)
ytrain.hat_rf     =     predict(randomf, xtrain)
residu.test.rf = ytest - ytest.hat_rf
residu.train.rf = ytrain - ytrain.hat_rf

model = c(rep('Lasso', n), rep('ElasticNet', n), rep('Ridge', n), rep('RandomForest', n))
type  = c(rep('train', ntrain), rep('test', ntest), rep('train', ntrain), rep('test', ntest), rep('train', ntrain), rep('test', ntest), rep('train', ntrain), rep('test', ntest))
res   = c(residu.train.las, residu.test.las, residu.train.ela, residu.test.ela, residu.train.rid, residu.test.rid, residu.train.rf, residu.test.rf)
residu = res[c(1:262136)]
boxdata = data.frame(model, type, residu)
model_order = c('Lasso', 'ElasticNet', 'Ridge', 'RandomForest')

ggplot(boxdata, aes(x = factor(model, level = model_order), y = residu, fill = type)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "Residual") +
  scale_x_discrete(name = "Model") +
  ggtitle("Boxplot of Residual_train and Residual_test for Each Model") +
  theme_bw() +
  theme(plot.title = element_text(size = 12, family = "Tahoma", face = "bold", hjust = 0.5),
        text = element_text(size = 12, family = "Tahoma"),
        axis.title = element_text(face="bold"),
        axis.text.x = element_text(size = 10),
        legend.position = "bottom") +
  scale_fill_brewer(palette = "Pastel1") +
  labs(fill = "")


#bootystrapped error bars 
bootstrapSamples =    100
beta.la.bs       =    matrix(0, nrow = 9, ncol = bootstrapSamples)   
beta.en.bs       =     matrix(0, nrow = 9, ncol = bootstrapSamples)
beta.ri.bs       =     matrix(0, nrow = 9, ncol = bootstrapSamples)
beta.rf.bs       =     matrix(0, nrow = 9, ncol = bootstrapSamples)


for (m in 1:bootstrapSamples){
  bs_indexes       <-     sample(n, replace = T)
  X.bs             <-     x[bs_indexes, ]
  y.bs             <-     y[bs_indexes]
  # fit bs lasso
  cv.fit           <-     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.fit$lambda.min)  
  beta.la.bs[,m]   <-     as.vector(fit$beta)
  
  # fit bs elastic-net
  cv.fit           <-     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   <-     as.vector(fit$beta)
  
  # fit bs ridge
  cv.fit           <-     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.fit$lambda.min)  
  beta.ri.bs[,m]   <-     as.vector(fit$beta)

  
  # fit bs rf
  rf               <-     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
  beta.rf.bs[,m]   <-     as.vector(rf$importance[,1])
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}
sd.bs.la    <-   apply(beta.la.bs, 1, "sd")
sd.bs.en    <-   apply(beta.en.bs, 1, "sd")
sd.bs.ri    <-   apply(beta.ri.bs, 1, "sd")
sd.bs.rf    <-   apply(beta.rf.bs, 1, "sd")

cv.lasso    <-     cv.glmnet(x, y, alpha = 1, nfolds = 10)
lasso       <-     glmnet(x, y, alpha = 1, lambda = cv.lasso$lambda.min)

# fit elastic-net to the whole data
cv.elast    <-     cv.glmnet(x, y, alpha = 0.5, nfolds = 10)
elast       <-     glmnet(x, y, alpha = 0.5, lambda = cv.elast$lambda.min)

# fit ridge to the whole data
cv.ridge    <-     cv.glmnet(x, y, alpha = 0, nfolds = 10)
ridge       <-     glmnet(x, y, alpha = 0, lambda = cv.ridge$lambda.min)

# fit rf to the whole data
rf          <-     randomForest(x, y, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)

betaS.rf               <-     data.frame(names(x[1,]), as.vector(rf$importance[,1]), 2*sd.bs.rf)
colnames(betaS.rf)     <-     c( "feature", "value", "err")

betaS.en               <-     data.frame(names(x[1,]), as.vector(elast$beta), 2*sd.bs.en)
colnames(betaS.en)     <-     c( "feature", "value", "err")

betaS.la               <-     data.frame(names(x[1,]), as.vector(lasso$beta), 2*sd.bs.la)
colnames(betaS.la)     <-     c( "feature", "value", "err")

betaS.ri               <-     data.frame(names(x[1,]), as.vector(ridge$beta), 2*sd.bs.ri)
colnames(betaS.ri)     <-     c( "feature", "value", "err")
betaS.rf$feature  <-  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature  <-  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.la$feature  <-  factor(betaS.la$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ri$feature  <-  factor(betaS.ri$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

rfPlot <-  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) +
  ggtitle("Feature Importance of Random Forest")

enPlot <-  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) + 
  ggtitle("Feature Importance of Elastic Net")


# Compare elastic net and lasso/ridge
laPlot <-  ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) +
  ggtitle("Feature Importance of Lasso")

riPlot <-  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) + 
  ggtitle("Feature Importance of Ridge")

grid.arrange(enPlot, rfPlot, riPlot, laPlot,nrow = 2,ncol=2)

#summary slide 
start_lasso  <-  Sys.time()
cv.lasso     <-  cv.glmnet(x, y, alpha = 1, nfolds = 10)
lasso        <-  glmnet(x, y, alpha = 1, lambda = cv.lasso$lambda.min)
end_lasso    <-  Sys.time()
time_lasso   <-  end_lasso - start_lasso

start_elast  <-  Sys.time()
cv.elast     <-  cv.glmnet(x, y, alpha = 0.5, nfolds = 10)
elast        <-  glmnet(x, y, alpha = 0.5, lambda = cv.elast$lambda.min)
end_elast    <-  Sys.time()
time_elast   <-  end_elast - start_elast

start_ridge  <-  Sys.time()
cv.ridge     <-  cv.glmnet(x, y, alpha = 0, nfolds = 10)
ridge        <-  glmnet(x, y, alpha = 0, lambda = cv.ridge$lambda.min)
end_ridge    <-  Sys.time()
time_ridge   <-  end_ridge - start_ridge

start_rf     <-  Sys.time()
rf           <-  randomForest(x, y, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
end_rf       <-  Sys.time()
time_rf      <-  end_rf - start_rf


model <- c('Lasso', 'ElasticNet', 'Ridge', 'RandomForest')
performance <- round(c(mean(rsq.test.lasso), mean(rsq.test.ela), mean(rsq.test.ridge), mean(rsq.test.rf)), 3)
time <- round(c(time_lasso, time_elast, time_ridge, time_rf), 2)

summary_table <- data.frame(model, performance, time)
summary_table

