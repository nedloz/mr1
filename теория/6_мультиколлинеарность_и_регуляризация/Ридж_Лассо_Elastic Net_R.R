install.packages(c("readxl", "glmnet", "caret"))
library(readxl)
library(glmnet)
library(caret)

y <- data$y
x <- as.matrix(data[, -1])

#LASSO
cv_lasso <- cv.glmnet(x, y, alpha = 1)
plot(cv_lasso)
best_lambda_lasso <- cv_lasso$lambda.min
m_lasso <- glmnet(x, y, alpha = 1, lambda = best_lambda_lasso)
coef(m_lasso)

#Ridge
cv_ridge <- cv.glmnet(x, y, alpha = 0)
plot(cv_ridge)
best_lambda_ridge <- cv_ridge$lambda.min
m_ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda_ridge)
coef(m_ridge)

#Elastic net
cv_elastic <- cv.glmnet(x, y, alpha = 0.5)
plot(cv_elastic)
best_lambda_elastic <- cv_elastic$lambda.min
m_elastic <- glmnet(x, y, alpha = 0.5, lambda = best_lambda_elastic)
coef(m_elastic)

pred_lasso <- predict(m_lasso, x)
pred_ridge <- predict(m_ridge, x)
pred_elastic <- predict(m_elastic, x)

mse_lasso <- mean((y - pred_lasso)^2)
mse_ridge <- mean((y - pred_ridge)^2)
mse_elastic <- mean((y - pred_elastic)^2)

