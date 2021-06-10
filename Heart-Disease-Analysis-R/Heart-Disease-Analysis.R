# Multiple linear regression :

# You are a public health researcher interested in social factors that influence heart disease. 
# You survey 500 towns and gather data on the percentage of people in each town who smoke, 
# the percentage of people in each town who bike to work, and the percentage of people in each town who have heart disease.
# Because you have two independent variables and one dependent variable, and all your variables are quantitative, you can use multiple linear regression to analyze the relationship between them.

library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)
library(RColorBrewer)

getwd()
data <- read.csv('heart.data.csv')
glimpse(data)

summary(data)

# No missing values :
sapply(data, function(x) sum(is.na(x)))

# No outliers : in other variables also : 
boxplot(data$heart.disease, col = 'skyblue', lwd = 2)

# *************************************************************************************************************************
# Checking Corelation between variables :
cor(data$biking, data$smoking)
print(paste('correlation : ', cor(data$biking, data$smoking)))          # "correlation :0.01513618"

# 0.015 means they are not highly correlated, the relation is small, its 1.5% correlation.
# Though we can include both the parameters/variables.

print(paste('correlation : ', cor(data$smoking, data$heart.disease)))   #"correlation :0.30913097"

print(paste('correlation : ', cor(data$biking, data$heart.disease)))    #"correlation :-0.93545547"
# Here it is strongly negative correlation.

# ************************************************************************************************************************
#
# Checking the 'Normality' : 

# (Normal distribution) by visualizing the data by histogram
hist(data$heart.disease, xlab = 'Heart disease', ylab ='Frequency', 
     col=brewer.pal(n = 5, name = "RdBu"), border = 'white', main = 'Heart disease / Frequency', 
     xlim = c(0,24), lwd = 2)

# Insight : 
# The distribution of observations is roughly bell-shaped, so we can proceed with the linear regression.
#
# *************************************************************************************************************************

# Checking 'Linearity' :

# We can check this using two scatterplots: one for heart disease with biking, and one for heart disease with smoking variable,
# Since its a multiple-linear-regression analysis / modelling :

# 1. heart disease with biking.
plot(data$heart.disease ~ data$biking, xlab = 'Biking', ylab = 'Heart disease numbers',
     main = 'Biking vs Heart disease', col=brewer.pal(n = 3, name = "Set1"))

# Insight : 
# We can see that there is a linearity 

# 2. heart disease vs smoking.
plot(data$heart.disease ~ data$smoking, xlab = 'Smoking', ylab = 'Heart disease numbers',
     main = 'Smoking vs Heart disease', col=brewer.pal(n = 3, name = "Set1"))

# Insight : 
# Although the relationship between smoking and heart disease is a bit less clear, it still appears linear. 
# We can proceed with linear regression.
#
# ************************************************************************************************************************* 

# Performing Multiple-Linear Analysis :

heart.disease_lm <- lm(heart.disease ~ biking + smoking, data = data)
summary(heart.disease_lm)

# Insights : 
# The estimated effect of biking on heart disease is -0.2, while the estimated effect of smoking is 0.178.

# This means that for every 1% increase in biking to work, there is a correlated 0.2% decrease in the incidence of heart disease. 
# Meanwhile, for every 1% increase in smoking, there is a 0.178% increase in the rate of heart disease.

# The standard errors for these regression coefficients are very small, and the t-statistics are very large (-147 and 50.4, respectively). 
# The p-values reflect these small errors and large t-statistics. For both parameters, there is almost zero probability that this effect is due to chance.

# *************************************************************************************************************************

# Checking for homoscedasticity
# Again, we should check that our model is actually a good fit for the data, and that we don't have large variation in the model error

par(mfrow = c(2, 2))
plot(heart.disease_lm, col = brewer.pal(n = 3, name = "Set1"))

# the residuals show no bias, so we can say our model fits the assumption of homoscedasticity.
#
# *************************************************************************************************************************

# Visualizing results with the graphs. 

# plotting the relationship between biking and heart disease at different levels of smoking. In this example, smoking will be treated as a factor with three levels, 
# just for the purposes of displaying the relationships in our data.

# 1. Create a new dataframe with the information needed to plot the model :
plotting.data<-expand.grid(
  biking = seq(min(data$biking), max(data$biking), length.out=30),
  smoking=c(min(data$smoking), mean(data$smoking), max(data$smoking)))

# 2. Predicting the values of heart disease based on our linear model :
plotting.data$predicted.y <- predict.lm(heart.disease_lm, newdata = plotting.data)

# 3. Rounding up the smoking numbers to two decimals :
plotting.data$smoking <- round(plotting.data$smoking, digits = 2)

# 4. Changing smoking variable into factor :
# This allows us to plot the interaction between biking and heart disease at each of the three levels of smoking we chose.
plotting.data$smoking <- as.factor(plotting.data$smoking)

# 5. Plotting the original data : 
theme_set(
  theme_minimal() +
    theme(legend.position = "top")
)

heart.plot <- ggplot(data, aes(x = biking, y = heart.disease, color = "data")) + geom_point()
heart.plot

# 6. Adding the regression line(s) :
heart.plot <- heart.plot + geom_line(data = plotting.data, aes(biking, predicted.y, color = smoking), size = 1)
heart.plot

# 7. Making the final graph ready for publication : 
heart.plot <- heart.plot + theme_bw() + 
  labs(title = " Rates of heart disease (% of population) \n as a function of biking to work and smoking", 
       x = 'Biking to work (% of the population)', 
       y = 'Heart disease (% of the population)', 
       color = "Smoking \n (% of population)")
heart.plot


heart.plot + annotate(geom="text", x=30, y=1.75, label=" = 15 + (-0.2*biking) + (0.178*smoking)")



# **************************************************************************************************************
#
# Reporting results of simple linear regression : 
#
# We found a significant relationship between income and happiness (p < 0.001, R2 = 0.73 ± 0.0193), with a 0.73-unit 
# increase in reported happiness for every $10,000 increase in income.
#
#
# Results of multi-linear regression : 
# 
# 1. In our survey of 500 towns, we found significant relationships between the frequency of biking to work and the 
# frequency of heart disease and the frequency of smoking and frequency of heart disease (p < 0 and p<0.001, respectively).
#
# 2. Specifically we found a 0.2% decrease (± 0.0014) in the frequency of heart disease for every 1% increase in biking, 
# and a 0.178% increase (± 0.0035) in the frequency of heart disease for every 1% increase in smoking.
#
# **************************************************************************************************************














