
# 1.To record the patient statistics, the agency wants to find the age category of people who frequently visit the hospital and has the maximum expenditure.
getwd()
hosp_data <- read.csv('HospitalCosts.csv')
head(hosp_data)
summary(hosp_data)
table(hosp_data$AGE)

# Treating missing values
colSums(is.na(hosp_data))
hosp_data <- na.omit(hosp_data)
colSums(is.na(hosp_data))

# Record patient statistics:
summary(as.factor(hosp_data$AGE))
max(table(hosp_data$AGE))
max(summary(as.factor(hosp_data$AGE)))
which.max(table(hosp_data$AGE))                
# age group 0 to 1

# Histogram of Age group to understand frequency of particular age group visiting hospital :
hist(hosp_data$AGE, main = 'Histogram of Age Group and their hospital visits',
     xlab = 'Age group', border = 'black', col = c('light blue', 'dark green'), xlim = c(0,20), ylim = c(0,350))
# Even seeing the histogram we come to know that the most frequently visiting age group is 0 - 1 years. 

# Summarizing total expenditure based on age group :
ExpBasedAge <- aggregate(TOTCHG ~ AGE, data = hosp_data, sum)
print(paste('Total Expenditure =', max(ExpBasedAge)))
# "Total Expenditure = 678118"

# Visualization : Age vs Expenditure
barplot(tapply(ExpBasedAge$TOTCHG, ExpBasedAge$AGE, FUN = sum), main = 'Age vs Expenditure',
        xlab='Age group', ylab = 'Expenses', border='black', col = 'darkorange', xlim = c(0,20))



# 2.In order of severity of the diagnosis and treatments and to find out the expensive treatments, the agency wants to find the 
# diagnosis-related group that has maximum hospitalization and expenditure.
summary(as.factor(hosp_data$APRDRG))
DiagnosisCost <- aggregate(TOTCHG ~ APRDRG, data = hosp_data, FUN = sum)
DiagnosisCost[which.max(DiagnosisCost$TOTCHG),]

#  APRDRG TOTCHG
#    640  437978

# 640 diagnosis related group had a max cost of 437978


# 3.To make sure that there is no malpractice, the agency needs to analyze if the race of the patient is related to the hospitalization costs
summary(as.factor(hosp_data$RACE))
raceInfluence <- lm(hosp_data$TOTCHG ~ hosp_data$RACE)
summary(raceInfluence)

# ****************************************************************************************************
# Call:                                                                                              *
# lm(formula = hosp_data$TOTCHG ~ hosp_data$RACE)                                                    *
#                                                                                                    *
# Residuals:                                                                                         *
#   Min     1Q Median     3Q    Max                                                                  *
# -2256  -1560  -1227   -258  45600                                                                  *
#                                                                                                    *
# Coefficients:                                                                                      *
#                  Estimate Std. Error t value Pr(>|t|)                                              *
# (Intercept)        2925.7      405.0   7.224 1.92e-12 ***                                          *
#   hosp_data$RACE   -137.3      339.1  -0.405    0.686                                              *
# ---                                                                                                *
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1                                   *
#                                                                                                    *
# Residual standard error: 3895 on 497 degrees of freedom                                            *
# Multiple R-squared:  0.0003299,	Adjusted R-squared:  -0.001681                                     * 
# F-statistic: 0.164 on 1 and 497 DF,  p-value: 0.6856                                               *
#                                                                                                    *
# ****************************************************************************************************

# Here p-value is 0.69 it is much higher than 0.5
# We can say that race doesn't affect the hospitalization costs.



# 4.To properly utilize the costs, the agency has to analyze the severity of the hospital costs by age and gender for the proper allocation of resources.
table(hosp_data$FEMALE)
a <- aov(TOTCHG ~ AGE + FEMALE, data = hosp_data)
summary(a)
b <- lm(TOTCHG ~ AGE + FEMALE, data = hosp_data)
summary(b)

# ****************************************************************************************************
# Call:                                                                                              *
# lm(formula = TOTCHG ~ AGE + FEMALE, data = hosp_data)                                              *
#                                                                                                    *
# Residuals:                                                                                         *
#   Min     1Q Median     3Q    Max                                                                  *
# -3403  -1444   -873   -156  44950                                                                  *
#                                                                                                    *
# Coefficients:                                                                                      *
#               Estimate  Std. Error t value Pr(>|t|)                                                *
# (Intercept)    2719.45     261.42  10.403  < 2e-16 ***                                             *
#   AGE            86.04      25.53   3.371 0.000808 ***                                             *
#   FEMALE       -744.21     354.67  -2.098 0.036382 *                                               *
# ---                                                                                                *
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1                                   *
#                                                                                                    *
# Residual standard error: 3849 on 496 degrees of freedom                                            *
# Multiple R-squared:  0.02585,	Adjusted R-squared:  0.02192                                         *        
# F-statistic: 6.581 on 2 and 496 DF,  p-value: 0.001511                                             *
#                                                                                                    *
# ****************************************************************************************************

# Since the pValues of AGE is much lesser than 0.05, it means AGE has the most statistical significance. 
# Similarly, gender is also less than 0.05. 
# Hence, we can conclude that the model is statistically significant.


# 5. Since the length of stay is the crucial factor for inpatients, the agency wants to find if the length of stay can be predicted from age, gender, and race.
table(hosp_data$LOS)
stay <- aov(LOS ~ AGE + FEMALE + RACE, data = hosp_data)
summary(stay)
stay <- lm(LOS ~ AGE + FEMALE + RACE, data = hosp_data)
summary(stay)

# ****************************************************************************************************                                                                                                  
# Call:                                                                                              *
# lm(formula = LOS ~ AGE + FEMALE + RACE, data = hosp_data)                                          *         
#                                                                                                    *
# Residuals:                                                                                         *
#   Min     1Q Median     3Q    Max                                                                  *
# -3.22  -1.22  -0.85   0.15  37.78                                                                  *
#                                                                                                    *
# Coefficients:                                                                                      *
#              Estimate Std. Error t value Pr(>|t|)                                                  *
# (Intercept)  2.94377    0.39318   7.487 3.25e-13 ***                                               *
#   AGE       -0.03960    0.02231  -1.775   0.0766 .                                                 *
# FEMALE       0.37011    0.31024   1.193   0.2334                                                   *
# RACE        -0.09408    0.29312  -0.321   0.7484                                                   *
# ---                                                                                                *        
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1                                   *
#                                                                                                    *        
# Residual standard error: 3.363 on 495 degrees of freedom                                           *
# Multiple R-squared:  0.007898,	Adjusted R-squared:  0.001886                                      *
# F-statistic: 1.314 on 3 and 495 DF,  p-value: 0.2692                                               *
#                                                                                                    *
# ****************************************************************************************************


# 6.To perform a complete analysis, the agency wants to find the variable that mainly affects hospital costs.
compAnalysis <- aov(TOTCHG ~., data = hosp_data)
summary(compAnalysis)
compAnalysis <- lm(TOTCHG ~., data = hosp_data)
summary(compAnalysis)

# ****************************************************************************************************
# Call:                                                                                              *
# lm(formula = TOTCHG ~ ., data = hosp_data)                                                         *
#                                                                                                    *
# Residuals:                                                                                         *
#   Min     1Q Median     3Q    Max                                                                  *
# -6377   -700   -174    122  43378                                                                  *
#                                                                                                    *
# Coefficients:                                                                                      *
#               Estimate  Std. Error t value  Pr(>|t|)                                               *
# (Intercept)   5218.6769   507.6475  10.280  < 2e-16 ***                                            *
#   AGE          134.6949    17.4711   7.710 7.02e-14 ***                                            *
#   FEMALE      -390.6924   247.7390  -1.577    0.115                                                *
#   LOS          743.1521    34.9225  21.280  < 2e-16 ***                                            *
#   RACE        -212.4291   227.9326  -0.932    0.352                                                *
# APRDRG          -7.7909     0.6816 -11.430  < 2e-16 ***                                            *
#   ---                                                                                              *
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1                                   *     
#                                                                                                    *
# Residual standard error: 2613 on 493 degrees of freedom                                            *
# Multiple R-squared:  0.5536,	Adjusted R-squared:  0.5491                                          *
# F-statistic: 122.3 on 5 and 493 DF,  p-value: < 2.2e-16                                            *
#                                                                                                    *
# ****************************************************************************************************

# We can say that Age, Length of stay (LOS) and patient refined diagnosis related groups(APRDRG) have 
# three stars (***) next to it. So they are the ones with statistical significance.
# These three variables mainly effects the hospital cost.



# Analysis Conclusion:

# 1. health care costs is dependent on age, length of stay and the diagnosis type.

# 2. Healthcare cost is the most for patients in the 0-1 yrs age group category.
#    i. Maximum expenditure for 0-1 yr is 678118

# 3. Length of Stay increases the hospital cost.

# 4. All Patient Refined Diagnosis Related Groups also affects healthcare cost.
#    i. 640 diagnosis related group had a max cost of 437978.

# 5. Race or gender doesn't have that much impact on hospital cost






















