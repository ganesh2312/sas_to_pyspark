data sample_data;
    input ID Age Gender $ Height Weight BMI Group $ Income;
    datalines;
1 25 M 175 70 22.9 A 50000
2 34 F 160 55 21.5 B 60000
3 29 M 180 80 24.7 A 55000
4 42 F 165 60 22.0 C 70000
5 31 M 170 75 25.9 B 65000
6 28 F 158 50 20.0 A 48000
7 36 M 178 85 26.8 C 72000
8 45 F 162 58 22.1 B 68000
9 30 M 172 77 26.0 A 53000
10 38 F 167 62 22.2 C 75000
;
run;

proc univariate data=sample_data;
run;


proc means data=sample_data mean median stddev min max n;
    var Age Height Weight;
    class Gender;
    title "Summary Statistics";
run;


/* Regression Analysis with Output Dataset */
proc reg data=sample_data outest=reg_coeff;
    model Income = Age BMI / vif tol r influence;
    output out=reg_output p=PredictedIncome r=Residuals;
    title "Regression Analysis with Output Data";
run;



proc logistic data=sample_data plots(only)=(roc); 
    class Group(ref='A');
    model Group = Age Income / selection=stepwise lackfit clparm=wald expb;
    output out=predictions p=predicted_prob;
    roc 'Age & Income Model';
    title "Enhanced Logistic Regression with ROC Curve and Model Diagnostics";
run;

/* 1. Summary Statistics using PROC MEANS */
proc means data=sample_data mean median stddev min max n;
    var Age Height Weight;
    class Gender;
    title "Summary Statistics";
run;


proc sgplot data=sample_data;
    scatter x=Age y=Income / group=Gender markerattrs=(symbol=circlefilled);
    reg x=Age y=Income / group=Gender;
    title "Scatter Plot with Regression Lines";
run;



proc univariate data=sample_data;
    var Income;
    histogram Income / normal kernel;
    title "Univariate Analysis of Income";
run;




proc format;
    value income_fmt
        low-50000 = 'Low'
        50001-70000 = 'Medium'
        70001-high = 'High';
run;

data sample_data_formatted;
    set sample_data;
    Income_Category = put(Income, income_fmt.);
run;




proc rank data=sample_data out=ranked_data percent;
    var Income;
    ranks Income_Percentile;
run;

proc tabulate data=sample_data;
    class Gender Group;
    var Income;
    table Gender*Group, Income*(mean std);
    title "Multi-Dimensional Summary Table";
run;


proc anova data=sample_data;
    class Group;
    model Income = Group;
    means Group / hovtest=levene;
    title "ANOVA with Levene's Test";
run;


proc gplot data=sample_data;
    plot Income*Age=Group;
    title "Income vs Age by Group";
run;







proc freq data=sample_data;
    tables Gender*Group / chisq expected nocol nopercent;
  
run;



proc print data=sample_data noobs;
    format Income dollar10.;
    title "Sample Dataset";
run;

proc means data=sample_data n mean std min max maxdec=2;
    class Group;
    var Age Height Weight BMI Income;
    title "Summary Statistics by Group";
run;




proc sort data=sample_data out=sorted_data;
    by Group descending Income;
run;

/* 5. PROC UNIVARIATE: Detailed analysis with histograms */
proc univariate data=sample_data;
    var Income;
    histogram Income / normal kernel;
    title "Univariate Analysis of Income";
run;




OPTIONS NONOTES NOSTIMER NOSOURCE NOSYNTAXCHECK;

/* 7. PROC REPORT: Advanced report with computed columns */
proc report data=sample_data nowd;
    column Group Income IncomeMean; /* Separate the computed column */
    define Group / group;
    define Income / mean format=dollar10.2;
    define IncomeMean / computed "Mean Income"; /* Define computed column */
    
    compute IncomeMean;
        IncomeMean = Income.mean; /* Use the mean of Income */
    endcomp;
    
    title "Advanced Report with Computed Columns";
run;

/* 8. PROC CORR: Correlation matrix with significance */

/* 9. PROC GLM: ANOVA with interaction effects */
proc glm data=sample_data;
    class Group;
    model Income = Group Age Group*Age / solution;
    lsmeans Group / pdiff adjust=tukey;
    title "ANOVA with Interaction Effects";
run;

/* 10. PROC MIXED: Mixed-effects model with random intercept */
proc mixed data=sample_data;
    class Group;
    model Income = Age BMI / solution;
    random Group;
    title "Mixed-Effects Model with Random Intercept";
run;



/* 12. PROC LOGISTIC: ROC curve and model fit */
proc logistic data=sample_data;
    class Group(ref='A');
    model Group = Age Income / ctable pprob=0.5;
    roc;
    title "Logistic Regression with ROC Curve";
run;


/* 16. PROC TRANSPOSE: Transpose with custom naming */
proc transpose data=sample_data out=transposed_data name=Variable prefix=Value;
    var Age Height Weight BMI Income;
    id ID;
run;

/* 17. PROC FORMAT: Custom formats for categorical data */

proc print data=sample_data;
    format Income income_fmt.;
    title "Dataset with Custom Income Format";
run;

/* 18. PROC CONTENTS: Detailed metadata */
proc contents data=sample_data varnum;
    title "Dataset Metadata";
run;



/* 20. PROC FACTOR: Factor analysis with rotation */
proc factor data=sample_data method=principal rotate=varimax;
    var Age Height Weight BMI Income;
    title "Factor Analysis with Varimax Rotation";
run;

/* 21. PROC CLUSTER: Hierarchical clustering */
proc cluster data=sample_data method=ward outtree=tree;
    var Age Height Weight BMI Income;
    id ID;
    title "Hierarchical Clustering";
run;

/* 22. PROC GPLOT: Additional visualization */
proc gplot data=sample_data;
    plot Income*Age=Group;
    title "Income vs Age by Group";
run;





/* -------------------------------------------------------------------------------------------- */


/* Create a sample dataset using DATALINES */
data sample_data;
    input Name $ Age Gender $ Height Weight;
    datalines;
John 23 M 68 155
Jane 22 F 65 130
Mike 25 M 70 175
Anna 24 F 64 120
Tom 23 M 72 180
Lucy 21 F 63 115
;
run;


/* 11. Tabular Summary using PROC TABULATE */
proc tabulate data=sample_data;
    class Gender;
    var Height Weight;
    table Gender, (Height Weight)*(mean stddev);
    title "Tabular Summary of Height and Weight by Gender";
run;

/* 1. Summary Statistics using PROC MEANS */
proc means data=sample_data mean median stddev min max n;
    var Age Height Weight;
    class Gender;
    title "Summary Statistics";
run;

/* 2. Frequency Analysis using PROC FREQ */
proc freq data=sample_data;
    tables Gender / nocum;
    title "Frequency Table for Gender";
run;

/* 3. Correlation Analysis using PROC CORR */
proc corr data=sample_data nosimple;
    var Height Weight;
    title "Correlation Analysis";
run;

/* 4. Regression Analysis using PROC REG */
proc reg data=sample_data outest=reg_estimates;
    model Weight = Height Age;
    output out=reg_output p=predicted_weight r=residuals;
    title "Linear Regression: Predicting Weight from Height and Age";
run;

/* 5. General Linear Model using PROC GLM */
proc glm data=sample_data;
    class Gender;
    model Weight = Height Age Gender / solution;
    means Gender / hovtest=levene;
    output out=glm_out predicted=predicted_weight residual=residuals;
    title "General Linear Model: Predicting Weight using Height, Age, and Gender";
run;

/* 6. Logistic Regression using PROC LOGISTIC */
proc logistic data=sample_data outmodel=logistic_model;
    class Gender (ref='F') / param=ref; /* F is the reference group */
    model Gender(event='M') = Height Weight Age / selection=stepwise;
    output out=logistic_output p=predicted_prob;
    ods output ParameterEstimates=logistic_parameters;
    title "Logistic Regression: Predicting Gender using Height, Weight, and Age";
run;

/* 7. Mixed-Effects Modeling using PROC MIXED */
proc mixed data=sample_data method=type3;
    class Gender;
    model Weight = Height Age / solution;
    random Gender;
    title "Mixed-Effects Model: Predicting Weight with Random Effect for Gender";
run;

/* 8. Generalized Linear Model using PROC GENMOD */
proc genmod data=sample_data;
    class Gender;
    model Weight = Height Age Gender / dist=normal link=identity;
    title "Generalized Linear Model: Predicting Weight with Normal Distribution";
run;

/* 9. Univariate Analysis using PROC UNIVARIATE */
proc univariate data=sample_data;
    var Age Height Weight;
    histogram / normal;
    inset mean stddev skewness kurtosis / position=ne;
    title "Univariate Analysis: Distribution of Age, Height, and Weight";
run;

/* 10. Analysis of Variance using PROC ANOVA */
proc anova data=sample_data;
    class Gender;
    model Weight = Gender;
    means Gender / tukey;
    title "ANOVA Analysis: Effect of Gender on Weight";
run;



/* 12. Chart Generation using PROC SGPLOT */
proc sgplot data=sample_data;
    scatter x=Height y=Weight / group=Gender;
    reg x=Height y=Weight / group=Gender;
    title "Height vs. Weight Scatter Plot with Regression Lines";
run;

/* 13. Transpose Data using PROC TRANSPOSE */
proc transpose data=sample_data out=transposed_sample;
    var Age Height Weight;
    title "Transposed Sample Data";
run;

/* 14. Rank Variables using PROC RANK */
proc rank data=sample_data out=ranked_sample ties=mean;
    var Height Weight;
    ranks Rank_Height Rank_Weight;
    title "Ranked Sample Data";
run;

/* 15. Factor Analysis using PROC FACTOR */
proc factor data=sample_data method=principal n=2;
    var Age Height Weight;
    title "Factor Analysis on Age, Height, and Weight";
run;

/* 16. Cluster Analysis using PROC CLUSTER */
proc cluster data=sample_data outtree=sample_tree method=average;
    var Height Weight;
    title "Cluster Analysis on Height and Weight";
run;

/* 17. Print the resulting dendrogram using PROC TREE */
proc tree data=sample_tree out=clustered_sample nclusters=3;
    title "Dendrogram for Cluster Analysis";
run;

/* 18. Display the contents of the dataset */
proc print data=sample_data label;
    title "Sample Data";
    var Name Age Gender Height Weight;
    label Name = "Full Name"
          Age = "Age (Years)"
          Gender = "Gender"
          Height = "Height (Inches)"
          Weight = "Weight (Pounds)";
run;

/* 19. Sort the dataset by Age */
proc sort data=sample_data out=sorted_sample;
    by Age;
run;

/* 20. Display the sorted dataset */
proc print data=sorted_sample label;
    title "Sorted Sample Data by Age";
    var Name Age Gender Height Weight;
    label Name = "Full Name"
          Age = "Age (Years)"
          Gender = "Gender"
          Height = "Height (Inches)"
          Weight = "Weight (Pounds)";
run;

/* 21. Another example - Predicting Weight from Height only */
proc reg data=sample_data;
    model Weight = Height;
    title "Linear Regression Analysis: Predicting Weight from Height";
run;