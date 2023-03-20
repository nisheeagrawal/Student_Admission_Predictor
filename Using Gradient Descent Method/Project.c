//NISHEE AGRAWAL FINAL PROJECT COMP 526
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//Declaration of Variables to Calculate Mathematical Operations
//Declaration of Mean Function
double MEAN_X(int N, double x[]);
//Declaration of Standard Deviation function     
double STD_X(int N, double x[],double x_bar);     
//Declaration of Normalize Function    
double** NORMALIZE_X(int M,int N,double mat[M][N]);   
//Declaration of Function to add Column 
double** ADD_UNITCOL2_X(int M,int N, double** norm_return);     
//Declaration of Sigmoid function for optimization    
double SIGMOID(double z);
//Declaration of Cost Function to calculate cost value
struct getCost GET_COST(int M,int N,double Y[],double theta[],double ** mat);
//Declaration of Gradiendt Descent function s
struct gradDescent GRAD_DESCENT(int Niter,int M,int N,double alpha,double Y[],double theta[],double ** X,double tolerance);
//Declaration of Predict function to calculate values
double * PREDICT(int M,int N,double theta[],double ** X);
//Declaration of Precision function 
double PRECISION(int M,double * Y,double * YPred);
//Declaration of Recall function
double RECALL(int M,double * Y,double * YPred);
//Declaration of WriteToFile function
void writeToFile(double *jh,double trainPRECISION, double trainRECALL,double testPRECISION, double testRECALL);

// Variable used to specify file name
static const char input_file_name[] = "student_data.txt";
//Declaring structure for Cost function return value
struct getCost{
    double cost;
    double *GV;
};
//Declaring structure for Gradient Descent function return value 
struct gradDescent{
    double *jh;
    double *theta;
};
//Main function 
int main()
{
//Declaration of variables 
   char buffer[1024] ;
   char *record,*line;
   int i=0,j=0;
   //Declaring variables for dimension of input 
   int M=400,N=3; 
   double X[M][N]; 
   //Variables for returning the value of Normalize Function and Adding column function 
   double **NX, **UX;
   double Y[M];
   //Command to read the input file 
   FILE *fstream = fopen(input_file_name,"r");
   if(fstream == NULL)
   {

      printf("\n file opening failed ");
      return -1 ;
   }
   //Reading input lines
   int count = 1;
   while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
   {
   //Splitting input lines based on tab space
     record = strtok(line,"\t");
     while(record != NULL)
     {
     if(count == 1){
        Y[i] = atof(record);
     } 
     else{
         X[i][j++] = atof(record); 
     }
    //Recording the track of input length for Y variable inputs
     count++;
     if(count == 5) count = 1;
     record = strtok(NULL,"\t");
     }
     j = 0;
     ++i ;
   }

    NX = NORMALIZE_X(M, N, X);
    UX = ADD_UNITCOL2_X(M, N, NX);
   
   int trainEndIndx = M * 0.8;

   //Declaration of variable for Test and Train Data
   double* trainY = malloc(trainEndIndx * sizeof(double));
   double* testY = malloc((M-trainEndIndx) * sizeof(double));
   //Copying Train and test Y data
   memcpy(trainY, Y, trainEndIndx * sizeof(double));
   memcpy(testY, Y + trainEndIndx, (M-trainEndIndx) * sizeof(double));
   double** trainX;
   double** testX;

   //Memory allocation in Train data
   trainX = malloc(sizeof(double*) * trainEndIndx);
   for(i = 0; i < trainEndIndx; i++) { 
       trainX[i] = malloc(sizeof(double*) * 3); 
    }
    testX = malloc(sizeof(double*) * (M-trainEndIndx));
   for(i = 0; i < (M-trainEndIndx); i++) { 
       testX[i] = malloc(sizeof(double*) * 3); 
    }
    //Copying Train and test X data
    for(i = 0 ; i < trainEndIndx ; i++){
        trainX[i] = UX[i];
    }
    for(i = trainEndIndx; i < M ; i++){
        testX[trainEndIndx-i] = UX[i];
    }

//Declaring number of Iteration    
   int Niter = 1400;
   double alpha = 0.01;
   double tolerance = 0.000001;
   double theta[N+1];
   for(i = 0 ; i < N+1; i++){
       theta[i] = 0.0;
   }
//Calculation of Theta values and value of cost function   
   struct gradDescent descent = GRAD_DESCENT(Niter,trainEndIndx,N+1,alpha,trainY,theta,trainX,tolerance);
   printf("cost at last step = %lf\n",descent.jh[1399]);
   printf("Theta = [%lf, %lf ,%lf ,%lf] \n",descent.theta[0] ,descent.theta[1] ,descent.theta[2] ,descent.theta[3]);
//Predicting Y values for train Data   
   double *predicted = PREDICT((M-trainEndIndx),N+1, descent.theta, testX);
 //Calculating Y values for test Data and value of Precision and recall  
   double testPrecision = PRECISION((M-trainEndIndx), testY, predicted);
   double testRecall = RECALL((M-trainEndIndx), testY, predicted);
   printf("Test precision : %lf ,Test recall : %lf \n", testPrecision, testRecall);
   predicted = PREDICT((trainEndIndx),N+1, descent.theta, trainX);
   double trainPrecision = PRECISION((trainEndIndx), trainY, predicted);
   double trainRecall = RECALL((trainEndIndx), trainY, predicted);
   printf("Train precision : %lf ,Train recall : %lf\n", trainPrecision, trainRecall);
//Writing history of cost function, Precision and recall values of Train and Test data to a file 
   writeToFile(descent.jh, trainPrecision, trainRecall, testPrecision, testRecall);
   return 0 ;
}
//Recalling the function 
void writeToFile(double *jh,double trainPrecision, double trainRecall,double testPrecision, double testRecall){
   FILE *fp;
   int i; 
   fp = fopen("CostFunctionHistory.txt", "a+");
   fprintf(fp,"\nHistory of cost function:\n");
   for( i = 0 ; i < 1400 ; i++){
       fprintf(fp, "%lf\t", jh[i]);  
   }
   //Writing the values into file
   printf("\n");
   fprintf(fp,"\nTrain Precision: %lf \n",trainPrecision);
   fprintf(fp,"Train Recall: %lf \n",trainRecall);
   fprintf(fp,"Test Precision: %lf \n",testPrecision);
   fprintf(fp,"Test Recall: %lf \n",testRecall);
   fclose(fp);
}
//Function to calculate mean of given vector
double MEAN_X(int N, double x[])         
{
    int i;
    int k;
    
    double sum = 0;
    double x_bar = 0;
    for(i = 0; i < N; i++){
       sum = sum+x[i];
    }
    x_bar = sum/(double)N;
   
    return x_bar;
}
//Function to calculate standard deviation of given vector
double STD_X(int N, double x[],double x_bar)         
{
    int i;
    double sum = 0;
    double sigma_x;
    for(i = 0; i < N; i++){
       sum = sum+(x[i]-x_bar) * (x[i]-x_bar);
    }
    sigma_x = sqrt(sum/(N-1));
    return sigma_x;
}
//Function to normalize given vector
double** NORMALIZE_X(int M,int N, double mat[M][N]){
    int i,j;
    double **W; 
    W = malloc(sizeof(double*) * M);
    for(i = 0; i < M; i++) { 
    W[i] = malloc(sizeof(double*) * N); 
     }  

 
    double SX[N];
    double MX[N];
    double temp[M];
    for(j = 0; j < N; j++){
       for(i = 0 ; i < M; i++){
           temp[i] = mat[i][j];
       }
       MX[j] = MEAN_X(M,temp);
       SX[j] = STD_X(M,temp,MX[j]);
    }

    for(i = 0; i < M; i++){
        for(j = 0; j < N; j++){
             W[i][j] = mat[i][j] - MX[j];
             W[i][j] = W[i][j]/SX[j];
           
        }
    }
    return  W;
}
//Function to add unit vector to a given matrix
double** ADD_UNITCOL2_X(int M,int N,double** norm_return){
    int i,j;
    double **unit; 
    unit = malloc(sizeof(double*) * M);
    for(i = 0; i < M; i++) { 
    unit[i] = malloc(sizeof(double*) * (N+1)); 
     }  

    for(i = 0; i < M; i++){
        for(j = 0; j < N+1; j++){
           if(j == 0) {
               unit[i][j] = 1.0;
           }
           else{
            unit[i][j] = norm_return[i][j-1];
           } 
        }
    }
    return  unit;
}
//Function to calculate sigmoid of given values
double SIGMOID(double z){
   
    return 1.0/(1.0+exp(-z));
}
// function to calculate the value of cost function used in optimisation
struct getCost GET_COST(int M,int N,double Y[],double theta[],double ** mat){
    double costvalue,scf;
    double H[M];
    double HS[M];
    int i,j;
    double *G; 
    G = malloc(sizeof(double*) * N);
    struct getCost cost;
    for(i = 0; i < M;i++){
        H[i] = 0.0;
        for(j = 0; j < N;j++){
            H[i] = H[i]+mat[i][j] * theta[j];
        }
    }
    for(i = 0;i < M; i ++){
        HS[i] = SIGMOID(H[i]);
    }
    scf = 0.0;
    for(i = 0; i < M; i++){
        scf = scf + ((-1) * Y[i] * log(HS[i]))- ((1-Y[i]) * log(1-HS[i]));
    }
    costvalue = scf/M;
    for(j = 0; j < N;j++){
        G[j] = 0.0;
        for(i = 0; i < M; i++){
            G[j] = G[j]+((HS[i] - Y[i]) * mat[i][j]);
        }
        G[j] = G[j]/M;
    }

     cost.cost = costvalue;
     cost.GV = G;    

  
    return cost;
}

//function to calculate the value of theta vector
struct gradDescent GRAD_DESCENT(int Niter,int M,int N,double alpha,double Y[],double theta[],double ** X,double tolerance){
    double *jh; 
    double Jval;
    double malpha;
    struct gradDescent descent;
    struct getCost cost;
    jh = malloc(sizeof(double*) * Niter);
    double H[M];  
    double *G; 
    G = malloc(sizeof(double*) * N);
    int i,j,k;
    k = 0;
    Jval = 2 * tolerance;
    while((k <= Niter) && (Jval > tolerance)){
         cost = GET_COST(M,N, Y, theta, X);
         Jval = cost.cost;
         G = cost.GV;
         malpha = alpha/M;
         for(j = 0; j < N;j ++){
             theta[j] = theta[j] -(malpha * G[j]);
         }
        
         jh[k] = Jval;
         k = k+1;
    }
    descent.jh = jh;
    descent.theta = theta;
    return descent;
}
//function to predict the outcome
double * PREDICT(int M,int N,double theta[],double ** X){
    int i,j;
    double *YPred;
    YPred = malloc(sizeof(double) * M);
    for(i = 0; i < M; i++){
        double z = 0.0;
        for(j = 0; j < N+1; j++){
            z = z + (X[i][j] * theta[j]);
        }
        double prediction = SIGMOID(z);
        if(prediction >= 0.5){
            YPred[i] = 1.0;
        } else {
            YPred[i] = 0.0;
        }
    }
    return  YPred;
}

//function to calculate the precision value
double PRECISION(int M,double * Y,double * YPred){
    int i;
    double truePositive = 0,falsePositive = 0;
    for(i = 0; i < M; i++){
        if(YPred[i] == 1.0 && Y[i] == 1.0){
           truePositive = truePositive + 1;
        }
        else if(YPred[i] == 1.0 && Y[i] == 0.0){
            falsePositive = falsePositive + 1;
        }
    }
    return  truePositive/(truePositive + falsePositive);
}
// function to calculate the recall value
double RECALL(int M,double * Y,double * YPred){
  int i;
  double truePositive = 0,falseNegative = 0;
    for(i = 0; i < M; i++){
        if(YPred[i] == 1.0 && Y[i] == 1.0){
           truePositive = truePositive + 1;
        }
        else if(YPred[i] == 0.0 && Y[i] == 1.0){
            falseNegative = falseNegative + 1;
        }
    }
    return  truePositive/(truePositive + falseNegative);
}

