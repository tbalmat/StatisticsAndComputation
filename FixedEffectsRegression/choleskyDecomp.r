options(max.print=1000)      # number of elements, not rows
options(stringsAsFactors=F)
options(scipen=999999)

library(Rcpp)

setwd("C:\\Projects\\Statistics\\ModelsMethodsTests\\CholeskyDecomposition")

#######################################################################################
#### Compute Cholesky decomposition of positive definite matrix X
#### Method that solves for elements of R by accumulating required sums along columns 
#######################################################################################

cSource <- "
  #include <Rcpp.h>
  #include <omp.h>
  // [[Rcpp::plugins(openmp)]]
  // following prefixes Rcpp type with Rcpp::
  using namespace Rcpp;

  // Algorithm:
  // 1. Compute elements of first row (row 0) of R.  These derive from the basic
  //    equation R'R = X:
  //    a. r00 = sqrt(x00)
  //    b. r0j = x0j/r00, j>0
  // 2. Proceed through rows 2 through p.  For each row i and column j>i:
  //    a. Compute diagonal element, rii = sqrt of [xii - sum of rki**2], k<i
  //    b. Compute rij = [xij - sum of rki*rkj]/rii, k<i

  // Notes:
  // 1. An array of p vector pointers, each addressing one row of R is used to
  //    avoid crash with large matrices (arrays created within function of size
  //    greater than [2047,2047] cause crash on execution)
  // 2. Static arrays of size greater than approximately [10000,10000] cause
  //    compilation errors, assembler messages are returned:  value of x too large
  //    for field of 4 bytes, possibly due to, say, 20,000 X 20,000 =
  //    400,000,000 * 8 (double) = 3.2 billion, which is greater than unsigned long
  //    capacity (it is assumed that assembler instructions must be generated to
  //    address matrix positions offset from position 0)
  // 3. Declaring a matrix with long long constant, as in static double
  //    R1[20000LL][20000LL]; also produces compiler 'too large' message
  // 4. It is assumed that the input matrix, X, is square, symmetric, and positive-
  //    definite

  // [[Rcpp::export]]
  NumericMatrix choleskyDecomp(NumericMatrix X, int nCore=0) {
    // Use long indices for high dimension addressing, but signed due to lack of
    // non-support of unsigned indices in Rcpp matrix objects
    long p=X.ncol(), i, j, k;
    double s;
    // Create p-length arrays of pointers, each addressing one column of R
    // Note that the length of column j is j since we compute upper triangle
    // and diagonal elements only
    double **R = new double *[p];
    for(j=0; j<p; j++) {
      R[j] = new double [j+1];
      memset(R[j], 0, sizeof(double)*(j+1));
    }

    // Set the max number of cores to use
    if(nCore>0)
      omp_set_num_threads(nCore);

    // Compute row 0
    R[0][0]=sqrt(X(0,0));
    for(j=1; j<p; j++)
      R[j][0]=X(0,j)/R[0][0];

    // Compute rows 1 through p-1
    for(i=1; i<p; i++) {

      // Compute diagonal
      s=0;
      for(k=0; k<i; k++)
        s+=R[i][k]*R[i][k];
      R[i][i]=sqrt(X(i,i)-s);

      // Compute columns i+1 through p-1
      #pragma omp parallel for private(j, k, s)
      for(j=i+1; j<p; j++) {
        s=0;
        for(k=0; k<i; k++)
          s+=R[i][k]*R[j][k];
        R[j][i]=(X(i,j)-s)/R[i][i];
      }

    }

    // Copy R to output array
    NumericMatrix Rout(p,p);
    std::fill(Rout.begin(), Rout.end(), 0);
    for(j=0; j<p; j++)    
      for(i=0; i<=j; i++)
        Rout(i,j)=R[j][i];

    // Release memory allocated to dynamic columns and accumulators
    for(j=0; j<p; j++)
      delete [] R[j];
    delete [] R;

    return(Rout);
  }"

# compile and create .dll
# cacheDir contains .dll and R instructions for loading and creating function from .dll
sourceCpp(code=cSource, rebuild=T, showOutput=T, cacheDir=getwd(), cleanupCacheDir=F)
