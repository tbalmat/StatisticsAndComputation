options(max.print=1000)      # number of elements, not rows
options(stringsAsFactors=F)
options(scipen=999999)

library(Rcpp)

setwd("C:\\Projects\\Statistics\\ModelsMethodsTests\\CholeskyDecomposition")

#########################################################################################
#### Compute inverse of nonsingular matrix X using prior computed Cholesky
#### decomposition R (note R'R=X, where X is a positive-definite matrix)
#### Return diagonal of X inverse
#######################################################################################

cSource <- "
  #include <Rcpp.h>
  #include <omp.h>
  // [[Rcpp::plugins(openmp)]]
  // following prefixes Rcpp type with Rcpp::
  using namespace Rcpp;

  // Algorithm:

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
  NumericVector cholInvDiag(NumericMatrix R, int nCore=0) {
    // Use long indices for large addressing, but signed since i is decremented to -1
    long p=R.ncol();
    long i, j, k;
    double s;

    // Set the max number of cores to use
    if(nCore>0)
      omp_set_num_threads(nCore);

    // Create internal copy of R, since it is used in parallel omp operations and
    // use of Rcpp objects within parallel operations exhibits instability (crash!)
    // and reduced performance (when compared to using native C arrays and pointers)
    // Create a p-length array of pointers, one for each row of R, the ith element
    // addressing row i of R
    double **R2 = new double *[p];
    for(i=0; i<p; i++) {
      R2[i] = new double [p];
      memset(R2[i], 0, sizeof(double)*p);
      // R is upper triangular, so do not copy lower zeroes
      // Divide rows of R by diagonals (sets diag to 1)
      // This eliminates the need of dividing by R[i,i] when constructing elements
      // in row i+1
      R2[i][i]=1;
      for(j=i+1; j<p; j++)
        R2[i][j]=R(i,j)/R(i,i);
    }

    // iX is the computed inverse of X, where X=R'R
    // It is the solution to R'R*iX = I
    double **iX = new double *[p];
    for(i=0; i<p; i++) {
      iX[i] = new double [p];
      memset(iX[i], 0, sizeof(double)*p);
      // Diagonals of right hand matrix (initially I) are the squared reciprocals
      // of original R diagonals
      // They are the initial values of the inverse
      iX[i][i]=1/R(i,i)/R(i,i);
    }

    // The vector d will contain the diagonal of iX and is returned
    NumericVector d(p);

    // Let's go!

    // Construct upper rows of the inverse in reverse order
    // Copy to transpose positions for use by following row construction
    // Note that rows cannot be constructed in parallel since prior rows
    // are necessary to construct subsequent rows
    //omp_set_num_threads();
    for(i=p-2; i>=0; i--) {
      // Construct columns in parallel since elements of rows are referenced
      #pragma omp parallel for private(j, k, s)
      for(j=i+1; j<p; j++) {
        s=0;
        // The following iterative loop is the computationally intensive
        // part this algorithm
        // Note that, although the analytical matrix equations multiply
        // R[i,k]' by iX[k,j], it is observed that iX is symmetric, which
        // allows the equivalent operation R[i,k]' * iX[j,k] and is
        // considerably more efficient as implemented, since the elements
        // of iX[j][k] are contiguous in memory (the C compiler may
        // implement 'loop unrolling')
        // Note that each increment of j corresponds to an in memory
        // distance of at least p*8, where 8 is the number of bytes
        // per double floating point element
        for(k=i+1; k<p; k++)
          s+=R2[i][k]*iX[j][k];
        iX[i][j]=-s;
        // Copy to transpose position
        iX[j][i]=-s;
      }
      // Compute diagonal element on current row
      for(j=i+1; j<p; j++)
        iX[i][i]=iX[i][i]-R2[i][j]*iX[i][j];
    }

    // Done!
    // Copy all diagonal elements of computed inverse to result vector
    for(i=0; i<p; i++)
      d(i)=iX[i][i];

    // Release memory allocated to dynamic arrays
    for(i=0; i<p; i++) {
      delete [] R2[i];
      delete [] iX[i];
    }
    delete [] R2;
    delete [] iX;

    return(d);
  }"

# compile and create .dll
# cacheDir contains .dll and R instructions for loading and creating function from .dll
sourceCpp(code=cSource, rebuild=T, showOutput=T, cacheDir=getwd(), cleanupCacheDir=F)



