###############################################################################################
#### The following script generates an R function for efficiently composing X'X matrices for
#### use in solving high dimension fixed effects regression problems.  Data sets and models
#### resulting in observation counts of up to 100,000,000 with multiple fixed effects of up
#### to 50,000 total levels have been tested with good (low) execution time results.
#### Two-level interactions between continuous and continuous variables, continuous variables
#### and fixed effects, and two fixed effects are supported.  Continuous variables and
#### interactions are optional.  At least one fixed effect must be specified.
####
#### The function accepts:
#### 1. A data frame containing dependent and independent variable data
####    a. Numeric continuous independent columns
####    b. Character fixed effect (categorical) columns
####    c. A numeric continuous dependent column
#### 2. Three vectors specifiyng which data frame columns are continuous independent,
####    fixed effect, and continuous dependent
#### 3. A vector of fixed effect reference levels, one value for each effect ("" for a
####    column instructs to use the minimum value observed in the data
#### 4. A vector specifying interactions, one element per interaction, each element a
####    colon separated pair of column names from the source data frame
####
#### The function produces:
#### 1. X'X, where X is the design matrix consisting of a vector of 1s (for the intercept),
####    the specified continuous vectors, expanded indicator vectors (one for each non-
####    reference level of each specified fixed effect), and interaction columns (pair-wise
####    interactions involving any of the specified continuous or fixed effect vectors, in
####    continuous X continuous, continuous X fixed effect, or fixed effect X fixed effect
####    combinations).  X'X columns are named as folows:
####    Intercept column ..... b0
####    Continuous columns ... column names from supplied data frame
####    Fixed effects ........ column name from data frame and level from values in
####                           respective data frame column, values in parentheses
####                           (reference levels excluded, example Race(E))
####    Interactions ......... interacting variable names and fixed effect levels
####                           separated by periods; examples:  Age.Sex(F),
####                           Sex.Race(F.E)
####                           NOTE THAT COLUMNS (ROWS) ARE INCLUDED IN X'X ONLY FOR
####                           LEVEL COMBINATIONS THAT APPEAR IN THE DATA, OTHERWISE
####                           A SINGULAR MATRIX OCCURS (EXAPLE:  IF OCCUPATION LEVELS
####                           ARE A, B, C, D AND LOCATION LEVELS ARE L1, L2, L3, L4,
####                           BUT NO OBSERVATIONS EXIST FOR OCCUPATION B IN LOCATION
####                           L4, THEN A COMPUTED INTERACTION COLUMN FOR B, L4 WOULD
####                           BE A VECTOR OF ZEROS AND CAUSE A LINEAR DEPENDENCY
####                           BETWEEN ALL X'X VECTORS - 0 TIMES OTHERS + 1 TIMES
####                           ZERO VECTOR = 0)
#### 2. X'Y, where Y is a numeric vector from the supplied data frame.  Omitted if Y
####    parameter="".
#### 3. Sparse, row-wise fixed effect indicator vectors.  One index row is generated for
####    each observation in the supplied data frame, each index row consisting of k
####    columns, one for each specified fixed effect, each index column containing the
####    (0-based) design column corresponding to the level (of the FE) for which the
####    observation is coded.  Reference levels appear as -1.  Sparse indices are useful
####    in efficient computation of robust and clustered standard errors (used by related
####    function feXTXcRobustSE().
####
#### Note that continuous variables (columns, vectors), a response (Y) vector, and an
#### interaction specification are optional.
#### 
#### Program outline: 
####
#### 1. Validate consistency of supplied data and parameters (ensure all specified columns
####    exist and are appropriate for specified role (numeric, fixed, existence of referencle
####    levels in data, etc.)
#### 2. Construct fixed effect level index vectors (similar to factors)
#### 3. Assemble interaction structures:
####    Continuous X continuous ....... a vector of row-wise products is appended to existing
####                                    continuous vectors
####    Fixed effect X fixed effect ... a vector of concatenated values from interacting
####                                    columns is appended to fixed effect columns, with
####                                    reference level equal to concatenated respective
####                                    reference levels (note that all observations coded
####                                    for one or both reference levels has its interaction
####                                    recoded as the interaction reference level) 
####    continuous X fixed effect ..... retain a recod of interacting columns, to be used
####                                    during X'X and X'Y construction
#### 4. Compose sparse fixed effect row index vectors
#### 5. Construct X'X and X'Y, with column (row) names composed from source data frame names,
####    fixed effect levels, and specified interaction configuration
#### 6. Construct result list with the following elements:
####    status ........ a two element vector, the first containing either "ok" or "Error" and
####                    the second containg additional information when an error occurs
####    call .......... vector containing supplied parameter values, each element named with 
####                    corresponding parameter (continuousCol, FECol, interactionCol, Y).
####    XTX ........... p X p numeric matrix, where p = the nmber of columns (variables) in
####                    the design = 1 + n_continuous + n_non_ref_FE_levels +
####                    n_non_ref_FE_interaction_levels
####    XTY ........... n X 1 vector, product of design X (with fixed effects expanded and
####                    interactions constructed) and response Y, n = number of observations
####    FE ............ sub-list of fixed effects structure, with elements:
####                    name ........ vector of FE names, from supplied data frame
####                    levels ...... sub-list of vector, one for each FE, of distinct data
####                                  values from supplied data, reference value is omitted
####                    refLevel .... derived reference levels, default ("") FERefLevel
####                                  parameter values converted to minimum data values,
####                                  should agree with first element of levels vector
####                                  note:  FE1 X FE2 ref level = dot separated FE ref levels
####                    nLevel ...... number of X columns occupied by FE or FE X FE interaction
####                    offsetX ..... column position in (fully expanded) design matrix of
####                                  first (non-reference) level of FE (or FE X FE
####                                  interaction), given continuous and FE parameters
####                                  positions are 0-based (b0 in col 0)
####                    obsFEColX ... n-length vector, element i giving the 0-based column
####                                  position in X corrresponding to the level for this FE
####                                  that observation i is encoded - note that FE X FE
####                                  interaction indicator columns are included
####    interaction ... sub-list with two elements:
####                    interactionCfg ... matrix with one row per specified interaction
####                                       col 0:  interaction type (cc, ff, or cf)
####                                       col 1:  interaction variable name 1
####                                       col 2:  interaction variable name 2
####                    interactionIdx ... matrix with one row per interaction, rows
####                                       correspond to interactionCfg rows
####                                       col 0:  position in continuousCol or FECol of
####                                               interaction variable 1
####                                       col 1:  position in continuousCol or FECol of
####                                               interaction variable 2
####                 
####
#### Jan 2018, Tom Balmat, Duke University Human Capital and Synthetic Data Projects
####
###############################################################################################

options(max.print=1000)      # number of elements, not rows
options(stringsAsFactors=F)
options(scipen=999999)

library(Rcpp)

setwd("C:\\Projects\\Statistics\\ModelsMethodsTests\\FixedEffectsRegression\\feXTX-C")

cSource <- "
  #include <Rcpp.h>
  #include <omp.h>
  // [[Rcpp::plugins(openmp)]]
  using namespace Rcpp;
  // [[Rcpp::export]]
  List feXTXc(DataFrame data,
              CharacterVector continuousCol=CharacterVector(0),
              CharacterVector FECol=CharacterVector(0),
              CharacterVector FERefLevel=CharacterVector(0),
              CharacterVector interactionCol=CharacterVector(0),
              CharacterVector Y=CharacterVector(0),
              LogicalVector retObsFEIndices=LogicalVector(0),
              int nCore=0) {

    // Parameters (note that all, except the source data frame, are vectors since
    // this is R's minimal variable structure):

    // data .............. data frame containing independent and dependent vectors.
    // continuousCol ..... Vector of names of continuous columns in data.  Omit this
    //                     parameter, or supply a 0-length vector if model does not
    //                     contain continuous variables.
    // FECol ............. Vector of names of fixed effect columns in data.  Fixed
    //                     effects must be specified.
    // FERefLevel ........ Fixed effect reference levels, one per FE, in order of FECol.
    // interactionCol .... Vector of colon separated pairs of data columns from which
    //                     to compose additional interaction columns.  Omit, or supply
    //                     0-length vector, if model does not contain interactions.
    //                     Example: \"X1:X2, X1:LaborCategory, Race:Zipcode\"
    // Y ................. Name of continuous response column in data.  Non-empty
    //                     vector causes computation and return of X'Y[1], where Y[1]
    //                     (first position in non-empty Y vector) is the data column
    //                     containing continuous response values.  X'Y is returned in
    //                     lReturn[XTY].
    // retObsFEIndices ... TRUE to save generated sparse FE design matrix column offsets
    //                     in output list elements FE[obsFEColX].  One vector for each
    //                     FE is returned, each with one element per observation,
    //                     each element indicating which column of the design (X, X'X)
    //                     corresponds to the FE and level that the observation is coded
    //                     for.  These are the 1 values of FE indicator columns in the
    //                     fully expanded design.  Since there is one vector per FE and
    //                     one element per vector per observation, these can be a large.
    //                     Omit by setting retObsFEIndices to an empty vector or a vector
    //                     with FALSE in its first position.  Indices are returned in
    //                     lReturn[FE][obsFEColX], a list of vectors, one per FE.
    // nCore ............. Maximum number of cores to use in parallel operations

    // Use long indices for high dimension addressing, but signed due to lack of support for
    // unsigned indices in Rcpp matrix objects
    long i, i2, j, k, k2, k3, m, m2, p, q, q2, t, t2, n2, iref;
    long n=data.nrow(), nFE=FECol.size(), nffInteraction, nFETotal, nlevFETotal;
    // Note that continuous variables are not required, ncontX may be 0
    long ncontX=continuousCol.size(), ncontXTotal, nccInteraction;
    // Interactions are not required, nInteraction may be 0
    long nInteraction=interactionCol.size(), ncfInteraction;
    long idxBase, idxBase2;
    int *lngPtr1, *lngPtr2;
    double s, *dblPtr, *yPtr, *XTYPtr;
    std::string a;
    char *d, d2[1000];
    char *ref1, *ref2, ref12[1000];
    // Note that Rcpp vectors can be set to vectors passed from the R environment, as in
    // b1=data[1].  Modifying elements of such vectors in the C environment modifies
    // elements in the corresponding vector in the R environment. 
    CharacterVector b1, b2, bn(n), lev, c1(1), c2(1);
    NumericVector x1, x2, x3(1), xlev;
    IntegerVector z;
    CharacterVector stat=CharacterVector::create(\"ok\", \"\");
    CharacterVector xNames=data.names();
    // Interaction Cfg matrix:
    // col 0:  type of interaction (cc = cont X cont, ff = FE X FE, cf = cont X FE)
    // col 1:  interaction column (name from source data frame)
    // col 2:  interaction column (name from source data frame)
    CharacterMatrix interactionCfg(nInteraction, 3);
    // Interaction variable index matrix
    // ith row corresponds to ith row of InteractionCfg
    // col 0:  0-based index into continuous or FE cols of interactionCfg col 1
    // col 1:  0-based index into continuous or FE cols of interactionCfg col 2
    int interactionIdx[nInteraction][2];
    // Create result list with empty elements
    // Note that list elements can be created dynamically - at any time during execution,
    // the instruction lResult(\"el\")=CharacterVector::create(\"a\", \"b\"); will
    // append a new element to lResult called 'el' containing the vector ('a', 'b')
    List lResult=List::create(
      _[\"status\"],
      _[\"call\"]=List::create(
        _[\"continuousCol\"]=continuousCol,
        _[\"FECol\"]=FECol,
        _[\"interactionCol\"]=interactionCol),
      _[\"XTX\"],
      _[\"XTY\"],
      _[\"FE\"]=List::create(
        _[\"name\"]=FECol,
        _[\"levels\"]=List(FECol.size()),
        _[\"refLevel\"]=CharacterVector(FECol.size()),
        _[\"nLevel\"],
        _[\"offsetX\"],
        _[\"obsFEColX\"]),
      _[\"interaction\"]=List::create(
        _[\"interactionCfg\"]=CharacterMatrix(nInteraction, 3),
        _[\"interactionIdx\"]=IntegerMatrix(nInteraction, 2))
    );

    // Set the max number of cores to use
    if(nCore>0)
      omp_set_num_threads(nCore);

    ////////////////////////////////////////////////////////////////////////////////////////
    //// STEP I.  VALIDATE PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////////////

    // Verify that all specified columns appear in supplied data and are of appropriate type
    // Continuous cols
    for(i=0; i<ncontX && stat(0)==\"ok\"; i++) {
      // match() returns 1-based column indicators
      k=match(as<CharacterVector>(continuousCol(i)), xNames)[0];
      if(k!=NA_INTEGER) {
        if(TYPEOF(data[k-1])!=INTSXP && TYPEOF(data[k-1])!=REALSXP) {
          stat(0)=\"Error\";
          stat(1)=\"Non-numeric column specified as continuous:  \";
          stat(1)+=continuousCol(i);
        }
      } else {
        stat(0)=\"Error\";
        stat(1)=\"Unrecognized continuous col in supplied data frame:  \";
        stat(1)+=continuousCol(i);
      }
    }
    if(stat(0)==\"ok\")
      // Verify existence of FE cols
      for(i=0; i<nFE && stat(0)==\"ok\"; i++)
        if(match(as<CharacterVector>(FECol(i)), xNames)[0]==NA_INTEGER) {
          stat(0)=\"Error\";
          stat(1)=\"Unrecognized fixed effect col:  \";
          stat(1)+=FECol(i);
        }
    if(stat(0)==\"ok\")
      // Verify number of reference levels
      if(FERefLevel.size()!=FECol.size()) {
        stat(0)=\"Error\";
        stat(1)=\"Number of reference levels and fixed effects are different\";
      }
    if(stat(0)==\"ok\")
      // Verify existence of interaction columns
      // Construct matrix of interaction types and pairs of cols
      for(i=0; i<nInteraction && stat(0)==\"ok\"; i++) {
        // Locate colon
        a=interactionCol(i);
        // assignment of length, here, avoids unsigned int/long conversion error in for()
        n2=a.size();
        k=-1;
        for(j=0; j<n2 && k<0; j++)
          if(a.substr(j, 1)==\":\")
            k=j;
        if(k>0 && k<n2-1) {
          // Colon found
          // Parse text on left and right of colon
          c1(0)=a.substr(0, k);
          c2(0)=a.substr(k+1, a.size()-k-1);
          // Validate existence of interaction cols as continuous or FE
          // match() results are 1-based
          if((k=match(c1, continuousCol)[0])!=NA_INTEGER) {
            // Left side is continuous; save and validate right side
            interactionCfg(i,1)=c1(0);
            interactionIdx[i][0]=k-1;
            if((k=match(c2, continuousCol)[0])!=NA_INTEGER) {
              // Right side is also continuous; save
              interactionCfg(i,0)=\"cc\";
              interactionCfg(i,2)=c2(0);
              interactionIdx[i][1]=k-1;
            } else if((k=match(c2, FECol)[0])!=NA_INTEGER){
                // Right side is FE; save
                interactionCfg(i,0)=\"cf\";
                interactionCfg(i,2)=c2(0);
                interactionIdx[i][1]=k-1;
            } else {
              // Right side unrecognized
              stat(0)=\"Error\";
              stat(1)=\"Unrecognized interaction column (\";
              stat(1)+=c2(0);
              stat(1)+=\") - does not appear in continuous or FE list\";
            }
          } else if((k=match(c1, FECol)[0])!=NA_INTEGER) {
            // Left side is FE; save and validate right side
            if((k2=match(c2, continuousCol)[0])!=NA_INTEGER) {
              // Right side is continuous; save, continuous col first
              interactionCfg(i,0)=\"cf\";
              interactionCfg(i,1)=c2(0);
              interactionCfg(i,2)=c1(0);
              interactionIdx[i][0]=k2-1;
              interactionIdx[i][1]=k-1;
            } else if((k2=match(c2, FECol)[0])!=NA_INTEGER){
              // Right side is FE; save
              interactionCfg(i,0)=\"ff\";
              interactionCfg(i,1)=c1(0);
              interactionCfg(i,2)=c2(0);
              interactionIdx[i][0]=k-1;
              interactionIdx[i][1]=k2-1;
            } else {
              // Right side unrecognized
              stat(0)=\"Error\";
              stat(1)=\"Unrecognized interaction column (\";
              stat(1)+=c2(0);
              stat(1)+=\") - does not appear in continuous or FE list\";
            }
          } else {
            // Left side unrecognized
            stat(0)=\"Error\";
            stat(1)=\"Unrecognized interaction column (\";
            stat(1)+=c1(0);
            stat(1)+=\") - does not appear in continuous or FE list\";
          }
        } else if(k==0) {
          stat(0)=\"Error\";
          stat(1)=\"Invalid interaction specification, missing first column in \";
          stat(1)+=a;
        } else if(k==n2-1) {
          stat(0)=\"Error\";
          stat(1)=\"Invalid interaction specification, missing second column in \";
          stat(1)+=a;
        } else {
          stat(0)=\"Error\";
          stat(1)=\"Invalid interaction specification, missing colon in \";
          stat(1)+=a;
        }
      }
    if(nInteraction>0 && stat(0)==\"ok\") {
      // Save interaction matrices
      as<List>(lResult(\"interaction\"))(\"interactionCfg\")=interactionCfg;
      for(i=0; i<nInteraction; i++)
        for(j=0; j<2; j++)
          as<IntegerMatrix>(as<List>(lResult(\"interaction\"))(\"interactionIdx\"))(i,j)=
            interactionIdx[i][j];
    }
    colnames(as<List>(lResult(\"interaction\"))(\"interactionCfg\"))=
      CharacterVector::create(\"type\", \"v1\", \"v2\");
    colnames(as<List>(lResult(\"interaction\"))(\"interactionIdx\"))=
      CharacterVector::create(\"v1\", \"v2\");
    if(stat(0)==\"ok\" && Y.size()>0) {
      // Verify existence and numeric type of Y column
      k=match(as<CharacterVector>(Y(0)), xNames)[0];
      if(k!=NA_INTEGER) {
        a=Y(0);
        if(TYPEOF(data(a))!=INTSXP && TYPEOF(data(a))!=REALSXP) {
          stat(0)=\"Error\";
          stat(1)=\"Non-numeric column specified for Y parameter:  \";
          stat(1)+=Y(0);
        }
      } else {
        stat(0)=\"Error\";
        stat(1)=\"Unrecognized column specified for Y in supplied data frame:  \";
        stat(1)+=Y(0);
      }
    }

    if(stat(0)==\"ok\") {

      //////////////////////////////////////////////////////////////////////////////////////
      // STEP II.
      // a. COMPOSE INTERNAL ARRAYS FOR PERFORMANCE AND TO AVOID UPDATE OF SOURCE DATA
      // b. FACTOR FIXED EFFECTS
      // c. INTERPRET INTERACTION REQUEST
      // d. COMPOSE CONTINUOUS X CONTINUOUS AND FE X FE INTERACTION COLUMNS
      //////////////////////////////////////////////////////////////////////////////////////

      // Enumerate interactions
      nccInteraction=0;
      nffInteraction=0;
      ncfInteraction=0;
      for(i=0; i<nInteraction; i++) {
        if(interactionCfg(i,0)==\"cc\")
          nccInteraction++;
        if(interactionCfg(i,0)==\"ff\")
          nffInteraction++;
        if(interactionCfg(i,0)==\"cf\")
          ncfInteraction++;
      }

      // Compute total number of continuous and FE variables
      // Continuous variables will include continuous X continuous interactions
      // Fixed effects will include FE X FE interactions 
      ncontXTotal=ncontX+nccInteraction;
      nFETotal=nFE+nffInteraction;

      // Construct internal C array of continuous columns from source vectors
      // This may seem an unnecessary copy of existing data, since the R matrix is
      // addressable here using pointers (contX[i]=&x1[0], where x1=data[continuousCol(i)])
      // but the method is unstable for certain high dimension data sets (crash!)
      // For continuous X continuous interactions, create new vectors, compute intreraction
      // levels, and direct pointers
      // Note that a performance improvement of 3 or more can be gained by using internal
      // C matrices and pointers instead of a corresponding Rcpp NumericMatrix
      // A NumericMatrix is a collection of column-wise vectors, while  C matrix is a
      // collection of row-wise vectors
      // Row-wise access of C matrix element is efficient, possibly prompting the compiler
      // to implement loop unrolling, while row-wise access of NumericMatrix matrix
      // elements is inefficeient (likewise, column-wise access of a C matrix is
      // inefficient, while it is unknown whether the compiler treats, for instance,
      // A(i,j) as an element within a C array)
      // Use of pointers to vectors reduces access to base objects and should give the
      // compiler maximum flexibiliity in choosing an optimization scheme
      // Also, Rcpp executables habitually crash when high dimension C arrays are
      // defined, while use of pointers to high dimension vectors is stable
      double **contX = new double *[ncontXTotal];
      for(i=0; i<ncontX; i++) {
        a=continuousCol(i);
        x1=data[a];
        //contX[i]=&x1[0]; 
        contX[i] = new double [n];
        for(j=0; j<n; j++)
          contX[i][j]=x1(j);
      }
      // Compose continuous X continuous interactions
      m=-1;
      for(i=0; i<nInteraction; i++)
        if(interactionCfg(i,0)==\"cc\") {
          m++;
          // Identify vectors
          a=interactionCfg(i,1);
          x1=data[a];
          a=interactionCfg(i,2);
          x2=data[a];
          // Create new vector, multiply continuous elements, and copy
          contX[ncontX+m] = new double [n];
          for(j=0; j<n; j++)
            contX[ncontX+m][j]=x1[j]*x2[j];
        }

      // Factor each supplied fixed effect vector using corresponding reference level
      // Compose FE X FE interactions by concatenating row-wise levels from both FEs

      // Create array of level index vectors (factors)
      long **feX = new long *[nFETotal];

      // Create vector of counts of levels represented in X (and X'X) by FE and
      // FE X FE interactions
      long nlevFE[nFETotal];

      // Create vector, in reult list, to contain number of levels for FEs and
      // FE X FE interactions
      as<List>(lResult(\"FE\"))(\"nLevel\")=IntegerVector(nFETotal);

      // Create list to contain FE X FE non-reference level interactions
      // These are used to construct column names for the result X'X matrix
      List ffInteractionLabel=List::create(
        _[\"FE\"]=CharacterMatrix(nffInteraction, 2),
        _[\"level\"]=List(nffInteraction));

      // Factor FEs
      // As tempting as a parallel algorithm is, this particular one does not produce
      // valid results.  Although lev is declared private, when it is assigned the result
      // of unique() in a parallel for loop, each private version contains duplicated
      // values (on inspection, the generated lev=unique(b) vectors have repeated values).
      // Perhaps Rcpp objects (lev, here) are not kept private.  An attempt was made to
      // create a list of lev vectors and specify lev(i) in the for loop, but this
      // presented problems with passing individual vectors to match().
      // Create a within FE status matrix, record status of each FE, and report
      // aggregate results after all FEs processed [this is necessary due to independent
      // parallel processes simultaneously generating status info, along with associated
      // compilation errors on attempt to terminate on error using using
      // for(i=0; i<nFETotal && stat(0)==\"ok\"; i++)].
      // stat2 cols:  0=source, 1=status, 2=message
      CharacterMatrix stat2(nFETotal, 3);
      ////#pragma omp parallel for private(i, a, b, lev, iref, j, z)
      for(i=0; i<nFE; i++) {
        // Create integer level (factor) vector for current FE
        feX[i] = new long [n];
        stat2(i,0)=\"FE\";
        stat2(i,1)=\"ok\";
        // Address FE col, extract and sort unique levels, factor
        a=FECol(i);
        if(TYPEOF(data[a])==INTSXP || TYPEOF(data[a])==REALSXP) {
          // FE is numeric, do not convert to character
          // This leaves level in original mode and does not incur an efficiency
          // penalty for conversion
          x1=data[a];
          xlev=unique(x1);
          xlev.sort();
          // Rearrange levels, placing reference level first
          // Default (first value in alphabetic order) is specified with an empty string
          if(FERefLevel(i)!=\"\") {
            // Identify position of ref level in levels vector
            // Note that positions are 1-based
            // Convert supplied reference level to numeric
            // d points to first non float character, 0-length string if none
            x3(0)=strtod(FERefLevel(i), &d);
            if(strlen(d)==0) {
              iref=match(x3, xlev)[0];
              if(iref!=NA_INTEGER) {
                // Move reference level to first position of levels vector
                for(j=iref-1; j>0; j--)
                  xlev(j)=xlev(j-1);
                xlev(0)=x3(0);
              } else {
                stat2(i,1)=\"Error\";
                stat2(i,2)=\"No observations with specified FE reference level (\";
                stat2(i,2)+=a;
                stat2(i,2)+=\", \";
                stat2(i,2)+=FERefLevel(i);
                stat2(i,2)+=\")\";
              }
            } else {
              stat2(i,1)=\"Error\";
              stat2(i,2)=\"Non-numeric format in reference level for numeric fixed effect (\";
              stat2(i,2)+=a;
              stat2(i,2)+=\", \";
              stat2(i,2)+=FERefLevel(i);
              stat2(i,2)+=\") [omit non-numerals and spaces]\";
            }
          }
          if(stat2(i,1)==\"ok\") {
            // Reference level is in first position of levels vector, others are ordered alphabetically.
            // Convert character values into corresponding 1-based position in levels vector.
            z=match(as<NumericVector>(x1), xlev);
            for(j=0; j<n; j++)
              feX[i][j]=z[j];
            // Save count of non-reference levels
            nlevFE[i]=xlev.size()-1;
            // Save levels in return list, omit reference level
            as<List>(as<List>(lResult(\"FE\"))(\"levels\"))(i)=tail(xlev, nlevFE[i]);
            as<IntegerVector>(as<List>(lResult(\"FE\"))(\"nLevel\"))(i)=nlevFE[i];
            as<CharacterVector>(as<List>(lResult(\"FE\"))(\"refLevel\"))(i)=xlev(0);
          }
        } else {
          // FE is non-numeric, treat as character
          b1=data[a];
          lev=unique(b1);
          lev.sort();
          // Rearrange levels, placing reference level first
          // Default (first value in alphabetic order) is specified with an empty string
          if(FERefLevel(i)!=\"\") {
            // Identify position of ref level in levels vector
            // Note that positions are 1-based
            iref=match(as<CharacterVector>(FERefLevel(i)), lev)[0];
            if(iref!=NA_INTEGER) {
              // Move reference level to first position of levels vector
              for(j=iref-1; j>0; j--)
                lev(j)=lev(j-1);
              lev(0)=FERefLevel(i);
            } else {
              stat2(i,1)=\"Error\";
              stat2(i,2)=\"No observations with specified FE reference level (\";
              stat2(i,2)+=a;
              stat2(i,2)+=\", \";
              stat2(i,2)+=FERefLevel(i);
              stat2(i,2)+=\")\";
            }
          }
          if(stat2(i,1)==\"ok\") {
            // Reference level is in first position of levels vector, others are ordered alphabetically.
            // Convert character values into corresponding 1-based position in levels vector.
            z=match(b1, lev);
            for(j=0; j<n; j++)
              feX[i][j]=z[j];
            // Save count of non-reference levels
            nlevFE[i]=lev.size()-1;
            // Save levels in return list, omit reference level
            as<List>(as<List>(lResult(\"FE\"))(\"levels\"))(i)=tail(lev, nlevFE[i]);
            as<IntegerVector>(as<List>(lResult(\"FE\"))(\"nLevel\"))(i)=nlevFE[i];
            as<CharacterVector>(as<List>(lResult(\"FE\"))(\"refLevel\"))(i)=lev(0);
          }
        }
      }
      // Combine status results (errors only)
      for(i=0; i<nFE; i++)
        if(stat2(i,1)!=\"ok\") {
          stat(0)=\"Error\";
          stat(1)+=stat2(i,2);
          stat(1)+=\" ........ \";
        }

      if(stat(0)==\"ok\") {
        // Factor FE X FE interactions
        // Construct a new FE column using product (concatenation) of two FEs
        // Use m to indicate current ff interaction being processed
        m=-1;
        for(i=0; i<nInteraction; i++)
          if(interactionCfg(i,0)==\"ff\") {
            // Advance offset to current FE position
            m++;
            // Create vector for current FE
            feX[nFE+m] = new long [n];
            stat2(nFE+m,0)=\"interaction\";
            stat2(nFE+m,1)=\"ok\";
            // Identify interacting FEs
            a=interactionCfg(i,1);
            b1=data[a];
            a=interactionCfg(i,2);
            b2=data[a];
            // Retrieve reference levels for both FEs
            // For efficiency, use char pointers to previously saved FE reference levels
            ref1=as<CharacterVector>(as<List>((lResult)(\"FE\"))(\"refLevel\"))(interactionIdx[i][0]);
            ref2=as<CharacterVector>(as<List>((lResult)(\"FE\"))(\"refLevel\"))(interactionIdx[i][1]);
            // Compose interaction reference level from interacting FE reference levels
            // Note that an interaction reference level is the concatenation of ref levels
            // for respective interaction FEs - all observations with reference levels in
            // either FE are assigned the concatenated ref level value
            // Limit to 490 chars of each level
            for(k=0; ref1[k]!=0 && k<490; k++)
              ref12[k]=ref1[k];
            ref12[k++]=46;
            for(k2=0; ref2[k2]!=0 && k2<490; k2++)
              ref12[k++]=ref2[k2];
            ref12[k]=0;
            // Iterate through all observations (rows of data) and concatenate FE levels
            // Convert combinations with either FE at ref level to interaction ref level
            for(k=0; k<n; k++)
              if(strcmp(b1(k), ref1)!=0 && strcmp(b2(k), ref2)!=0) {
                // Note that, to preserve the original data, a copy is made and altered 
                // Also, at most, 490 characters of each level are retained
                // Copy level of first FE
                d=b1(k);
                for(k2=0; d[k2]!=0 && k2<490; k2++)
                  d2[k2]=d[k2];
                // Include delimiter
                d2[k2++]=46;
                // Copy level of second FE
                d=b2(k);
                for(k3=0; d[k3]!=0 && k3<490; k3++)
                  d2[k2++]=d[k3];
                // Terminate the string
                d2[k2]=0;
                // Save concatenated levels
                bn(k)=d2;
              } else {
                bn(k)=ref12;
              }
            // Index levels (factor)
            lev=unique(bn);
            lev.sort();
            // Rearrange levels, placing reference level first
            // Identify position of ref level in levels vector
            // Note that positions are 1-based
            iref=match(CharacterVector::create(ref12), lev)[0];
            if(iref!=NA_INTEGER) {
              // Move reference level to first position of levels vector
              for(j=iref-1; j>0; j--)
                lev(j)=lev(j-1);
              lev(0)=ref12;
            } else {
              stat2(nFE+m,1)=\"Error\";
              stat2(nFE+m,2)=\"No observations with specified FE reference level (\";
              stat2(nFE+m,2)+=interactionCfg(i,1);
              stat2(nFE+m,2)+=\":\";
              stat2(nFE+m,2)+=interactionCfg(i,2);
              stat2(nFE+m,2)+=\", \";
              stat2(nFE+m,2)+=ref12;
              stat2(nFE+m,2)+=\")\";
            }
            // Reference level is in first position of levels vector, others are ordered alphabetically
            // Convert character values into corresponding 1-based position in levels vector
            z=match(bn, lev);
            for(j=0; j<n; j++)
              feX[nFE+m][j]=z[j];
            // Save count of non-reference levels for use in X'X construction
            nlevFE[nFE+m]=lev.size()-1;
            as<IntegerVector>(as<List>(lResult(\"FE\"))(\"nLevel\"))(nFE+m)=lev.size()-1;
            // Save level combinations for column names
            // Note that levels are not generated for combinations that do not appear in the data
            // To do so would cause a singular X'X
            // Retain FEs and levels here to avoid a rescan of the data when constructing names
            as<CharacterMatrix>(ffInteractionLabel(\"FE\"))(m, 0)=interactionCfg(i,1);
            as<CharacterMatrix>(ffInteractionLabel(\"FE\"))(m, 1)=interactionCfg(i,2);
            as<List>(ffInteractionLabel(\"level\"))(m)=lev;
          }
        // Combine status results (errors only)
        for(i=0; i<nFETotal; i++)
          if(stat2(i,0)==\"interaction\" && stat2(i,1)!=\"ok\") {
            stat(0)=\"Error\";
            stat(1)+=stat2(i,2);
            stat(1)+=\" ........ \";
          }
      }

      if(stat(0)==\"ok\") {

        ////////////////////////////////////////////////////////////////////////////////////////////
        // STEP III.  CONSTRUCT X'X
        ////////////////////////////////////////////////////////////////////////////////////////////

        // Compose X'X from continuous and fixed effect columns using sparse fixed effect row index
        // vectors.  Continuous columns include continuous X continuous interactionsand fixed effect
        // columns include FE X FE interactions.  Continuous X FE interactions are composed as
        // needed in X'X row and column operations.

        // Requirements and assumptions:
        // 1. Continuous independent cols appear in the contX matrix in the same order as
        //    corresponding column names appear in the continuousCol parameter.
        // 2. Continuous X continous interactions are composed (as row-wise products of interacting
        //    columns) and appear in contX matrix columns following main continuous effects in order
        //    of continous interactions specified in the interactionCol parameter.
        // 3. Fixed effect numeric level indices (factor vectors) appear in the feX matrix, one
        //    column for each effect, in the same order as corresponding names appear in the FECol
        //    parameter.  Vector j of feX should resemble the result of the R instruction
        //    as.integer(relevel(factor(data[,FECol[j]), ref=FERefLevel[j])).
        // 4. Reference levels for all fixed effects are coded as 1.
        // 5. nlevFE vector exists with one element per FE, the ith element containing the
        //    number of non-reference levels in the data for FE i.
        // 6. FE X FE interactions appear in the feX matrix following the main FEs in the same
        //    order as FE interactions appear in the interactionCol parameter.  Reference level
        //    (value 1) appears for all observations where either interaction FE is coded for
        //    its respective reference level.
        // 7. Continuous X fixed effect interactions are specified in the cfInteraction matrix, which
        //    contains one row for each interaction and two columns per row: column 0 for the 0-based
        //    column index into the contX matrix of the continuous interacting variable, column 1
        //    for the 0-based index into the feX matrix of the fixed effect interacting variable.
        //    cfInteraction is constructed from the 'cf' rows of the interactionCfg matrix
        //    constructed in a previous step.

        // The result of this step is the p X p matrix X'X, where X consists of
        // an intercept column (all 1s), continuous columns specified in contX, interactions
        // of continuous cloumns with continuous columns, fixed effect columns (in feX)
        // expanded into indicator columns (one for each non-reference level), interaction of
        // fixed effects, and interactions of continuous columns with fixed effect columns 
        // the dimension, p, of X'X is 1 (for b0) + the number of continuous columns
        // (recall that this includes continuous interaction products assembled prior to call) +
        // the sum of the number of non-reference levels of all fixed effects (includes FE
        // interaction vectors assembled prior to call) + the sum of non-reference levels
        // of all fixed effects specified in cfInteraction matrix.

        // To efficiently compute elements of X'X involving fixed effects (products of FE
        // transpose rows with FE or continuous X FE interactions), indices into rows of X'
        // corresponding to FEs are constructed.  Explain the intesection of k index.

        // Enumerate total number of X (and X'X) columns occupied by FEs
        nlevFETotal=0;
        for(i=0; i<nFETotal; i++)
          nlevFETotal+=nlevFE[i];
        // Record, for each FE, the column position in X (and X'X) of its first
        // (non-reference) level.  These are used as a basis for offset indexing
        // of FEs and levels in X'X.
        long jdxFE[nFETotal];
        jdxFE[0]=1+ncontXTotal;
        for(i=1; i<nFETotal; i++)
          jdxFE[i]=jdxFE[i-1]+nlevFE[i-1];

        // Record, in return list, 1-base offsets of first col in X for each FE
        as<List>(lResult(\"FE\"))(\"offsetX\")=IntegerVector(nFETotal);
        for(i=0; i<nFETotal; i++)
          as<IntegerVector>(as<List>(lResult(\"FE\"))(\"offsetX\"))(i)=jdxFE[i];

        // Compute dimension of X'X for the intercept, continuous, FE,
        // continuous X continuous interactions, and FE X FE interactions
        p=1+ncontXTotal+nlevFETotal;

        // Configure continuous X FE interactions.  Each row of cfInteraction
        // contains:
        // col 0 ... the index (position in continuousCol parameter) of the
        //           interacting continuous variable
        // col 1 ... the index (in FECol) of the interacting fixed effect
        // col 2 ... the col offset in X (and X'X) of the first (non-reference)
        //           level of the fixed effect
        // Include in X'X dimension the number of non-reference levels for each
        // interaction FE
        long cfInteraction[ncfInteraction][3];
        for(i=0, j=0; i<nInteraction && j<ncfInteraction; i++)
          if(interactionCfg(i,0)==\"cf\") {
            // Record continuous and FE var indices
            cfInteraction[j][0]=interactionIdx[i][0];
            cfInteraction[j][1]=interactionIdx[i][1];
            // Compute X col offset of interaction
            // First one immediately follows FEs, others follow preceding interaction
            if(j==0)
              cfInteraction[j][2]=jdxFE[0]+nlevFETotal;
            else
              cfInteraction[j][2]=cfInteraction[j-1][2]+nlevFE[cfInteraction[j-1][1]];
            p+=nlevFE[cfInteraction[j][1]];
            j++;
          }

        // Create empty X'X array.  Use a C array for efficiency and stability,
        // then convert to a NumericMatrix prior to exit.
        double ** XTX = new double *[p];
        for(i=0; i<p; i++) {
          XTX[i] = new double [p];
          memset(XTX[i], 0, sizeof(double)*p);
        }

        // Create sparse row index array, one vector for each row of the design matrix.
        // The ith vector contains column indices corresponding to columns of X (and X'X)
        // for the levels of each FE for observation i.  Use pointer array to avoid crash
        // with high deimension FEs.
        long **ridx = new long *[n];

        // Compose sparse row index vectors for fixed effects.  These are used for
        // efficient computation of the products of intersecting non-zero row positions
        // of X' with non-zero column positions of X.  Since high dimension fixed
        // effect columns represent the vast majority of columns in X and are of low
        // density (are sparse) and only intersecting row and column elements are
        // evaluated, use of these row indices is primarily responsible for the overall
        // efficiency of the present X'X agorithm.  Note that there is one index row
        // per design (X) row and one column for each FE.  Index values are columns
        // of the design matrix corresponding to non-zero levels.
        for(i=0; i<n; i++)
          ridx[i] = new long [nFETotal];
        #pragma omp parallel for private(j, k)
        for(j=0; j<nFETotal; j++)
          for(k=0; k<n; k++)
            // Save FE level design column index
            // Convert to 0-base by subtracting 2 (ref level is 1, first non-ref level is 2)
            // Flag reference levels as -1
            if(feX[j][k]>1)
              ridx[k][j]=jdxFE[j]+feX[j][k]-2;
            else
              ridx[k][j]=-1;

        // Begin X'X construction

        // Construct diagonal and upper triangle of X'X by multiplying indexed rows
        // of X' and indexed columns of X

        // X'X constant row, sums of X column values
        // row 0, b0
        XTX[0][0]=n;

        // X'X constant row, continuous vars
        for(i=0; i<ncontXTotal; i++) {
          s=0;
          for(j=0; j<n; j++)
            s+=contX[i][j];
          XTX[0][1+i]=s;
        }

        // X'X constant row, FEs, number of observation by FE and (non-reference) level.
        // Iterate through all row indices, accumulate number of indexed rows (i) by column (j)
        // Note the parallelization along FEs
        // This is effective (works) since X'X column indices (in ridx) are disjoint between FEs
        // If multiple FEs indexed a common X'X column then possible simultaneous update
        // contention could arise (example:  if ridx[25][0]=14 and ridx[37][2]=14 while k(FE0)=25
        // and k(FE2)=37 simultaneously, then X'X[0][14] is targeted by two assignments)
        // WARNING!!! No error or warning is generated when contention exists as described - the
        // updated values are simply incorrect (presumably due to simultaneous process reading a
        // value and each executing addition with no knowledge of the others' operations)
        #pragma omp parallel for private(j, k)
        for(j=0; j<nFETotal; j++)
          for(k=0; k<n; k++)
            if(ridx[k][j]>=0)
              XTX[0][ridx[k][j]]++;
        // X'X constant row, cf interactions
        // Accumulate sum of indexed continuous values (i) by FE and level (XTX col j)
        // Update X'X positions for FE level using interaction offset column
        #pragma omp parallel for private(t, m, q, idxBase, k)
        for(t=0; t<ncfInteraction; t++) {
          // Retrieve continuous and FE indices.
          m=cfInteraction[t][0];
          q=cfInteraction[t][1];
          // Adjust X'X update offset for interaction by the offset of the interaction FE, since
          // ridx indices are X'X columns for the interaction FE
          idxBase=cfInteraction[t][2]-jdxFE[q];
          // Accumulate continuous values into XTX col for indexed level of interaction FE
          for(k=0; k<n; k++)
            if(ridx[k][q]>=0)
              XTX[0][idxBase+ridx[k][q]]+=contX[m][k];
        }

        // X'X continuous rows
        for(i=0; i<ncontXTotal;  i++) {
          // Continuous X continuous, sum of row-wise products
          // i2 indicates current X'X row being updated, begin with first continuous row
          i2=1+i;
          #pragma omp parallel for private(j, k, s)
          for(j=i; j<ncontXTotal; j++) {
            s=0;
            for(k=0; k<n; k++)
              s+=contX[i][k]*contX[j][k];
            XTX[i2][1+j]=s;
          }
          // Continuous X FE, sum of continuous values corresponding to FE indices
          // Note parallel indexing (k within j) along disjoint FEs (index j)
          #pragma omp parallel for private(j, k)
          for(j=0; j<nFETotal; j++)
            for(k=0; k<n; k++)
              if(ridx[k][j]>=0)
                XTX[i2][ridx[k][j]]+=contX[i][k];
          // Continuous X cf interactions
          // Iterate through all interactions
          // Accumulate sum of product of continuous values corresponding to interaction FE indices
          #pragma omp parallel for private(t, m, q, idxBase, k)
          for(t=0; t<ncfInteraction; t++) {
            // Retrieve continuous and FE indices
            m=cfInteraction[t][0];
            q=cfInteraction[t][1];
            // Adjust X'X update offset for interaction by the offset of the interaction FE, since
            // ridx indices are X'X columns for the interaction FE
            idxBase=cfInteraction[t][2]-jdxFE[q];
            for(k=0; k<n; k++)
              if(ridx[k][q]>=0)
                XTX[i2][idxBase+ridx[k][q]]+=contX[i][k]*contX[m][k];
          }
        }

        // X'X fixed effect rows
        // FE X FE, count of intersecting indices
        // Accumulate X' (i) row times X col (j) products into corresponding i,jth
        // position of X'X.  Note that the observation indices appearing in both row i
        // and col j form the intersection of non-zero elements and are the only ones
        // that affect the resulting X'X.  Efficiency results from evaluating products
        // of intersecting elements only.  Also, FE X FE products are necessarily 1, so
        // multiplication is unnecessary.  All combinations of products in row index i
        // form a single row, column product and must be evaluated.  Since the upper
        // triangle of X'X is being generated, evaluation is limited to rows and columns
        // such that i<j.  Within-FE diagonal sums are omitted since each equals the
        // corresponding column value from row 0 (these are copied later).
        // Note that, although simultaneous parallel processes may use a common j,
        // i, j pairs are distinct between parallel processes
        // This gives distinct ridx[k][i] and ridx[k][j] pairs (X'X coordinates)
        // and guarantees that simultaneous update of a single X'X element is avoided
        #pragma omp parallel for private(i, j, k)
        for(i=0; i<nFETotal; i++)
          for(j=i+1; j<nFETotal; j++)
            for(k=0; k<n; k++)
              // Omit if either FE level is reference
              if(ridx[k][i]>=0 && ridx[k][j]>=0)
                  XTX[ridx[k][i]][ridx[k][j]]++;
        // FE X cf interactions
        // Iterate through all interactions
        // Accumulate sum of continuous values corresponding to intersection of FE indices
        #pragma omp parallel for private(t, m, q, idxBase, k, k2)
        for(t=0; t<ncfInteraction; t++) {
          // Retrieve continuous and FE indices.
          m=cfInteraction[t][0];
          q=cfInteraction[t][1];
          // Adjust X'X update offset for interaction by the offset of the interaction FE, since
          // ridx indices are X'X columns for the interaction FE
          idxBase=cfInteraction[t][2]-jdxFE[q];
          for(k=0; k<n; k++)
            if(ridx[k][q]>=0)
              // Accumulate continuous values for intersecting FE index positions.
              for(k2=0; k2<nFETotal; k2++)
                if(ridx[k][k2]>0)
                  XTX[ridx[k][k2]][idxBase+ridx[k][q]]+=contX[m][k];
        }

        // X'X cf interaction products
        // Iterate through all interactions.  Accumulate sum of both continuous values
        // corresponding to intersection of FE indices.  Since upper triangle is being composed,
        // evaluate, for each interaction, the interaction and all subsequent interactions.
        #pragma omp parallel for private(t, t2, m, m2, q, q2, idxBase, idxBase2, k)
        for(t=0; t<ncfInteraction; t++)
          for(t2=t; t2<ncfInteraction; t2++) {
            // Retrieve continuous and FE indices.
            m=cfInteraction[t][0];
            q=cfInteraction[t][1];
            m2=cfInteraction[t2][0];
            q2=cfInteraction[t2][1];
            // Adjust X'X update offset for interaction by the offset of the second interaction
            // FE, since ridx indices are X'X columns for that interaction
            idxBase=cfInteraction[t][2]-jdxFE[q];
            idxBase2=cfInteraction[t2][2]-jdxFE[q2];
            for(k=0; k<n; k++)
              // Ignore if either index a reference level
              if(ridx[k][q]>=0 && ridx[k][q2]>=0)
                XTX[idxBase+ridx[k][q]][idxBase2+ridx[k][q2]]+=contX[m][k]*contX[m2][k];
          }

        ////////////////////////////////////////////////////////////////////////////////////////
        //// STEP IV.  PACKAGE RESULTS
        ////////////////////////////////////////////////////////////////////////////////////////

        // Create list-embedded R matrix to contain X'X elements
        // Initialize row and col pointers to matrix[0,0] and use offsets for update
        // Note that Rcpp matrices are store in column major order (col 0 elements are
        // contigous in memory, followed by col 2, ...) while the C vectors each contain
        // elements for a single row
        // Note, also, that diagonals of fixed effect rows/cols were not computed, since
        // they equal values in corresponding positions of row 1
        lResult(\"XTX\")=NumericMatrix(p,p);
        dblPtr=&as<NumericMatrix>(lResult(\"XTX\"))(0,0);
        // Copy upper triangle to upper tri and transpose of upper tri to lower tri
        //#pragma omp parallel for private(i, j)
        for(i=0; i<p; i++) {
          for(j=i+1; j<p; j++) {
            // Copy row i, beginning at col i+1, to row i and col i
            // Because we use a pointer to doubles, the compiler accounts for the size
            // of doubles in its arithmetic
            // Row i positions
            dblPtr[i+j*p]=XTX[i][j];
            // Column i positions
            dblPtr[i*p+j]=XTX[i][j];
          }
        }
        // b0 and continuous diagonals
        for(j=0; j<1+ncontXTotal; j++)
          dblPtr[j*p+j]=XTX[j][j];
        // FE diagnonals, equal to sums in first row
        // Iterate through all FE positions in X'X
        for(j=jdxFE[0]; j<jdxFE[0]+nlevFETotal; j++)
          dblPtr[j*p+j]=XTX[0][j];
        // Continuous X FE interaction diagonals
        for(j=jdxFE[0]+nlevFETotal; j<p; j++)
          dblPtr[j*p+j]=XTX[j][j];

        // Compose X'X column names
        CharacterVector namesXTX=CharacterVector(p);
        // Use q to indicate current element of name vector being updated
        // This avoids having to calculate element positions for each variable
        q=0;
        namesXTX(q++)=\"b0\";
        for(i=0; i<continuousCol.size(); i++) {
          namesXTX(q)=continuousCol(i);
          q++;
        }
        // X'X col names - continuous X continuous interactions
        for(i=0; i<nInteraction; i++)
          if(interactionCfg(i,0)==\"cc\") {
            namesXTX(q)=interactionCfg(i,1);
            namesXTX(q)+=\".\";
            namesXTX(q)+=interactionCfg(i,2);
            q++;
          }
        // X'X col names - FEs - format:  name(level)
        for(i=0; i<nFE; i++) {
          // Note that reference levels were omitted during construction
          lev=as<List>(as<List>(lResult(\"FE\"))(\"levels\"))(i);
          for(j=0; j<lev.size(); j++) {
             namesXTX(q)=FECol(i);
             namesXTX(q)+=\"(\";
             namesXTX(q)+=lev(j);
             namesXTX(q)+=\")\";
             q++;
          }
        }
        // X'X col names - FE X FE interactions - format:  FE1(level1).FE2(level2)
        for(i=0; i<nffInteraction; i++) {
          // Retrieve FE names and concatenated levels
          b1=as<CharacterMatrix>(ffInteractionLabel(\"FE\"))(i,_);
          lev=as<List>(ffInteractionLabel(\"level\"))(i);
          // Concatenate FE names with decomposed levels for each interaction level
          // Omit reference level in pos 0
          for(j=1; j<lev.size(); j++) {
            // Retrieve concatenated level
            a=lev(j);
            m=a.find(\".\");
            // FE1 name and parenthesized level
            namesXTX(q)=b1(0);
            namesXTX(q)+=\"(\";
            if(m>-1)
              namesXTX(q)+=a.substr(0,m);
            else
              namesXTX(q)+=a;
            namesXTX(q)+=\")\";
            // Delimter
            namesXTX(q)+=\".\";
            // FE2 name and parenthesized level
            namesXTX(q)+=b1(1);
            namesXTX(q)+=\"(\";
            if(m>-1)
              namesXTX(q)+=a.substr(m+1,a.size()-m);
            namesXTX(q)+=\")\";
            q++;
          }
        }
        // X'X col names - continuous X FE interactions - format:  nameContinuous.nameFE(level)
        for(i=0; i<nInteraction; i++)
          if(interactionCfg(i,0)==\"cf\") {
            // Retrieve levels of interacting FE
            lev=as<List>(as<List>(lResult(\"FE\"))(\"levels\"))(interactionIdx[i][1]);
            // Iterate through all levels, reference level was omitted during construction
            for(j=0; j<lev.size(); j++) {
               namesXTX(q)=interactionCfg(i,1);
               namesXTX(q)+=\".\";
               namesXTX(q)+=interactionCfg(i,2);
               namesXTX(q)+=\"(\";
               namesXTX(q)+=lev(j);
               namesXTX(q)+=\")\";
               q++;
            }
          }
        colnames(lResult(\"XTX\"))=namesXTX;
        rownames(lResult(\"XTX\"))=namesXTX;

        // Name FE level vectors
        as<List>(as<List>(lResult(\"FE\"))(\"levels\")).names()=FECol;

        // Compute X'Y
        if(Y.size()>0) {
          // Create return list embedded X'Y vector and corresponding pointer
          lResult(\"XTY\")=NumericVector(p);
          XTYPtr=&as<NumericVector>(lResult(\"XTY\"))(0);
          // Retrieve Y vector from supplied data
          a=Y(0);
          x1=data[a];
          yPtr=&x1(0);
          // Position 0 - sum of Y
          s=0;
          for(k=0; k<n; k++)
            s+=yPtr[k];
          XTYPtr[0]=s;
          // Continuous positions
          //#pragma omp parallel for private(j, k, s)
          for(j=0; j<ncontXTotal; j++) {
            s=0;
            for(k=0; k<n; k++)
              s+=contX[j][k]*yPtr[k];
            XTYPtr[1+j]=s;
          }
          // Fixed effect positions
          // Disjoint ridx[.][j] indices between FEs avoids update contention
          //#pragma omp parallel for private(j, k)
          for(j=0; j<nFETotal; j++)
            for(k=0; k<n; k++) {
              // Ignore reference level observations (coded with -1)
              if(ridx[k][j]>=0)
                XTYPtr[ridx[k][j]]+=yPtr[k];
            }
          // Continuous X FE positions
          for(i=0; i<ncfInteraction; i++) {
            // Index the continuous and FE columns
            m=cfInteraction[i][0];
            j=cfInteraction[i][1];
            // Retrieve X column offset of interaction
            // Adjust by subtracting FE offset, since ridx cols include it and
            // will be added to the interaction offset
            q=cfInteraction[i][2]-jdxFE[j];
            // Iterate through all observations
            // Accumulate interacting X and Y products where FE indicator is 1
            // Update X'Y positions using interaction offset
            for(k=0; k<n; k++) {
              if(ridx[k][j]>=0)
                XTYPtr[q+ridx[k][j]]+=contX[m][k]*yPtr[k];
            }
          }

        }

        // Save sparse row-wise FE observation design column indices
        // Note that the following method of pointer assignment generates an
        // 'int to long int conversion' error message, despite the list-embedded
        // object (obsFEColX) being an IntegerMatrix (collection of long int
        // vectors) and the pointers being of type long int (compatible with matrix)
        // Adding -fpermissive to the Makefileconf file (c:/users/.../R/.../etc/)
        // CXXFLAGS reduces the error to a warning and returned vectors are
        // confirmmed accurate
        // Attempt to declare lngPtr1 and lngPtr2 as type intptr_t does not resolve
        // the error (for more on this, see
        // http://rcpp-devel.r-forge.r-project.narkive.com/5KAUeG7Q/
        // 64-bit-ints-on-windows-64-bit-via-size-t-unsigned-long-long-or-unsigned-int64)
        // DECLARING lngPtr1 AND lngPtr2 AS INT POINTERS RESOLVES THE ERROR AND
        // PRODUCES ACCURATE RESULTS - IT APPEARS THAT INT POINTERS ARE ACTUALLY POINTERS
        // TO LONG INTS - THIS METHOD IS ADOPTED 
        if(retObsFEIndices.size()>0)
          if(retObsFEIndices(0)) {
            // Create matrix within result list
            as<List>(lResult(\"FE\"))(\"obsFEColX\")=IntegerMatrix(n,nFETotal);
            // Address row 0, col 0 of result matrix
            lngPtr1=&as<IntegerMatrix>(as<List>(lResult(\"FE\"))(\"obsFEColX\"))(0,0);
            // Iterate through all row indices and transfer
            //#pragma omp parallel for private(j, k, lngPtr2)
            for(j=0; j<nFETotal; j++) {
              lngPtr2=lngPtr1+j*n;
              for(k=0; k<n; k++)
                lngPtr2[k]=ridx[k][j];
            }
            // Name sparse row indicators
            CharacterVector namesFEColIdx=CharacterVector(nFETotal);
            // Main effects
            for(i=0; i<nFE; i++)
              namesFEColIdx(i)=FECol(i);
            // Interactions
            m=-1;
            for(i=0; i<nInteraction; i++)
              if(interactionCfg(i,0)==\"ff\") {
                m++;
                namesFEColIdx(nFE+m)=interactionCfg(i,1);
                namesFEColIdx(nFE+m)+=\".\";
                namesFEColIdx(nFE+m)+=interactionCfg(i, 2);
              }
            colnames(as<List>(lResult(\"FE\"))(\"obsFEColX\"))=namesFEColIdx;
          }

        // Release memory allocated to dynamic vectors and arrays
        for(i=0; i<n; i++)
          delete [] ridx[i];
        delete [] ridx;

        for(i=0; i<ncontXTotal; i++)
          delete [] contX[i];
        delete [] contX;

        for(i=0; i<p; i++)
          delete [] XTX[i];
        delete [] XTX;

        for(i=0; i<nFETotal; i++)
          delete [] feX[i];
        delete [] feX;

      }

    }

    if(stat(0)==\"ok\") {
      lResult(\"status\")=stat;
      return(lResult);
    } else {
      stop(as<std::string>(stat(1)));
    }

  }"

sourceCpp(code=cSource, rebuild=T, showOutput=T, cacheDir=getwd(), cleanupCacheDir=F)



