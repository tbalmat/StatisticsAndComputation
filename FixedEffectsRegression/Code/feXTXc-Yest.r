###############################################################################################
#### The following script computes response estimates resulting from a source data set and
#### parameter estimates from a model fit to the data using X'X, X'Y, and sparse fixed effect
#### index columns produced by feXTXc
####
#### The function accepts:
#### 1. A data frame containing dependent data columns:
####    a. Numeric continuous independent columns
####    b. Character fixed effect (categorical) columns
#### 2. The list generated by a previous execution of feXTXc using data from item 1
####    List elements include:
####    a. A vector of continuous column names from the data frame
####    b. A vector of fixed effects columns from the data frame
####    c. An interaction configuration matrix
#### 3. A vector of computed parameter estimates corresponding to the model used to produce
####    the list in item 2
####
#### Required list structure is described in the program notes for function feXTXc
####
#### The function produces a vector of dependent variable estimated values, one for each row in
#### the source data frame, compose from the product of independent variable values and
#### corresponding supplied parameter estimates
####
#### Note that fixed effect variable values are treated as either 0 (observation, or row, is
#### not encoded for a given FE and level) or 1 (observation is encoded for a FE and level
####
#### Fixed effect reference levels, as specified in the feXTXc call, are excluded from prior
#### computed X'X, X'Y, and parameter estimates, making parameter estimates for reference
#### levels effectively 0 and causing parameter estimates of remaining levels to be a deviation
#### from reference
#### 
#### Program outline: 
####
#### 1. Validate supplied data and parameters (ensure all specified columns exist in the data
####    and that recorded fixed effect levels agree with those in the data)
#### 2. Compute one y estimate for each row of supplied data by multiplying dependent column
####    levels to corresponding parameter estimates
####
#### Expected input parameters:
#### 1. Data frame containing columns of independent data
#### 2. feXTXc list structure:
####    status ........ a two element vector, the first containing either "ok" or "Error" and
####                    the second containg additional information when an error occurs
####    call .......... list containing supplied parameter values, each element named with 
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
#### 3. beta:  Numeric vector of parameter estimates, one for each parameter in the model fit
####    to the data.  Order is b0, continuous, continuous X continuous interactions, FE,
####    FE X FE interactions, continuous X FE interactions.  The order of parameter estimates
####    within continous and fixed effects must be identical to their apperance in the
####    continuousCol and FECol parameters of the supplied feXTXc result list (this is
####    satisfied when estimates are solved using X'X*beta=X'Y, where X'X and X'Y are output
####    from feXTXc).
#### 4. verifyModel:  T/F (default T) indicator to verify existence of specified columns in
####    supplied data and positions of parameter estimates with suplied X'X matrix (presumed
####    to have been used to compute estimates)
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
  NumericVector feXTXcYest(DataFrame data, List feXTXcModel, NumericVector beta,
                           LogicalVector verifyModel=LogicalVector(0), int nCore=0) {

    // Parameters:
    // data .......... Data frame containing independent data columns
    // feXTXcModel ... List generated by previous execution of feXTXc
    // beta .......... Numeric vector of computed parameter estimates
    // verifyModel ... Logical vector, first element instructing whether to
    //                 verify input model structure with supplied data
    //                 and parameter estimates
    // nCore ......... Maximum number of cores to use in parallel operations

    // Use long indices for high dimension addressing, but signed due to lack of support for
    // unsigned indices in Rcpp matrix objects
    long i, j, k, m;
    long n=data.nrow();
    std::string c;
    NumericVector yEst(n), x1, x2;
    CharacterVector continuousCol=as<List>(feXTXcModel(\"call\"))(\"continuousCol\");
    long ncontX=continuousCol.size();
    CharacterVector FECol=as<List>(feXTXcModel(\"call\"))(\"FECol\");
    long nFE=FECol.size();
    CharacterMatrix interactionCfg=
      as<CharacterMatrix>(as<List>(feXTXcModel(\"interaction\"))(\"interactionCfg\"));
    long nInteraction=interactionCfg.nrow();
    CharacterVector xNames=data.names();
    CharacterVector stat=CharacterVector::create(\"ok\", \"\");
    CharacterVector a, b;
    IntegerMatrix obsFEColX;
    IntegerVector offsetX;
    IntegerVector nLevel;

    // Set the max number of cores to use
    if(nCore>0)
      omp_set_num_threads(nCore);

    /////////////////////////////////////////////////////////////////////////////////////////
    // Validate input parameters
    /////////////////////////////////////////////////////////////////////////////////////////

    if(verifyModel.size()>0)
      if(verifyModel(0)) {

        // Compare order of parameter estimates with columns in X'X
        if(!Rf_isNull(as<NumericMatrix>(feXTXcModel(\"XTX\")).attr(\"dimnames\"))) {
          a=colnames(feXTXcModel(\"XTX\"));
          if(!Rf_isNull(beta.attr(\"names\"))) {
            b=beta.names();
            if(a.size()==b.size()) {
              for(j=0; j<a.size() && stat(0)==\"ok\"; j++)
                if(a(j)!=b(j)) {
                  stat(0)=\"Error\";
                  stat(1)=\"X'X and beta names different (\";
                  stat(1)+=a(j);
                  stat(1)+=\", \";
                  stat(1)+=b(j);
                  stat(1)+=\")\";
                }
            } else {
              stat(0)=\"Error\";
              stat(1)=\"X'X and beta name vectors of different length - cannot compare\";
            }
          } else {
            stat(0)=\"Error\";
            stat(1)=\"Beta names are empty - cannot compare to X'X matrix\";
          }
        } else {
          stat(0)=\"Error\";
          stat(1)=\"X'X column names are empty - cannot compare to beta vector\";
        }

        // Verify that continuous columns exist in data
        if(stat(0)==\"ok\")
          for(j=0; j<ncontX && stat(0)==\"ok\"; j++)
            // match() returns 1-based column indicators
            if(match(as<CharacterVector>(continuousCol(j)), xNames)[0]==NA_INTEGER) {
              stat(0)=\"Error\";
              stat(1)=\"Continuous column missing in supplied data (\";
              stat(1)+=continuousCol(j);
              stat(1)+=\")\";
            }

        // Verify that FE columns exist in data
        if(stat(0)==\"ok\")
          for(j=0; j<nFE && stat(0)==\"ok\"; j++)
            // match() returns 1-based column indicators
            if(match(as<CharacterVector>(FECol(j)), xNames)[0]==NA_INTEGER) {
              stat(0)=\"Error\";
              stat(1)=\"Fixed effect column missing in supplied data (\";
              stat(1)+=FECol(j);
              stat(1)+=\")\";
            }

        // Verify that sparse row index vectors exist
        if(stat(0)==\"ok\") {
          if(TYPEOF(as<List>(feXTXcModel(\"FE\"))(\"obsFEColX\"))!=INTSXP) {
            stat(0)=\"Error\";
            stat(1)=\"Sparse FE indicator vectors (obsFEColX) missing in supplied model cfg (feXTXcModel)\";
          } else if(as<IntegerMatrix>(as<List>(feXTXcModel(\"FE\"))(\"obsFEColX\")).nrow()!=n) {
            stat(0)=\"Error\";
            stat(1)=\"Sparse FE indicator vectors (obsFEColX) row dimension different from supplied data\";
          }
        }

      }

    /////////////////////////////////////////////////////////////////////////////////////////
    // Compute y estimates
    /////////////////////////////////////////////////////////////////////////////////////////

    if(stat(0)==\"ok\") {

      // Compute y estimates

      // intercept
      for(i=0; i<n; i++)
        yEst(i)=beta(0);

      // Continuous cols
      for(j=0; j<ncontX; j++) {
        c=continuousCol(j);
        x1=data[c];
        for(i=0; i<n; i++)
          yEst(i)+=beta(1+j)*x1(i);
      }

      // Continuous X continuous interactions
      // Note the assumption that beta estimates appear in the same order as
      // interactions appear in rows of interaction cfg matrix (that is how
      // they are composed by feXTXc)
      // Index design matrix (and beta) position of first c X c interaction
      m=1+ncontX; 
      for(k=0; k<nInteraction; k++)
        if(interactionCfg(k,0)==\"cc\") {
          c=interactionCfg(k,1);
          x1=data[c];
          c=interactionCfg(k,2);
          x2=data[c];
          for(i=0; i<n; i++)
            yEst(i)+=beta(m)*x1(i)*x2(i);
          m++;
        }

      // Fixed effects
      // Sparse index vectors (obsFEColX) indicate, for each row and FE, the
      // design matrix column (and beta position) corresponding to the encoded
      // level for the FE (reference levels are indicated with an index value
      // of -1 and are omitted)
      // FE X FE interactions are encoded in obsFEColX columns following main
      // FE columns (reference levels are encoded as -1 and correspond to
      // a reference level in either of the main interacting FEs)
      // FE (and FE X FE interaction) design column offsets (column of first level)
      // are in the offsetX vector, ordered as in obsFEColX
      obsFEColX=as<IntegerMatrix>(as<List>(feXTXcModel(\"FE\"))(\"obsFEColX\"));
      offsetX=as<IntegerVector>(as<List>(feXTXcModel(\"FE\"))(\"offsetX\"));

      for(j=0; j<obsFEColX.ncol(); j++) {
        for(i=0; i<n; i++) {
          // Ignore reference level observations (coded as -1)
          if(obsFEColX(i,j)>=0)
            // Index position in beta vector corresponding to FE and level encoded
            // for observation (design matix offset corresponds to beta position)
            yEst(i)+=beta(obsFEColX(i,j));
        }
      } 

      // Continuous X FE interactions
      // Note the assumption that beta estimates appear in the same order as
      // interactions appear in rows of interaction cfg matrix (that is how
      // they are composed by feXTXc)
      // Index design matrix (and beta) position of first c X FE interaction
      // This follows the last fixed effect, which is assumed to exist
      nLevel=as<IntegerVector>(as<List>(feXTXcModel(\"FE\"))(\"nLevel\"));
      m=offsetX(offsetX.size()-1)+nLevel(nLevel.size()-1);
      for(k=0; k<nInteraction; k++)
        if(interactionCfg(k,0)==\"cf\") {
          // Index continuous data column
          c=interactionCfg(k,1);
          x1=data[c];
          // Index FE column
          // match() returns 1-based column indicators
          // It is assumed that FE col existence has been verified (no check done here)
          j=match(as<CharacterVector>(interactionCfg(k,2)), FECol)[0]-1;
          // Multiply row-wise cf interaction coef for FE level by continuous value
          // Convert FE level indices from design col positions to offsets within FE
          // by subtracting FE first level offset from ith observation col index
          // This is also an offset within the cf beta coefficients corresponding to
          // the FE level
          for(i=0; i<n; i++)
            if(obsFEColX(i,j)>=0)
              //yEst(i)+=beta(m+obsFEColX(i,j)-offsetX(j))*x1(i);
              yEst(i)+=beta(m+obsFEColX(i,j)-offsetX(j))*x1(i);
          // Advance to offset to beyond current cf interaction
          m+=nLevel(j);
        }

    }

    if(stat(0)==\"ok\")
      return(yEst);
    else
      stop(as<std::string>(stat(1)));

  }"

sourceCpp(code=cSource, rebuild=T, showOutput=T, cacheDir=getwd(), cleanupCacheDir=F)



