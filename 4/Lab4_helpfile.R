# --------------------------------------------------------
# Lab 4 helpfile
# --------------------------------------------------------

# my addition
rstudioapi::getSourceEditorContext()$path |> 
  dirname() |>
  setwd()

# --------------------------------------------------------
# Load packages
# --------------------------------------------------------
library(data.table)
library(ggplot2)
library(mlbench)
library(elasticnet)      # New
library(mclust)          # New
# --------------------------------------------------------



# --------------------------------------------------------
# Clustering some flowers
# --------------------------------------------------------

# To demonstrate the functions you will use to do clustering,
# I will consider the classic data set "iris".

data("iris")
head(iris)

# We'll start by excluding the id-variable (Species)
iris2 <- iris
iris2$Species <- NULL
head(iris2)

# Second, we want to standardize our data as the 
# columns are on different scales
iris2 <- scale(iris2)

# Now we are ready to cluster!
# - To find an appropriate k, we'll loop over a range of plausible candidates
#   and for each, compute the "total within cluster sum of squares"
ks <- 1:20
wss <- c()
for(i in 1:length(ks)){
  temp <- kmeans(x = iris2, centers = ks[i],nstart = 100) # Use nstart=100 for stability
  wss[i] <- sum(temp$withinss)
}
plot(wss) # Somewhere in the range 3-5 seems reasonable. 
# Let's go with 3.

set.seed(1234)
finalk <- kmeans(x = iris2, centers = 3, nstart = 100)

# Inspect centroids
finalk$centers
# 1: Cluster 1: High sepal.width, low on rest
# 2: Cluster 2: Small sepal.width, medium on rest
# 3: Cluster 3: High sepal.length, petal.length, petal.width

# Rather distinct clusters

# Add cluster solution to original data
iris$cluster <- finalk$cluster

# Plot cluster solution together with data
ggplot(iris,aes(x = Sepal.Length,y=Sepal.Width, color=factor(cluster))) + geom_point(size=3)

# Clusters that clearly "hang together". 
# The border between 2 & 3 seem rather sharp though.
# (of course, one should note that this considers only
# 2/4 dimensions)

# ---- Let's also try GMM on this data

# As for kmeans, we want to figure out an appropriate "k"
# (or G as they call it here)
lls <- bics <- c()
ks <- 1:10
for(i in 1:10){
  temp <- mclust::Mclust(data = iris2,G = ks[i])
  lls[i] <- temp$loglik
  bics[i] <- temp$bic
}
plot(lls)  # 4
plot(bics) # 2

gmm <- mclust::Mclust(data = iris2) # If we don't specify "G", mclust will try to find an optimal for you.
#gmm <- mclust::Mclust(data = iris2,G = 4)
gmm$G # 2 here (because it uses BIC as criteria)
gmm$bic

# Extract centers
gmm$parameters$mean

# Extract uncertainity for each observation
gmm$uncertainty
iris$gmm_uncertainity <- gmm$uncertainty
ggplot(iris,aes(x = Sepal.Length,
                y = Sepal.Width,
                color = gmm_uncertainity)) + 
  geom_point(size=3)

# Plot PC1, PC2, colored by uncertainity
pca <- prcomp(x = iris2)
iris_pca <- cbind(iris,pca$x[,c(1,2)])
ggplot(iris_pca,aes(x = PC1,y=PC2, color=gmm_uncertainity)) + geom_point(size=3)

# Plot PC1, PC2 colored by GMM assignment
iris_pca$gmm_cluster <- gmm$classification
ggplot(iris_pca,aes(x = PC1,y=PC2, color=factor(gmm_cluster))) + geom_point(size=3)

# Extract soft assignment
gmm$z



mymod <- densityMclust(iris2)
mymod |> 
  plot(what = "density", type = "persp")

# --------------------------------------------------------
# Kmeans vs. GMM on Gender / Height.
# --------------------------------------------------------

# Import simulated data on heights
heights_dt <- readRDS(file = 'heights_dt.rds')

# k-means
kmeans_solution <- kmeans(x = heights_dt$height, centers = 2)
heights_dt[,kmeans_cluster := factor(kmeans_solution$cluster)]
ggplot(heights_dt,aes(x=height,fill=factor(kmeans_cluster))) + geom_density(alpha=0.5)
ggplot(heights_dt,aes(x=height,y=kmeans_cluster,color=kmeans_cluster)) + geom_point()

# GMM
gmm_height <- Mclust(data = heights_dt$height, G = 2)
gmm_height$z

heights_dt2 <- cbind(heights_dt, gmm_height$z)
setnames(heights_dt2,c('V1','V2'),c('gmm1','gmm2'))
heights_dt2 <- heights_dt2[order(height)]
heights_dt2[,row_id := .I]
ggplot(data = heights_dt2, aes(x=height,y=gmm1)) + 
  geom_point(color="red") +            # GMM component 1
  geom_point(aes(y=gmm2),color="blue") # GMM component 2


mymod2 <- densityMclust(heights_dt)
mymod2 |> 
  plot(what = "density", type = "persp")


# --------------------------------------------------------
# PCA & USArrest example (from lecture)
# --------------------------------------------------------

# Load data
data("USArrests")

# Standardize
USArrests2 <- scale(USArrests)

# Fit standard PCA
standard_pca <- prcomp(USArrests2)

# PVE
res_standard_pca <- summary(standard_pca)
res_standard_pca$importance
plot(res_standard_pca$importance[2,])
plot(cumsum(res_standard_pca$importance[2,])) # 2PCs capture ~90% of variance

# Principal component loadings
standard_pca$rotation[,c(1,2)] # As noted in class, not super clean interpretation

# Sparse PCA comparison

## grid of different penalty values
## should be evenly spaced on a logscale
## (martin did this like in the glmnet package)
penalities <- c(0,1,5,10,20,80,100,200,500,1000) 

spca_pevs <- c()
spca_ps <- c()
for(i in 1:length(penalities)){
  temp <- spca(x = USArrests2, 
               K = 2, # principle components set informed by the standard PCA
               type = 'predictor', 
               para = c(rep(penalities[i],ncol(USArrests2))), # specify as many lambas as columns in the dataset 
               sparse = 'penalty')
  spca_pevs[i] <- sum(temp$pev)
  spca_loadings <- temp$loadings
  ps <- length(spca_loadings[abs(spca_loadings)<=0.01]) / length(spca_loadings)
  spca_ps[i] <- ps
  if(length(ps)==0){ps <- 0}
}

is_pev_dt <- data.table(lambda=penalities,
                        spca_PEV=spca_pevs,
                        spca_PS=spca_ps)
standard_PCA_PEV <- is_pev_dt[lambda==0]$spca_PEV
is_pev_dt[,IS:=standard_PCA_PEV * spca_PEV * spca_PS]

my_plot <- ggplot(is_pev_dt, aes(x=spca_PS)) +
  geom_line( aes(y=IS)) + 
  geom_line( aes(y=spca_PEV),linetype='dashed') +
  scale_y_continuous(name = "IS",
    sec.axis = sec_axis(~.*1, name="PEV",breaks = scales::pretty_breaks(n=8)),
    breaks = scales::pretty_breaks(n=8)) +
  ggtitle('PEV x IS x PS for USarrest data') +
  theme_gray(base_size = 10)
plot(my_plot) # when IS score is maximized it has the best balance (thick line at 0.5) --> increased sparsity of loadings to 50%

# Re-fit best spec
final_gmm <- spca(x = USArrests2, 
                  K = 2, 
                  type = 'predictor', 
                  para = rep(20, ncol(USArrests2)), 
                  sparse = 'penalty')

# Compute (and plot) principal scores
final_gmm$loadings <- abs(final_gmm$loadings)
zscores1_2 <- USArrests2[,1] * final_gmm$loadings[1,1] + USArrests2[,2] * final_gmm$loadings[2,1] + USArrests2[,3] * final_gmm$loadings[3,1] + USArrests2[,4] * final_gmm$loadings[4,1]
zscores2_2 <- USArrests2[,2] * final_gmm$loadings[1,2] + USArrests2[,2] * final_gmm$loadings[2,2] + USArrests2[,3] * final_gmm$loadings[3,2] + USArrests2[,4] * final_gmm$loadings[4,2]
z_scores_dt <- data.table(state=rownames(USArrests2),
                          z1=zscores1_2,
                          z2=zscores2_2)

my_plot2 <- ggplot(z_scores_dt,aes(x=z1,y=z2)) +
  geom_text(aes(label=state),size=3) +
  xlab('PC1 (~crime)') + ylab('PC2 (~pop)') + 
  geom_hline(yintercept = 0, linetype='dashed') +
  geom_vline(xintercept = 0, linetype='dashed') +
  theme_gray(base_size = 10)
my_plot2






