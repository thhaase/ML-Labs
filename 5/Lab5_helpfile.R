# --------------------------------------------------------
# Lab 4 help file
# --------------------------------------------------------



# --------------------------------------------------------
# Load packages
# --------------------------------------------------------
library(data.table)
library(dplyr)
library(ggplot2)
library(slam)         # New: needed for perplexity calc
library(quanteda)     # New: to process text
library(topicmodels)  # New: to fit topic models
library(word2vec)     # New: to fit word embeddings
# --------------------------------------------------------



# --------------------------------------------------------
# Pre-processing text data
# --------------------------------------------------------

# Import data (this is the raw version of the bernie-trump-tweets dataset)
setwd('/Users/marar08/Documents/Teaching/MLSS_HT2025/Labs/W5/')
bernie_trump <- fread(input = 'bernietrump_w4_2.csv')
bernie_trump[,doc_id := .I]
dim(bernie_trump)
colnames(bernie_trump)
bernie_trump[1,]  # Note: we see things like period marks, and capital letters.

# =======================================
# Pre-process
# =======================================

# Create a (quanteda) corpus
# - this allows us to use this package's tools
#   to clean our texts (see next steps)
posts_corpus <- corpus(x = bernie_trump,
                       text_field = "text",
                       meta = list("username"),
                       docid_field = 'doc_id')
print(posts_corpus)

# Tokenize & clean from particular types of words
mytokens <- tokens(x = posts_corpus, 
                   remove_punct = TRUE, 
                   remove_numbers = TRUE, 
                   remove_symbols = TRUE,
                   remove_url = TRUE,
                   padding = FALSE)
print(mytokens)

# Remove English stop words
mytokens <-   tokens_remove(x = mytokens,
                            stopwords("en"), 
                            padding = FALSE)
# Remove @'s
mytokens <-   tokens_select(x = mytokens,
                            selection = 'remove',
                            valuetype = 'glob', 
                            pattern = '@', 
                            padding = FALSE)

# Make tokens lowercase
mytokens <- tokens_tolower(x = mytokens)

# How does it look now?
mytokens

# Create document term matrix
dtm <- dfm(x = mytokens)
dtm

# Exclude words with too low frequency
dtm <- dfm_trim(dtm, min_termfreq = 5)

# Exclude documents with too low frequency
rowsums <- rowSums(dtm)
keep_ids <- which(rowsums>=10)
dtm <- dtm[keep_ids,]

# Report final dimensionality of data set
dim(dtm)



# --------------------------------------------------------
# Topic modeling in R
# --------------------------------------------------------

# Fit LDA (package: topicmodels)
# - 30 topics
# - Estimation algorithm: Gibbs (mathematical/physical solution, very accurate, very precise and very demure)
# - 1000 iterations
# - Set seed for replicability
# - Print iter every 100th
set.seed(1)
K <- 30 #50
mylda <- LDA(x = dtm, 
             k = K, 
             method="Gibbs",
             control=list(iter = 1000, 
                          seed = 1, 
                          verbose = 100))

# Inspect top 10 from each topics
get_terms(mylda, 10)
get_terms(mylda, 10)[,1:10] # Only first 10 topics

# To get a more granular view, extract the probabilities 
mylda_posterior <- topicmodels::posterior(object = mylda)
topic_distr_over_words <- mylda_posterior$terms
topic_distr_over_words_dt <- data.table(topic=1:K, 
                                        topic_distr_over_words)
topic_distr_over_words_dt <- melt.data.table(topic_distr_over_words_dt, # If you use base R, you can use reshape's "melt()" function.
                                             id.vars = 'topic')

# Tidyr/Dplyr way of extracting top 10 rows by group
top10per_topic <- topic_distr_over_words_dt %>% 
  group_by(topic) %>% 
  slice_max(order_by = value, n = 10)

# data.table way of extracting top 10 rows by group
topic_distr_over_words_dt <- topic_distr_over_words_dt[order(value,decreasing = T)]
top10per_topic <- topic_distr_over_words_dt[,head(.SD,10),by='topic']

# Plot probability over words for a few topics.
ggplot(top10per_topic[topic %in% c(6,7,10)],aes(y=factor(variable),x=value)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(~topic,scales = 'free')

# To validate labeling; identify documents with highest proportion on any given topic
doc_topic_proportions <- mylda_posterior$topics
dim(doc_topic_proportions)
head(doc_topic_proportions)
doc_topic_proportions_dt <- data.table(doc_id = mylda@documents,
                                       doc_topic_proportions)
colnames(doc_topic_proportions_dt)[2:ncol(doc_topic_proportions_dt)] <- paste0('Topic',colnames(doc_topic_proportions_dt)[2:ncol(doc_topic_proportions_dt)]) # Assign topics as column names
doc_topic_proportions_dt[order(Topic7,decreasing = T)][1:5,]

doc_topic_proportions_dt[doc_id==4401,] # document with high proportion for topic 7

# Select rows from original data...
bernie_trump[doc_id==4401,]


# Data-driven way of selecting K (by perplexity)
# ==============================================
# - split data into train/test

# perplexity looks at the documents it has not yet seen and looks at how surprising they are. are the documents that you have not yet seen - do they align, how probable are they given your topic mdoel
row_ids <- 1:nrow(dtm)
tr_ids <- sample(x = row_ids, size = 0.8*length(row_ids), replace = F)
tst_ids <- row_ids[!row_ids %in% tr_ids]
tr_dt <- dtm[tr_ids,]
tst_dt <- dtm[-tr_ids,]
dim(tst_dt)

ks <- c(3,10,25,50,75,100)
perp <- c()
for(i in 1:length(ks)){
  
  # Train
  current_tm <- LDA(x = tr_dt,
                    k = ks[i], 
                    method="Gibbs",
                    control=list(iter = 500, 
                                 seed = 1, 
                                 verbose = 100))
  # Compute perplexity
  perp[i] <- perplexity(object = current_tm, 
                        newdata = as.simple_triplet_matrix(tst_dt)) # For some reason, perplexity() wants the test data in simple_triplet_matrix format.
  print(perp)
}
plot(x=ks,y=perp)


# --------------------------------------------------------
# Word embeddings in R
# --------------------------------------------------------

# Re-import & re-process data *without removing stopwords*
setwd('/Users/marar08/Documents/Teaching/MLSS_HT2025/Labs/W5/')
bernie_trump <- fread(input = 'bernietrump_w4_2.csv')

# Make into corpus
posts_corpus <- corpus(x = bernie_trump,
                       text_field = "text",
                       meta = list("username"),
                       docid_field = 'doc_id')

# Tokenize & clean from particular types of words
mytokens <- tokens(x = posts_corpus, 
                   remove_punct = TRUE, 
                   remove_numbers = TRUE, 
                   remove_symbols = TRUE,
                   remove_url = TRUE,
                   padding = FALSE)
mytokens <-   tokens_select(x = mytokens,selection = 'remove',
                            valuetype = 'glob', 
                            pattern = '@', 
                            padding = FALSE)

# Make tokens lowercase
mytokens <- tokens_tolower(x = mytokens)

# Collapse into strings within documents
txt <- sapply(mytokens,function(x)paste(x,collapse = ' '))
txt <- tolower(txt)
txt[3:5]

# Fit word embeddings
set.seed(123456789)
system.time(w2v <- word2vec(x = txt,          # Your data 
                            type = "cbow",    # Model type (the one we talked about in lecture)
                            window = 5,       # Context defined in terms of +-5 words.
                            dim = 50,         # Dimensionality of embeddings
                            iter = 50,        # Estimation iterations (higher means more time...)
                            hs = FALSE,       # Setting to FALSE here --> "negative sampling procedure" (learn difference between fake/true sentences) how many fake sentences should we have for fake/true sentences
                            negative = 15))   # Number of negative samples (how many fake sentences per true sentence)

# How many tokens did we use?
w2v$data$n  # This is a rather small dataset. Let's see how well it works.

# With the embeddings estimated, we can do all sorts of things

# What is the nearest (most semantically similar) word to say "president"?
predict(w2v, c('president'), type = "nearest", top_n = 5)

# To extract a particular embedding vector
president_embedding <- predict(w2v, c('president'), type = "embedding")
president_embedding

# If we want to extract the whole embedding matrix, we can do the following
embedding <- as.matrix(w2v)
dim(embedding)

# Suppose we wanted create a "sentiment" dimension, and project words onto this dimension
pos_terms <- fread('positive.txt',header = F)
neg_terms <- fread('negative.txt',header = F)

# To create the projection, we follow these steps:

# 1) Extract the relevant word vectors from the "embedding" matrix
pos_vectors <- embedding[which(rownames(embedding) %in% pos_terms$V1),]
neg_vectors <- embedding[which(rownames(embedding) %in% neg_terms$V1),]

# 2) Compute the average across each and keep matrix format
pos_vector <- as.matrix(apply(pos_vectors,2,mean))
neg_vector <- as.matrix(apply(neg_vectors,2,mean))

# 3) Transform to get 1 x K
pos_vector <- t(pos_vector)
neg_vector <- t(neg_vector)

# 4) Compute the difference to get a gender dimension
pos_neg_dimension <- pos_vector - neg_vector
neg_pos_dimension <- neg_vector - pos_vector

# Extract "positive" terms and "negative" terms
pos_assoc_terms <- word2vec::word2vec_similarity(x = pos_neg_dimension, 
                                                 y = embedding[-which(rownames(embedding) %in% pos_terms$V1),], 
                                                 top_n = 50)
neg_assoc_terms <- word2vec::word2vec_similarity(x = neg_pos_dimension, 
                                                 y = embedding[-which(rownames(embedding) %in% neg_terms$V1),], 
                                                 top_n = 50)
# Clear patterns for negative side. 
# Not as clear for positive.
# -- could partially be because we mix bernie & donald tweets here.
neg_assoc_terms
pos_assoc_terms[1:50,]


# ===============================
# EXTRA 1:
# In a **bonus** task in the assignments, you are asked to calculate bootstrap CI.
# Here is one way of calculating bootstrap confidence intervals for similarity ranks
# - Note: here using a single corpus
# - Objective: get an idea of the robustness of the negative sentiment of "fbi"
# ===============================

B <- 20 # number of bootstraps (minimum of 20 for 90% confidence interval)
fbi_ranks <- c() # to store each bootstrap's rank
for(b in 1:B){
  
  # Sample documents ids with replacement (= creating the b'th bootstrap sample)
  row_ids <- 1:nrow(bernie_trump)
  bootstrap_b_ids <- sample(x = row_ids, size = nrow(bernie_trump), replace = TRUE)
  bootstrap_b_data <- bernie_trump[bootstrap_b_ids,]

  # Then... we use the same code as we did before, without bootstrapping
  # Just that we replace the input to the corpus.
  
  # Make into corpus
  posts_corpus <- corpus(x = bootstrap_b_data, # Here we insert the boostraped data for this iteration
                         text_field = "text",
                         meta = list("username"),
                         docid_field = 'doc_id')
  
  # Tokenize & clean from particular types of words
  mytokens <- tokens(x = posts_corpus, 
                     remove_punct = TRUE, 
                     remove_numbers = TRUE, 
                     remove_symbols = TRUE,
                     remove_url = TRUE,
                     padding = FALSE)
  mytokens <-   tokens_select(x = mytokens,selection = 'remove',
                              valuetype = 'glob', 
                              pattern = '@', 
                              padding = FALSE)
  
  # Make tokens lowercase
  mytokens <- tokens_tolower(x = mytokens)
  
  # Collapse into strings within documents
  txt <- sapply(mytokens,function(x)paste(x,collapse = ' '))
  txt <- tolower(txt)
  
  # Fit word embeddings
  set.seed(123456789)
  system.time(w2v <- word2vec(x = txt,          # Your data 
                              type = "cbow",    # Model type (the one we talked about in lecture)
                              window = 5,       # Context defined in terms of +-5 words.
                              dim = 50,         # Dimensionality of embeddings
                              iter = 50,        # Estimation iterations (higher means more time...)
                              hs = FALSE,       # Setting to FALSE here --> "negative sampling procedure"
                              negative = 15))   # Number of negative samples
  
  # Pos/neg list (these could also have been imported before the loop)
  pos_terms <- fread('positive.txt',header = F)
  neg_terms <- fread('negative.txt',header = F)
  
  # Extract the whole embedding matrix
  embedding <- as.matrix(w2v)
  
  # Create the projection:
  # 1) Extract the relevant word vectors from the "embedding" matrix
  pos_vectors <- embedding[which(rownames(embedding) %in% pos_terms$V1),]
  neg_vectors <- embedding[which(rownames(embedding) %in% neg_terms$V1),]
  # 2) Compute the average across each and keep matrix format
  pos_vector <- as.matrix(apply(pos_vectors,2,mean))
  neg_vector <- as.matrix(apply(neg_vectors,2,mean))
  # 3) Transform to get 1 x K
  pos_vector <- t(pos_vector)
  neg_vector <- t(neg_vector)
  # 4) Compute the difference to get a gender dimension
  neg_pos_dimension <- neg_vector - pos_vector
  
  # Identify (and store) similarity rank (to sentiment dimension) for the word "fbi"
  neg_assoc_terms <- word2vec::word2vec_similarity(x = neg_pos_dimension, 
                                                   y = embedding[-which(rownames(embedding) %in% pos_terms$V1),], 
                                                   top_n = 10000) # Set this to a large value to get distance to all words
  neg_assoc_terms <- as.data.table(neg_assoc_terms)
  fbi_ranks[b] <- neg_assoc_terms[term2=="fbi"]$rank
  
  # Print iteration
  print(paste0('Bootstrap ',b,' finished. Rank: ', fbi_ranks[b]))
}

# Compute 90th interval
fbi_ranks <- fbi_ranks[order(fbi_ranks,decreasing=F)]
print(paste0('Boostrap 90% interval: [',fbi_ranks[2], ',',fbi_ranks[19],']'))


# ===========================
# EXTRA 2:
# Here is one way of calculating the frequency of a word in a corpus
# - This can be useful for selecting the words to compare across parties
# ===========================

# We can do this many ways, but sticking to the quanteda package, we can do:
# Step 1: make tokens object into a document term matrix
dfm <- quanteda::dfm(x = mytokens)
# Step 2: compute frequency for each term
word_freqs <- quanteda::featfreq(x = dfm) # equivalent to colSums(dfm)
# Step 3: identify your term of interest
word_freqs[names(word_freqs)=="fbi"]





