# ------------------------------------------------------------------------------
# Lab 3 helpfile
# ------------------------------------------------------------------------------

# If you do not have the libraries keras/tensorflow previously installed,
# please check out the "install_keras_and_more.R" file.

# Load packages
library(keras); library(keras3); library(tensorflow)
library(ggplot2); library(dplyr); library(tidyr)
library(data.table); library(rpart); library(caret)

reticulate::py_require("tensorflow")

reticulate::py_require("keras")

# ------------------------------------------------------------------------------
# PT1 - Neural networks for tabular data
# ------------------------------------------------------------------------------

# Load hitters data
if (requireNamespace("ISLR2", quietly = TRUE)) {
  data("Hitters", package = "ISLR2")
  hitters <- ISLR2::Hitters
} else if (requireNamespace("ISLR", quietly = TRUE)) {
  data("Hitters", package = "ISLR")
  hitters <- ISLR::Hitters
} else {
  stop("Please install the ISLR2 (preferred) or ISLR package to access 'Hitters'.")
}

# Prepare binary classification target: Salary >= median ==> 1, else 0
hitters <- na.omit(hitters)  # drop rows with missing Salary
hitters$y <- as.integer(hitters$Salary >= median(hitters$Salary, na.rm = TRUE))

# Build a design matrix (one-hot encode categorical) without the response
# model.matrix builds dummy variables for factors; the first column is intercept.
mm <- model.matrix(y ~ . - Salary, data = hitters)
X_all <- scale(mm[, -1])  # drop intercept, then standardize columns
y_all <- hitters$y        # labels (0/1)

# Train / test split (80/20)
set.seed(42)
n <- nrow(X_all)
idx <- sample(seq_len(n), size = floor(0.8 * n))
X_train <- X_all[idx, , drop = FALSE]
y_train <- y_all[idx]
X_test  <- X_all[-idx, , drop = FALSE]
y_test  <- y_all[-idx]

input_dim <- ncol(X_train)

# Small helper: accuracy on {0,1} labels given predicted probabilities
acc_bin <- function(p, y_true, thr = 0.5) mean((p > thr) == y_true)

# ------------------------------------------
# Model A: 0 hidden layers  (logistic in NN)
# ------------------------------------------
k_clear_session()
model_A <- keras_model_sequential() |>
  # layer_dense: fully connected (Dense) layer.
  # units = 1           -> one output unit (binary classification)
  # activation="sigmoid"-> maps real value to (0,1) probability
  # input_shape=input_dim -> number of input features (columns in X)
  layer_dense(units = 1, activation = "sigmoid", input_shape = input_dim)

# compile: configures the training process
model_A |>
  compile(
    optimizer = optimizer_adam(learning_rate = 1e-3), # Adam = adaptive gradient optimizer
    loss      = "binary_crossentropy",                # loss for binary classification
    metrics   = "accuracy"                            # track accuracy during training
  )

# fit: trains the model
history_A <- model_A |>
  fit(
    x = X_train, y = y_train,
    epochs = 25,            # number of passes over the data
    batch_size = 32,        # samples per gradient update
    validation_split = 0.2, # fraction of training used for validation
    verbose = 0
  )

# Evaluate on holdout
proba_A <- as.numeric(model_A |> predict(X_test))
acc_A   <- acc_bin(proba_A, y_test)
cat(sprintf("\n[Hitters] Model A (0 hidden) — test accuracy: %.3f\n", acc_A))

# ------------------------------------------
# Model B: 1 hidden layer
# ------------------------------------------
k_clear_session()
model_B <- keras_model_sequential() |>
  # Hidden layer:
  # units=32 (width of layer), activation="relu" (nonlinearity)
  layer_dense(units = 32, activation = "relu", input_shape = input_dim) |>
  # Output layer for binary classification:
  layer_dense(units = 1, activation = "sigmoid")

model_B |>
  compile(
    optimizer = optimizer_adam(1e-3),   # optimizer/learning rate
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history_B <- model_B |>
  fit(
    X_train, y_train,
    epochs = 25,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 0
  )

proba_B <- as.numeric(model_B |> predict(X_test))
acc_B   <- acc_bin(proba_B, y_test)
cat(sprintf("[Hitters] Model B (1 hidden) — test accuracy: %.3f\n", acc_B))

# ------------------------------------------
# Model C: 2 hidden layers
# ------------------------------------------
k_clear_session()
model_C <- keras_model_sequential() |>
  layer_dense(units = 64, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")

model_C |>
  compile(
    optimizer = optimizer_adam(1e-3),
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history_C <- model_C |>
  fit(
    X_train, y_train,
    epochs = 25,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 0
  )

proba_C <- as.numeric(model_C |> predict(X_test))
acc_C   <- acc_bin(proba_C, y_test)
cat(sprintf("[Hitters] Model C (2 hidden) — test accuracy: %.3f\n", acc_C))

# Quick table of results
hitters_res <- tibble::tibble(
  Model = c("A: 0 hidden","B: 1 hidden","C: 2 hidden"),
  Test_Accuracy = c(acc_A, acc_B, acc_C)
)
print(hitters_res)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# PT2 - Convolutional neural networks for image data
# ------------------------------------------------------------------------------

# Load MNIST (built into keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

# Peek at a digit (grayscale image)
# image() expects a matrix; reverse row order for upright display
par(mar = c(0,0,0,0))
img <- x_train[2,,]
image(t(apply(img, 2, rev)), col = gray.colors(256), axes = FALSE,
      main = "MNIST example (digit)")

# Preprocess:
# - reshape to 4D tensors: (samples, height, width, channels)
# - normalize pixel values to [0,1]
# - one-hot encode labels for 10 classes
img_rows <- 28L; img_cols <- 28L; channels <- 1L
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, channels)) # Note: only new part here is the channels (in our case, it is gray-scale; so only 1 channel; gets more interesting with color)
x_test  <- array_reshape(x_test,  c(nrow(x_test),  img_rows, img_cols, channels))
x_train <- x_train / 255 # normalize by dividing by max-value (1=most dark; 0=most bright)
x_test  <- x_test  / 255
y_train_cat <- to_categorical(y_train, num_classes = 10)  # one-hot targets
y_test_cat  <- to_categorical(y_test,  num_classes = 10)

# ------------------------------------------
# Simple CNN (M=1; K=1):
#   Conv -> Pool -> Flatten -> Dense -> Softmax
# ------------------------------------------
k_clear_session()
cnn <- keras_model_sequential() |>
  # layer_conv_2d:
  #  filters=32         -> number of feature maps learned
  #  kernel_size=c(3,3) -> size of the filters
  #  activation="relu"  -> nonlinearity after convolution
  #  input_shape        -> dimensions of input images (H, W, C)
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                input_shape = c(img_rows, img_cols, channels)) |>
  # layer_max_pooling_2d:
  #  pool_size=c(2,2)   -> downsample by taking local maxima over 2x2 windows
  layer_max_pooling_2d(pool_size = c(2,2)) |>
  # You can stack another conv+pool to improve accuracy (kept minimal here):
  # layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") |>
  # layer_max_pooling_2d(pool_size = c(2,2)) |>
  # Flatten: convert 2D feature maps into a 1D vector for Dense layer(s)
  layer_flatten() |>
  # Dense hidden layer after convolution (feature combination step)
  layer_dense(units = 128, activation = "relu") |>
  # Dropout:
  #  rate=0.3 -> randomly zeroes 30% of hidden units during training to reduce overfitting
  layer_dropout(rate = 0.3) |>
  # Output layer:
  #  units=10 (ten classes), activation="softmax" -> class probabilities sum to 1
  layer_dense(units = 10, activation = "softmax")

# Compile CNN:
#  optimizer: Adam (adaptive learning rate)
#  loss: categorical_crossentropy (multi-class classification)
#  metrics: "accuracy" for reporting
cnn |>
  compile(
    optimizer = optimizer_adam(),
    loss      = "categorical_crossentropy",
    metrics   = "accuracy"
  )

# Fit CNN:
#  epochs: small number for speed in a helpfile
#  batch_size: typical mini-batch size (how many observations to process before updating parameters)
#  validation_split: hold out part of training for monitoring
history_cnn <- cnn |>
  fit(
    x_train, y_train_cat,
    epochs = 3,
    batch_size = 128, 
    validation_split = 0.1,
    verbose = 0
  )

# Evaluate on test set
cnn_eval <- cnn |>
  evaluate(x_test, y_test_cat, verbose = 0)
cat(sprintf("\n[MNIST] CNN — test accuracy: %.3f\n", as.numeric(cnn_eval['accuracy'])))

# Predict a few samples
pred_classes <- cnn |>
  predict(x_test[1:9,,,], verbose = 0) |>
  k_argmax() |>
  as.numeric()
cat("\nFirst 9 predicted classes:", pred_classes, "\n")
cat("First 9 true classes     :", y_test[1:9], "\n")

# Take a sneak peak at those images
# Peek at a digit (grayscale image)
# image() expects a matrix; reverse row order for upright display
par(mar = c(0,0,0,0))
img <- x_test[2,,,]
image(t(apply(img, 2, rev)), col = gray.colors(256), axes = FALSE,
      main = "MNIST example (digit)")

