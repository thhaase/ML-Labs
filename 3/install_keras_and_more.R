# ------------------------------------------------------------
# Deep Learning Lab Setup (R + keras/keras3 + TensorFlow)
# Works on Windows / macOS (Intel or Apple Silicon) / Linux
# Run in a FRESH R session (before any Python is initialized).
# ------------------------------------------------------------

# Toggle: use Keras 3 (recommended). Set to FALSE to test legacy 'keras' API.
use_keras3 <- TRUE

# Target Python env name (created/managed by this script)
envname <- "r-tensorflow"

# 0) Helpers -------------------------------------------------
msg <- function(...) cat(paste0(..., "\n"))
section <- function(title) { msg("\n", title); msg(strrep("=", nchar(title))) }
ok  <- function(...) msg(paste("✔", ...))
warn <- function(...) msg(paste("! ", ...))

# Run a conda subcommand and ignore errors (return output invisibly)
conda_run <- function(conda_bin, ...) {
  args <- c(...)
  out <- tryCatch(
    system2(conda_bin, args, stdout = TRUE, stderr = TRUE),
    error = function(e) e
  )
  invisible(out)
}

accept_tos_and_pin_channels <- function(conda_bin) {
  # Accept Anaconda ToS non-interactively (idempotent; safe if already accepted)
  tos_channels <- c(
    "https://repo.anaconda.com/pkgs/main",
    "https://repo.anaconda.com/pkgs/r",
    "https://repo.anaconda.com/pkgs/msys2"
  )
  for (ch in tos_channels) {
    conda_run(conda_bin, "tos", "accept", "--override-channels", "--channel", ch)
  }
  
  # If this is reticulate's private r-miniconda, scope channel config to that install.
  is_r_miniconda <- grepl("r[-]?miniconda", conda_bin, ignore.case = TRUE)
  if (is_r_miniconda) {
    # Reset channels to conda-forge only, with strict priority, at the *install* level.
    conda_run(conda_bin, "config", "--system", "--remove-key", "channels")
    conda_run(conda_bin, "config", "--system", "--add", "channels", "conda-forge")
    conda_run(conda_bin, "config", "--system", "--set", "channel_priority", "strict")
    ok("Configured r-miniconda to use conda-forge with strict priority.")
  } else {
    warn("Using a non r-miniconda Conda installation; leaving global channel config unchanged.")
  }
}

# 1) Ensure CRAN repo and R packages -------------------------
if (is.null(getOption("repos")) || is.na(getOption("repos")["CRAN"]) ||
    getOption("repos")["CRAN"] %in% c("@CRAN@", "", NA)) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

needed <- c("reticulate", "tensorflow", "keras3", "keras")
to_install <- setdiff(needed, rownames(installed.packages()))
if (length(to_install)) {
  section("Installing required R packages (CRAN)")
  install.packages(to_install, quiet = TRUE)
}

suppressPackageStartupMessages({
  library(reticulate)
  library(tensorflow)  # do NOT load keras/keras3 yet (prevents early Python init)
})

# 2) Guard: if Python already initialized, handle carefully ---
py_inited <- reticulate::py_available(initialize = FALSE)
if (py_inited) {
  cfg <- reticulate::py_config()
  warn("Python was already initialized at: ", cfg$python)
  if (!reticulate::py_module_available("tensorflow") ||
      !reticulate::py_module_available("keras")) {
    stop(
      paste0("\nPython is already initialized to an interpreter without TensorFlow/Keras.\n",
             "Please restart R and run this script first thing in a fresh session.")
    )
  } else {
    ok("Existing Python has TensorFlow + Keras; proceeding with verification.")
  }
}

# 3) Ensure a Conda installation we can use ------------------
section("Checking for Conda / Miniconda")

conda_bin <- tryCatch(reticulate::conda_binary(), error = function(e) NULL)
have_conda <- !is.null(conda_bin)

if (!have_conda) {
  section("Installing Miniconda (one-time)")
  # Newer reticulate may try to immediately create an env and hit ToS. Catch and continue.
  tryCatch(
    reticulate::install_miniconda(update = FALSE),
    error = function(e) {
      warn("install_miniconda signaled an error (often ToS/channel related). Will repair and continue.")
    }
  )
  conda_bin <- reticulate::conda_binary()
  if (is.null(conda_bin)) stop("Miniconda installation failed (no conda binary found).")
  have_conda <- TRUE
}

# 3a) Accept Conda ToS and pin channels when appropriate -----
section("Accepting Conda Terms and configuring channels")
accept_tos_and_pin_channels(conda_bin)

# 3b) Create the target env if missing -----------------------
section(sprintf("Ensuring Python env '%s'", envname))
envs <- tryCatch(reticulate::conda_list()$name, error = function(e) character())
if (!(envname %in% envs)) {
  # Minimal base; TF will be installed via pip later (best for Apple silicon etc.)
  ok(sprintf("Creating conda env '%s' ...", envname))
  reticulate::conda_create(envname, packages = c("python=3.10", "pip", "numpy"))
} else {
  ok(sprintf("Conda env '%s' already exists.", envname))
}

# 4) Install TensorFlow + Keras (Python packages) ------------
section("Installing TensorFlow + Keras into the env")
# Use method = "auto" so macOS Apple silicon gets tensorflow-macos/metal via pip;
# on other platforms, wheels are chosen appropriately. We still target the conda env.
tensorflow::install_tensorflow(
  envname = envname,
  method  = "auto",
  extra_packages = c("keras"),
  reinstall = FALSE
)

# 5) Activate env for this session (if Python not yet inited) -
if (!py_inited) {
  reticulate::use_condaenv(envname, required = TRUE)
}

# 6) Final module checks -------------------------------------
if (!reticulate::py_module_available("tensorflow"))
  stop("TensorFlow Python module not found after installation.")

if (!reticulate::py_module_available("keras"))
  stop("Keras (v3) Python module not found after installation.")

tf <- reticulate::import("tensorflow")
k  <- reticulate::import("keras")
ok(paste("TensorFlow version:", as.character(tf$`__version__`)))
ok(paste("Keras version:",      as.character(k$`__version__`)))

# 7) Load the chosen R API (keras3 or legacy keras) ----------
if (use_keras3) {
  suppressPackageStartupMessages(library(keras3))
} else {
  suppressPackageStartupMessages(library(keras))
}

# 8) Quick training smoke test -------------------------------
section("Verifying by training a tiny Keras model")
set.seed(123)
n <- 100L
p <- 10L
x <- matrix(rnorm(n * p), ncol = p)
cls <- ifelse(rowSums(x[, 1:2]) + rnorm(n, sd = 0.5) > 0, 1L, 0L)

to_categorical_fn <- if (use_keras3) keras3::to_categorical else keras::to_categorical
y <- to_categorical_fn(cls, num_classes = 2L)

if (use_keras3) {
  model <- keras_model_sequential() |>
    layer_dense(units = 16, activation = "relu", input_shape = p) |>
    layer_dense(units = 2, activation = "softmax")
  model |> compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")
  history <- model |> fit(x, y, epochs = 2, batch_size = 16, verbose = 2)
} else {
  model <- keras::keras_model_sequential() |>
    keras::layer_dense(units = 16, activation = "relu", input_shape = p) |>
    keras::layer_dense(units = 2, activation = "softmax")
  model |> keras::compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")
  history <- model |> keras::fit(x, y, epochs = 2, batch_size = 16, verbose = 2)
}

ok("Tiny model trained successfully.")

# 9) Show available devices ----------------------------------
section("Available TensorFlow devices")
devices <- tryCatch(tf$config$list_physical_devices(), error = function(e) list())
if (length(devices)) {
  for (d in devices) msg(paste("-", d$device_type, d$name))
} else {
  msg("(No device list reported; CPU execution is still fine.)")
}

ok(sprintf("Setup complete. You're ready to run %s models from R.",
           if (use_keras3) "keras3" else "keras"))
# End of script
