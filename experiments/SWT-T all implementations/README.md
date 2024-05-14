# Sliding window method
Original method used in referenced research paper

# Wavelet Data Leakage Test
Testing possible signs of data leakage due to wavelet transformation

# Prediction Window Method
Use to tackle data leakage if we have no information of the prediction zone

# Transformer Variants
Data leakage accounted for
Test different transformer models from the original models directory, BaseTransformer, BBFixed, Sparse
Test different combinations of inputs and outputs

# Traditional Statistical Methods
ARIMA, BATS, etc

# Darts Transformer
No wavelet transformation used
Use the library DARTS to create a transformer

# multi MODWT Transformer
Use MODWT according to reference github page:
https://github.com/CapWidow/W-Transformer

They use historical forecast instead.

# Historical forecast
Tested Historical Forecast function provided by Darts Library

# Notes
Darts Transformer is currently using an input chunk of 50 and output chunk of 1
Given an output chunk of 1, essentially in the prediction zone, we will encounter the problem of data leakage as the prediction is done in a rolling window method

Historical forecast using only 12 input chunks to predict 1 output chunk
Higher input chunk length increases lookback period and therefore increases performance. But it does come with higher training time.

# Prediction methods (Data Leakage)
Prediction Zone of 1 day (1 time step at a time)
 - Sliding window

Prediction Zone of 1 day (no information of the prediction zone)
 - Prediction window