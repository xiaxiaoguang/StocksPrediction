import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def IC(prediction, real_value, null_val: float = np.nan):
    """
    Information Coefficient (Pearson Correlation between predicted and actual prices).
    Measures cross-sectional forecasting accuracy across all N assets at the final step.
    """
    # 1. Dimensionality checks
    if prediction.dim() == 4 and prediction.shape[3] != 1:
        prediction = prediction[:, :, :, 1]
    if real_value.dim() == 4 and real_value.shape[3] != 1:
        real_value = real_value[:, :, :, 1]
    if prediction.shape != real_value.shape:
        prediction = prediction.squeeze(-1)
        
    # 2. Extract final step price
    prediction_signal = prediction[:, -1] # Shape: [B, N]
    real_signal = real_value[:, -1]       # Shape: [B, N]
    
    # 3. Calculate means along the asset dimension (dim=1)
    pred_mean = torch.mean(prediction_signal, dim=1, keepdim=True)
    real_mean = torch.mean(real_signal, dim=1, keepdim=True)
    
    # 4. Center the variables
    pred_centered = prediction_signal - pred_mean
    real_centered = real_signal - real_mean
    
    # 5. Covariance and Standard Deviations
    cov = torch.sum(pred_centered * real_centered, dim=1)
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) + 1e-8)
    real_std = torch.sqrt(torch.sum(real_centered ** 2, dim=1) + 1e-8)
    
    # 6. Pearson Correlation (IC) for each batch
    ic_per_batch = cov / (pred_std * real_std + 1e-8)
    
    return torch.mean(ic_per_batch)


def MDD(K=10):
    """
    Maximum Drawdown metric (Additive).
    Evaluates the worst-case drop of the Top-K portfolio over the batch dimension,
    treating the standardized prices as additive strategy scores.
    """
    def _MDD(prediction, real_value, null_val: float = np.nan):
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if prediction.shape != real_value.shape:
            prediction = prediction.squeeze(-1)
            
        prediction_signal = prediction[:, -1]
        real_signal = real_value[:, -1]
        
        B = prediction_signal.shape[0]
        
        # Select Top-K stocks based on highest predicted close price
        _, topKindices = torch.topk(prediction_signal, dim=1, k=K)
        topKindices = topKindices.view(-1) 
        indices_array = torch.repeat_interleave((torch.arange(B)), K)
        
        # Average actual price/score of our selected Top-K portfolio at each step
        step_returns = real_signal[indices_array, topKindices].view(B, K).mean(dim=1)
        
        # Calculate cumulative additive wealth over the B dimension (treating B as time)
        cum_wealth = torch.cumsum(step_returns, dim=0)
        
        # Calculate Running Maximum (High Water Mark)
        running_max = torch.cummax(cum_wealth, dim=0)[0]
        
        # Calculate Additive Drawdowns (Distance from peak)
        drawdowns = running_max - cum_wealth
        
        # Return the Maximum Drawdown
        return torch.max(drawdowns)
    return _MDD

def SR(K):
    def Sharperatio(prediction, real_value, null_val: float=np.nan):
        # 1. Dimensionality checks
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if prediction.shape != real_value.shape:
            prediction = prediction.squeeze(-1)

        # 2. Extract final step price (No more compounding 1+r)
        prediction_signal = prediction[:, -1] # Shape: [B, N]
        real_signal = real_value[:, -1]       # Shape: [B, N]
        
        B = prediction_signal.shape[0]
        
        # 3. Greedy: Sort by highest predicted price
        _, topKindices = torch.topk(prediction_signal, dim=1, k=K)
        topKindices = topKindices.view(-1) # B * K
        indices_array = torch.repeat_interleave(torch.arange(B), K)
        
        # 4. Evaluate using the actual prices of chosen assets
        Expected = torch.mean(real_signal[indices_array, topKindices])
        Var = torch.var(real_signal[indices_array, topKindices])

        return Expected / (torch.sqrt(Var) + 1e-8)
    return Sharperatio

def predReturn(K):
    def Prediction_Return(prediction, real_value, buy_price, null_val: float = np.nan):
        # Handle dimensions for all inputs
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if buy_price.dim() == 4 and buy_price.shape[3] != 1:
            buy_price = buy_price[:, :, :, 1]
            
        if prediction.shape != real_value.shape:
            prediction = prediction.squeeze(-1)
            
        prediction_signal = prediction[:, -1]
        real_future = real_value[:, -1]
        real_buy = buy_price[:, -1]
        # Calculate the actual percentage return based on buy price and future price
        real_signal = (real_future - real_buy) / real_buy
        
        B = prediction_signal.shape[0]
        _, topKindices = torch.topk(prediction_signal, dim=1, k=K)
        topKindices = topKindices.view(-1)
        indices_array = torch.repeat_interleave((torch.arange(B)), K)
        
        # Get the actual returns of the chosen stocks
        chosen_returns = real_signal[indices_array, topKindices].view(B, K)
        
        # Total profit from $1 is the sum of (1/k * return) across the K chosen stocks
        portfolio_profit = torch.sum((1.0 / K) * chosen_returns, dim=1)
        
        # Final value of the $1 investment, averaged across the batch
        FinalMoney = torch.mean(1.0 + portfolio_profit)
        return FinalMoney
    return Prediction_Return

def predReturn_Short(K):
    def Prediction_Return(prediction, real_value, buy_price, null_val: float = np.nan):
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if buy_price.dim() == 4 and buy_price.shape[3] != 1:
            buy_price = buy_price[:, :, :, 1]
            
        if prediction.shape != real_value.shape:
            prediction = prediction.squeeze(-1)
            
        prediction_signal = prediction[:, -1]
        real_future = real_value[:, -1]
        real_buy = buy_price[:, -1]
        # Calculate the actual percentage return
        real_signal = (real_future - real_buy) / real_buy
        
        B = prediction_signal.shape[0]
        indices_array = torch.repeat_interleave((torch.arange(B)), K)

        # For shorting, we care about strongest absolute signals
        _, topKindices = torch.topk(torch.abs(prediction_signal), dim=1, k=K)
        topKindices = topKindices.view(-1)
        chosen_preds = prediction_signal[indices_array, topKindices].view(B, K)
        chosen_returns = real_signal[indices_array, topKindices].view(B, K)
        
        # Long gives real return, Short gives inverted real return
        returns = torch.where(chosen_preds > 0, chosen_returns, -chosen_returns)
        
        # Calculate portfolio profit and add back the initial $1
        portfolio_profit = torch.sum((1.0 / K) * returns, dim=1)
        FinalMoney = torch.mean(1.0 + portfolio_profit)
        return FinalMoney
    return Prediction_Return

def bstReturn(K):
    def Best_Return(prediction, real_value, buy_price, null_val: float = np.nan):
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if buy_price.dim() == 4 and buy_price.shape[3] != 1:
            buy_price = buy_price[:, :, :, 1]
            
        real_future = real_value[:, -1]
        real_buy = buy_price[:, -1]
        
        # Calculate actual returns to find the absolute best stocks to have bought
        real_signal = (real_future - real_buy) / real_buy
        
        # Top K is now based on the best actual percentage returns, not just highest raw price
        realtopK_returns, _ = torch.topk(real_signal, dim=1, k=K) 
        
        # 1 dollar invested perfectly (1/k per top performing stock)
        portfolio_profit = torch.sum((1.0 / K) * realtopK_returns, dim=1)
        BestMoney = torch.mean(1.0 + portfolio_profit)
        return BestMoney
    return Best_Return

def RndReturn(K):
    def Rand_Return(prediction, real_value, buy_price, null_val: float = np.nan):
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if buy_price.dim() == 4 and buy_price.shape[3] != 1:
            buy_price = buy_price[:, :, :, 1]
            
        if prediction.shape != real_value.shape:
            prediction = prediction.squeeze(-1)
            
        real_future = real_value[:, -1]
        real_buy = buy_price[:, -1]
        
        # Calculate actual returns
        real_signal = (real_future - real_buy) / real_buy
        
        B = prediction.shape[0]
        N = real_signal.shape[1] 
        
        indices_array = torch.repeat_interleave((torch.arange(B)), K)
        randomchoice = torch.randint(low=0, high=N, size=(B * K,))
        
        chosen_returns = real_signal[indices_array, randomchoice].view(B, K)
        
        # 1 dollar invested randomly (1/k per random stock)
        portfolio_profit = torch.sum((1.0 / K) * chosen_returns, dim=1)
        RandomMoney = torch.mean(1.0 + portfolio_profit)
        return RandomMoney
    return Rand_Return

def returnK(K):
    def return_(prediction, real_value, buy_price, null_val: float = np.nan):
        if prediction.dim() == 4 and prediction.shape[3] != 1:
            prediction = prediction[:, :, :, 1]
        if real_value.dim() == 4 and real_value.shape[3] != 1:
            real_value = real_value[:, :, :, 1]
        if buy_price.dim() == 4 and buy_price.shape[3] != 1:
            buy_price = buy_price[:, :, :, 1]
            
        if prediction.shape != real_value.shape:
            prediction = prediction.squeeze(-1)
            
        prediction_signal = prediction[:, -1]
        real_future = real_value[:, -1]
        real_buy = buy_price[:, -1]
        
        # Calculate actual returns
        real_signal = (real_future - real_buy) / real_buy
        
        B = prediction_signal.shape[0]
        _, topKindices = torch.topk(prediction_signal, dim=1, k=K)
        topKindices = topKindices.view(-1)
        indices_array = torch.repeat_interleave(torch.arange(B), K)
        
        # Get the returns for both the best K and our chosen K, shaped as [B, K]
        realtopK_returns, _ = torch.topk(real_signal, dim=1, k=K)
        chosen_returns = real_signal[indices_array, topKindices].view(B, K)
        
        # Calculate the final portfolio value for both: 1 dollar + sum(1/k * return)
        best_portfolio_value = 1.0 + torch.sum((1.0 / K) * realtopK_returns, dim=1)
        our_portfolio_value = 1.0 + torch.sum((1.0 / K) * chosen_returns, dim=1)
        
        # Regret/Error: the mean absolute difference between ideal portfolio and our portfolio
        dif2 = torch.mean(torch.abs(best_portfolio_value - our_portfolio_value))
        return dif2
    return return_