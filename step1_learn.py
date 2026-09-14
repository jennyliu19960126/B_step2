from core_func import GpLearnStock
if __name__ == '__main__': 
    gp = GpLearnStock()
    gp.set_params_backtest(need_parallel=True) 
    gp.prep_data_backtest() 
    gp.learn_formulation() 
    
