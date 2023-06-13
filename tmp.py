def get_experiments(args,
                    pool,
                    params:dict):
    args = (a,b, config)
    result = pool.apply_async(calibrate, args).get()
    
    for 