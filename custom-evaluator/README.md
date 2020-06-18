# Using custom evaluator

Default truth annotation is [val.json](val.json)(val dataset) and default answers annotation is [ans.json](ans.json)(Generated using Rohit's solution.py with resnet101_csv_05.h5 model). 

1. **Make sure you `cd` to this folder first**
2. Run `main.py`.
3. Pass custom annotation paths with `--truth_path` and `--ans_path` 

# Viewing graph of precision against recall
1. You may view the graph with [matplot.ipynb](matplot.ipynb). You can pass custom annotation paths in the `notebook_func` argument

# Notes
The ans.json `image_id`'s and `category_id`'s are off by -1 and +1 respectively (because Rohit didn't push his fixes in solution.py!). So I have temporarily hardcoded them in [evaluator.py](evaluator.py)'s *evaluate* function (line 25)
