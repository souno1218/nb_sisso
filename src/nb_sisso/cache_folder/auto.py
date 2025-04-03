import numpy as np
import pandas as pd
from make_cache import main, decryption

num_threads = 56
seed = np.random.randint(10000)
# seed = 2062
# seed = 1986
# seed = 9852
# max_n_plus = 6
# for n_plus in range(1, max_n_plus + 1):
#    main(n_plus, num_threads, len_x=40, log_interval=180, seed=seed)
n_plus = 7
main(n_plus, num_threads, len_x=40, log_interval=180, seed=seed)
