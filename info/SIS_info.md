# SIS
Select all combinations from the given initial features $`x`$ and operators to create new features.   
They are fed into the model_score and the how_many_to_save features are stored in order of increasing score.   
The flow is shown in the diagram below.   

<div align="center"><img width="500" alt="image" src="SIS_1.png"></div>

First, the `base data` (initial features) are stored as features with operator number 0 (`n_op = 0`) and the expression (`eq`) is stored in `saved data` for use when increasing the number of operators.   
It is then fed into `model_score`, `score1,2` is calculated, compared (`sort`) with the other `score1,2` already calculated and saved if it is in the upper `how_many_to_save`.   
Then, for operator number 1 (`n_op = 1`), the expression is created by combining the operator with `eq` stored in `saved data` with `op=0` and repeating in the same way.   
When the operator number `max_n_op` (`n_op == max_n_op`), the `saved data` is not saved in `saved data` as it is not used afterwards.   

The broad calculations are described above, but if the calculations are carried out without any elaboration, the following double calculations occur.   

$$ a+b=b+a\  ,\  \  \frac{a+b}{b+c} =\frac{a-c}{b+c} +1\  ,\  \  abc+bc=\left( 1+a \right) bc\  \  ... $$

This time, these duplicates are not calculated based on the pre-calculated `preprocessed_results`.   

<div align="center"><img width="500" alt="image" src="SIS_2.png"></div>

The above duplicate deletion allows (I believe) the deletion of duplicate expressions for expressions composed of (+,-,*,/), no matter what the initial feature values are.   
However, when constants such as $`\pi`$ are included in the initial features   

$$ a=\pi \  \  ,\  \  a\times b=\pi b $$

and are duplicated in e.g. classification problems. This is calculated because the overlap is not known until the initial features are determined.   
Also, $`2a,a+1`$ and so on are considered the same as $`a`$ and the SCORE is not calculated (although it is saved in `save data`) for both classification and regression problems.   
Also, $`\text{exp} \left( a+b \right)`$ and so on are actually `op = 2` but are treated as new features with `op = 0` when judged by `preprocessed_results`.   
Therefore, in calculations with unary operators in them (although they are removed to some extent), duplication occurs.   
