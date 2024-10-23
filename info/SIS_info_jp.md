# SIS
与えられた初期特徴量$`x`$とoperatorsからすべての組み合わせを選択し、新しい特徴を作成する。   
それらをmodel_scoreに投入し、スコアの高い順にhow_many_to_save個の特徴量を保存する。   
流れとしては以下の図のようになる。   

<div align="center"><img width="500" alt="image" src="https://i.imgur.com/33gcIFq.png"></div>

まず、`base data`(初期特徴量)を演算子数0(`n_op = 0`)の特徴量として、演算子数を増やす際に使うために`saved data`に式(`eq`)を保存する。   
その後`model_score`に投入し、`score1,2`を算出し、計算済みの他の`score1,2`と比較(`sort`)し、上位`how_many_to_save`に入れば保存する。   
その後、演算子数1(`n_op = 1`)では、`op=0`で`saved data`に保存した`eq`と演算子を組み合わせて式を作成して、同様に繰り返す。   
演算子数`max_n_op`(`n_op == max_n_op`)の際にはその後`saved data`を使うことがないため、`saved data`に保存しない。   

大まかな計算内容は上記の通りですが、何の工夫もなく計算を行うと以下のような二重の計算が発生する。   

$$ a+b=b+a\  ,\  \  \frac{a+b}{b+c} =\frac{a-c}{b+c} +1\  ,\  \  abc+bc=\left( 1+a \right) bc\  \  ... $$

今回はこれら重複を事前に計算した`preprocessed_results`に基づいて計算が行われないようにする。   

<div align="center"><img width="500" alt="image" src="https://i.imgur.com/WFFQCG5.png"></div>

上記の重複削除によって(+,-,*,/)で構成された式について初期特徴量がどんな値であっても重複する式は削除できる(と思っている)。   
ただし、初期特徴量に定数$`\pi`$などを入れた際には   

$$ a=\pi \  \  ,\  \  a\times b=\pi b $$

で、分類問題などでは重複している。これは初期特徴量が決まるまで重複しているかわからないため、計算される。   
また、$`2a,a+1`$などは$`a`$と同じとみなされ、分類問題、回帰問題のどちらでも(`save data`に保存はされるが)scoreの計算はされない。   
また、$`\text{exp} \left( a+b \right)`$などは実際には`op = 2`だが`preprocessed_results`で判定する際には`op = 0`の新しい特徴量として扱う。   
そのため、単項演算子が入った計算では(ある程度取り除いているが)重複が生じる。   
