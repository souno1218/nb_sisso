# nb_sisso
このプロジェクトは、[rouyang2017]によって開発された[SISSO](https://github.com/rouyang2017/SISSO)のいわばnumba実装です。   
元々は車輪の再発明をし、上記プロジェクトの理解を深めることを目的としていました。   
また、自身はfortranのコードが読めないため、元のコードをほとんど読めていないため、元のコードの完全な再実装ではありません。   
また、SISとSOでわけ別々に実装しているため、使用時にはそれぞれの結果をつなぐ必要があります(`info/how_to_use.ipynb`を確認)。   
自身の研究内容が分類問題で、回帰問題を取り扱う機会がなかったため、回帰問題での使用はuserによる指標の変更が必要です。   
ですが、numbaは読むだけならpython(numpy)と変わりがなく、pythonであれば書ける、読める人が多いだろうことから、参考程度にはなるかと思い公開します。   
また、pythonを書くことができれば新規指標(分類問題ではconvex hullを置き換えるもの)を作成することができます。   

オリジナル・プロジェクトの詳細については、[SISSO](https://github.com/rouyang2017/SISSO)をご覧ください。   

## Getting Started
### Prerequisites
numba,numpy,numba_progressが必要。なければ自動インストール?。

### Installing
まず、condaで区切られている場合は、仮想環境をアクティブにする。
```python
#examples
conda activate myenv
```
ダウンロードとインストール
```
pip install git+https://github.com/souno1218/nb_sisso.git
```

## Running
実行はinfo内にあるhow_to_use_jp.ipynbを見てください。

## Differences from the Original Project
このプロジェクトは[SISSO](https://github.com/rouyang2017/SISSO)と以下の点で異なる：
- python,numpy,numbaで書き直した。
- 不完全で汚いコード
- pythonで書かれていことによる、改変容易性

## Built With
* [numba](https://numba.pydata.org) - main code
* [numpy](https://numpy.org) - 様々な計算に使用
* [numba_progress](https://github.com/conda-forge/numba-progress-feedstock) - プログレスバーの表示

## Authors
* **Sonosuke Kono**

## License
This project is licensed under Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.   

## Acknowledgements
This project is based on [SISSO](https://github.com/rouyang2017/SISSO), originally developed by [Original Author(s)]. The original project is licensed under the Apache License 2.0, and a copy of the license can be found [here](http://www.apache.org/licenses/LICENSE-2.0).

Portions of this project are modifications based on work created and shared by the [rouyang2017] under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

## Finally.
このプロジェクトの基礎となった[SISSO](https://github.com/rouyang2017/SISSO)の開発者たちの素晴らしい仕事に感謝する。   

私は日本人ですが、これを書くまでGitHubを使ったことがありませんでした。  
英語が苦手なのでDeeplを使っています。  
このReedMeも以下のページを参考にして書いています。  
https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
