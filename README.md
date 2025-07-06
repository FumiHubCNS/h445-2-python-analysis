# H445実験用Python解析コード群

H445実験の解析コードのうちPythonで書かれたものをまとめたディレクトリです。

## 環境整備

sahoでは`curl`などのコマンドが使えないことに加え、`uv`で使用するpythonのバイナリがuserのローカルに必要らしい。
(でないと`uv`コマンドでネットワークに通信する時にブロックされる)

このような状況から`uv`を使えるようにするのは少しめんどくさいですよね。

一番簡単なのはバイナリを自分のPCにダウンロードしたのちにsahoに移し、展開する方法だと思います。

バージョンなどは自身で最適なものを選んでくれれば良いですが、ひとまず筆者は以下のバージョンをインストールしました。

[uv-x86_64-unknown-linux-musl.tar.gz](https://github.com/astral-sh/uv/releases/download/0.7.19/uv-x86_64-unknown-linux-musl.tar.gz)
[Python-3.13.0.tgz](https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz)

これらをsahoに移したとに以下のような適当なコマンドで展開してください。

```zsh
tar xvf [file]
```

あとは`.zshrc`などにパスを追加すれば使えるはずです。

```zsh
export PATH="[install path]/uv-x86_64-unknown-linux-musl:$PATH"
export PATH="[install path]/python-3.13/bin:$PATH"
```

## VNCサーバーの設定など

comming soon...

## スクリプトの説明

comming soon...



