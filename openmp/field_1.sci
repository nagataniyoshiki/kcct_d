// 二次元音響FDTD入門 by Yoshiki NAGATANI 20150525 (http://ultrasonics.jp/nagatani/) - Simplest Viewer for Scilab 5

// ウィンドウの準備 & 64 段階の Jet という色マップに設定
set(scf(), 'color_map',jetcolormap(64));

// 音場ファイル読み込み
field = read('field000200.txt',-1,400);

// 音場の表示
Matplot( abs(field*500) );

