// 二次元音響FDTD入門 by Yoshiki NAGATANI 20150525 (http://ultrasonics.jp/nagatani/) - Field Viewer for Scilab 5

h = scf(0);								// ウィンドウの準備（新しいウィンドウを開いて h というハンドラに割り当てる）
set(h, 'color_map',jetcolormap(64));	// 64 段階の Jet という色マップに設定

interval = 50;				// ステップ刻み
Nstep = 1000;				// 総ステップ数
image_intensity = 2000;		// 画像表示の明るさ

// ファイル名のループ - ↓ここから↓ （このループを使う場合は 14 行目はコメントアウト）
for n = 0:interval:Nstep

	// 音場ファイル読み込み
//	n = 100;
	txtfilename = sprintf("field%06d.txt",n);
	field = read(txtfilename,-1,400);

	// 音場の表示（64を超えないように min で頭を押さえる）
	Matplot( min(64,abs(field*image_intensity)), '031', rect=[0,0,size(field,2),size(field,1)] );

	// 見栄えを色々・・
	title(['step: ', string(n), ' / ', string(Nstep)]);
	xlabel('y direction');
	ylabel('x direction');

	// 画像ファイル保存
	imgfilename = sprintf("field%06d.png",n);
	xs2png(h,imgfilename);

end
// ファイル名のループ - ↑ここまで↑
