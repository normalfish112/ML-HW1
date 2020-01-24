# ML-HW1

主要問題:
1.在HW1-try中的temp-2var.py，第56行可以看出bias為一個(5652, 1)的矩陣，而85行test_x.dot(w)則為一個(240, 1)的矩陣。故85行會產生error。
2.bias感覺應該是要一個通用值(像第46行那樣)，然而在第56-58行做gradient deescent的時侯，會變成一個各個值都不同的矩陣。

