#include <iostream>
#include "MatLib.h"
using namespace std;
int main(){ 
	int epoch = 4000; float lr = 0.03;
	vector<int> hidden_nodes = {16, 8};
	Mat X, Y; X.loadmat("Dataset2/X_MLP_LG2.txt"); Y.loadmat("Dataset2/Y_MLP_LG2.txt");
	vector<Mat> W(hidden_nodes.size() + 1), B(hidden_nodes.size() + 1);
	Loss_History history;    
	// Mat Y_mean(1, Y.col), Y_std(1, Y.col);
	// // FeatureScaling(Y, Y, Y_mean, Y_std, "standard");
	StartTimer();
	MLP(X, Y, W, B, hidden_nodes, lr, epoch, history, "BCE", 1e-7, "L2", 0.0f);
	StopTimer();
	history.print_final();
	cout << "Final W:\n"; W[W.size() - 1].print();
	cout << "Final B:\n"; B[B.size() - 1].print();
	PrintTimer();
	Mat X_Test, Y_Test; X_Test.loadmat("Dataset2/X_MLP_LG2_TEST.txt"); Y_Test.loadmat("Dataset2/Y_MLP_LG2_TEST.txt");
	Mat Y_Pred(Y_Test.row, Y_Test.col);
	Forward_Pass_ReLU(X_Test, W, B, Y_Pred, "BCE");
	// Rescale_Y(Y_Pred,Y_mean,Y_std);
	cout << "\nPredicted    Exact:\n";
	int cnt = 0;
	for (int i = 0; i < Y_Pred.row; i+=1) {
		for (int j = 0; j < Y_Pred.col; j++) {
			// cout << Y_Pred(i, j) << "\t\t" << Y_Test(i, j) << "   ";
			// if (fabsf(Y_Test(i, j) - Y_Pred(i, j)) < 0.2 * fabsf(Y_Test(i, j)) || fabsf(Y_Test(i, j) - Y_Pred(i, j)) < 0.5f) cnt++;
			if ((Y_Pred(i, j) > 0.5) == Y_Test(i,j)) cnt++;
		}
		// cout << endl;
	}
	cout << "Correct Predictions: " << cnt << "/" << Y_Test.size() << endl;
}

