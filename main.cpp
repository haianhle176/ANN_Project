#include <iostream>
#include "MatLib.h"
using namespace std;


int main(){
    string loss_type = "MAE";
    int epoch = 120; float lr = 0.1f;
    vector<int> hidden_nodes = {128}; Loss_History history;
    Mat X, Y, X_Test, Y_Test;
    X.loadmat("Dataset2/X_Train_House.txt");
    Y.loadmat("Dataset2/Y_Train_House.txt");
    X_Test.loadmat("Dataset2/X_Test_House.txt");
    Y_Test.loadmat("Dataset2/Y_Test_House.txt");
    Mat Y_mean(1,Y.col), Y_std(1,Y.col);
    if (loss_type == "MSE" || loss_type == "MAE") FeatureScaling(Y, Y, Y_mean, Y_std);
    vector<Mat> W(hidden_nodes.size() + 1), B(hidden_nodes.size() + 1);

    StartTimer();
    MLP(X, Y, W, B, hidden_nodes, lr, epoch, history, loss_type , 0.5f, 1000);
    // K_Fold_MLP(X, Y, 5, true, hidden_nodes, lr, epoch, loss_type , 0.5f, 1000);
    StopTimer(); history.print_final();

    Mat Y_Pred(Y_Test.row, Y_Test.col);
    MLP_Test(X_Test, W, B, Y_Pred, loss_type, Y_mean, Y_std);
	// ShowSoftmaxPredict(Y_Pred, Y_Test);
    ModelEvaluation(Y_Test, Y_Pred, loss_type);

    PrintTimer();
}