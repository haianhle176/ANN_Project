#include <iostream>
#include "MatLib.h"
using namespace std;


// int main(){
//     string loss_type = "MAE";
//     int epoch = 120; float lr = 0.1f;
//     vector<int> hidden_nodes = {128}; Loss_History history;
//     Mat X, Y, X_Val, Y_Val, X_Test, Y_Test;
//     X.loadmat("Dataset2/X_Train_House.txt");
//     Y.loadmat("Dataset2/Y_Train_House.txt");
//     X_Val.loadmat("Dataset2/X_Valid_House.txt");
//     Y_Val.loadmat("Dataset2/Y_Valid_House.txt");
//     X_Test.loadmat("Dataset2/X_Test_House.txt");
//     Y_Test.loadmat("Dataset2/Y_Test_House.txt");
//     auto model = MLP(hidden_nodes, loss_type, 0.5f, 1000);

//     StartTimer();
//     model.fit(X, Y,  lr, epoch, history);
//     // model.k_fold(X, Y, 5, true, lr, epoch);
//     StopTimer(); history.print_final();

//     Mat Y_Pred = model.predict(X_Test);
//     // ShowSoftmaxPredict(Y_Pred, Y_Test);
//     model.evaluate(Y_Test, Y_Pred);

//     PrintTimer();
// }
int main(){
    Mat X, Y, X_Test, Y_Test;
    X.loadmat("Dataset2/X_Train_House.txt");
    Y.loadmat("Dataset2/Y_Train_House.txt");
    X_Test.loadmat("Dataset2/X_Test_House.txt");
    Y_Test.loadmat("Dataset2/Y_Test_House.txt");
    // X.loadmat("Dataset3/X_CCE.txt",1,500);
    // Y.loadmat("Dataset3/Y_CCE.txt",1,500);
    // X_Test.loadmat("Dataset3/X_CCE.txt",501,600);
    // Y_Test.loadmat("Dataset3/Y_CCE.txt",501,600);
    auto model = MLP({128}, "MAE", 0.005f, 1000, 0.5f, 100);
    // auto model = LinearRegression("MAE", 0.2f, 1200, 100);
    StartTimer();
    model.fit(X, Y);
    // model.k_fold(X, Y, 5, true);
    StopTimer();
    model.history.print_final();
    Mat Y_Pred = model.predict(X_Test);
    // ShowPredict(Y_Pred, Y_Test, "reg");
    model.evaluate(Y_Test, Y_Pred);
    // model.feature_importance(X_Test,Y_Test);
    PrintTimer();
}