#include <iostream>
#include <vector>

using namespace std;

int N, C;
vector<int> W, V;
vector< vector<int> > dp, used;


int f(int k, int c){
    if (k == -1) return 0;
    if (dp[k][c] != -1) return dp[k][c];
    int ret = f(k-1, c);
    if (V[k] <= c){
        int v1 = W[k] + f(k-1, c-V[k]);
        if (v1 > ret){
            ret = v1;
            used[k][c] = 1;
        }
    }
    return dp[k][c] = ret;
}


int main(){
    cin >> N >> C;
    W = vector<int>(N);
    V = vector<int>(N);
    dp = vector< vector<int> >(N, vector<int>(C+1, -1));
    used = vector< vector<int> >(N, vector<int>(C+1, 0));
    for (int i = 0; i < N; ++i) cin >> W[i];
    for (int i = 0; i < N; ++i) cin >> V[i];
    int cost = f(N-1, C);
    int cap = C;
    for (int i = N-1; i >= 0; --i){
        if (used[i][cap]){
            cout << i << " ";
            cap -= V[i];
        }
    }
    cout << C-cap << " " << cost << endl;
}
