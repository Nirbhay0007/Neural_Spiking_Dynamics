#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cstring>
using namespace std;

// 1. QUICK SORT + BINARY SEARCH
void quickSort(vector<int>& a, int low, int high) {
    if (low < high) {
        int pivot = a[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (a[j] < pivot) {
                i++;
                swap(a[i], a[j]);
            }
        }

        swap(a[i + 1], a[high]);
        int pi = i + 1;

        quickSort(a, low, pi - 1);
        quickSort(a, pi + 1, high);
    }
}

int binarySearch(vector<int>& a, int key) {
    int low = 0, high = a.size() - 1;

    while (low <= high) {
        int mid = (low + high) / 2;

        if (a[mid] == key)
            return mid;
        else if (a[mid] < key)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}


// 2. MERGE SORT
void merge(vector<int>& a, int low, int mid, int high) {
    vector<int> temp;
    int i = low, j = mid + 1;

    while (i <= mid && j <= high) {
        if (a[i] <= a[j])
            temp.push_back(a[i++]);
        else
            temp.push_back(a[j++]);
    }

    while (i <= mid)
        temp.push_back(a[i++]);

    while (j <= high)
        temp.push_back(a[j++]);

    for (int k = low; k <= high; k++)
        a[k] = temp[k - low];
}

void mergeSort(vector<int>& a, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        mergeSort(a, low, mid);
        mergeSort(a, mid + 1, high);
        merge(a, low, mid, high);
    }
}

// 3. HEAP SORT

void heapify(vector<int>& a, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && a[left] > a[largest])
        largest = left;

    if (right < n && a[right] > a[largest])
        largest = right;

    if (largest != i) {
        swap(a[i], a[largest]);
        heapify(a, n, largest);
    }
}

void heapSort(vector<int>& a) {
    int n = a.size();

    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(a, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(a[0], a[i]);
        heapify(a, i, 0);
    }
}

// 4. KNAPSACK USING GREEDY

struct Item {
    int value, weight;
};

bool compare(Item a, Item b) {
    double r1 = (double)a.value / a.weight;
    double r2 = (double)b.value / b.weight;
    return r1 > r2;
}

double fractionalKnapsack(int W, vector<Item>& items) {
    sort(items.begin(), items.end(), compare);

    double totalValue = 0.0;

    for (auto item : items) {
        if (W >= item.weight) {
            totalValue += item.value;
            W -= item.weight;
        } else {
            totalValue += item.value * ((double)W / item.weight);
            break;
        }
    }

    return totalValue;
}

// 5. DIJKSTRA ALGORITHM

void dijkstra(int graph[10][10], int n, int src) {
    int dist[10];
    bool visited[10];

    for (int i = 0; i < n; i++) {
        dist[i] = INT_MAX;
        visited[i] = false;
    }

    dist[src] = 0;

    for (int count = 0; count < n - 1; count++) {
        int u = -1;
        int minDist = INT_MAX;

        for (int i = 0; i < n; i++) {
            if (!visited[i] && dist[i] < minDist) {
                minDist = dist[i];
                u = i;
            }
        }

        visited[u] = true;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && graph[u][v] &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    cout << "Vertex\tDistance\n";
    for (int i = 0; i < n; i++)
        cout << i << "\t" << dist[i] << endl;
}

// 6. PRIM'S ALGORITHM

void primMST(int graph[10][10], int n) {
    int parent[10];
    int key[10];
    bool mstSet[10];

    for (int i = 0; i < n; i++) {
        key[i] = INT_MAX;
        mstSet[i] = false;
    }

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < n - 1; count++) {
        int u = -1;
        int minKey = INT_MAX;

        for (int i = 0; i < n; i++) {
            if (!mstSet[i] && key[i] < minKey) {
                minKey = key[i];
                u = i;
            }
        }

        mstSet[u] = true;

        for (int v = 0; v < n; v++) {
            if (graph[u][v] && !mstSet[v] &&
                graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    cout << "Edge \tWeight\n";
    for (int i = 1; i < n; i++)
        cout << parent[i] << " - " << i
             << "\t" << graph[i][parent[i]] << endl;
}

// 7. KRUSKAL ALGORITHM

struct Edge {
    int u, v, w;
};

bool edgeCompare(Edge a, Edge b) {
    return a.w < b.w;
}

int parentSet[100];

int findSet(int i) {
    if (parentSet[i] == i)
        return i;
    return parentSet[i] = findSet(parentSet[i]);
}

void unionSet(int x, int y) {
    parentSet[findSet(x)] = findSet(y);
}

void kruskal(vector<Edge>& edges, int vertices) {
    sort(edges.begin(), edges.end(), edgeCompare);

    for (int i = 0; i < vertices; i++)
        parentSet[i] = i;

    cout << "Edges in MST:\n";

    for (auto e : edges) {
        if (findSet(e.u) != findSet(e.v)) {
            cout << e.u << " - " << e.v
                 << " : " << e.w << endl;
            unionSet(e.u, e.v);
        }
    }
}

// 8. 0/1 KNAPSACK USING DP

int knapsackDP(int W, vector<int>& wt,
               vector<int>& val, int n) {

    int dp[100][100];

    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {

            if (i == 0 || w == 0)
                dp[i][w] = 0;

            else if (wt[i - 1] <= w)
                dp[i][w] = max(
                    val[i - 1] +
                    dp[i - 1][w - wt[i - 1]],
                    dp[i - 1][w]
                );

            else
                dp[i][w] = dp[i - 1][w];
        }
    }

    return dp[n][W];
}

// 9 & 10. TSP USING DP

int tsp(int graph[10][10], int mask, int pos,
        int n, vector<vector<int>>& dp) {

    if (mask == (1 << n) - 1)
        return graph[pos][0];

    if (dp[mask][pos] != -1)
        return dp[mask][pos];

    int ans = INT_MAX;

    for (int city = 0; city < n; city++) {
        if (!(mask & (1 << city))) {
            ans = min(ans,
                graph[pos][city] +
                tsp(graph, mask | (1 << city),
                    city, n, dp));
        }
    }

    return dp[mask][pos] = ans;
}

// 11. LONGEST COMMON SUBSEQUENCE

int LCS(string X, string Y) {
    int m = X.length();
    int n = Y.length();

    int dp[100][100];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {

            if (i == 0 || j == 0)
                dp[i][j] = 0;

            else if (X[i - 1] == Y[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;

            else
                dp[i][j] = max(dp[i - 1][j],
                               dp[i][j - 1]);
        }
    }

    return dp[m][n];
}

// 12 & 13. N-QUEEN

bool isSafe(vector<vector<int>>& board,
            int row, int col, int n) {

    for (int i = 0; i < col; i++)
        if (board[row][i])
            return false;

    for (int i = row, j = col;
         i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return false;

    for (int i = row, j = col;
         j >= 0 && i < n; i++, j--)
        if (board[i][j])
            return false;

    return true;
}

bool solveNQueen(vector<vector<int>>& board,
                 int col, int n) {

    if (col >= n)
        return true;

    for (int i = 0; i < n; i++) {

        if (isSafe(board, i, col, n)) {

            board[i][col] = 1;

            if (solveNQueen(board, col + 1, n))
                return true;

            board[i][col] = 0;
        }
    }

    return false;
}

// 14. SUM OF SUBSETS

void subsetSum(vector<int>& set, int n,
               int sum, int index,
               vector<int>& subset) {

    if (sum == 0) {
        cout << "{ ";
        for (int x : subset)
            cout << x << " ";
        cout << "}\n";
        return;
    }

    if (index == n || sum < 0)
        return;

    subset.push_back(set[index]);

    subsetSum(set, n, sum - set[index],
              index + 1, subset);

    subset.pop_back();

    subsetSum(set, n, sum,
              index + 1, subset);
}

// 15. NAIVE STRING MATCHING

void naiveSearch(string text, string pattern) {
    int n = text.length();
    int m = pattern.length();

    for (int i = 0; i <= n - m; i++) {

        int j;

        for (j = 0; j < m; j++) {
            if (text[i + j] != pattern[j])
                break;
        }

        if (j == m)
            cout << "Pattern found at index "
                 << i << endl;
    }
}

// 16. KMP STRING MATCHING

void computeLPS(string pat, vector<int>& lps) {
    int len = 0;
    int i = 1;

    while (i < pat.length()) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0)
                len = lps[len - 1];
            else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(string text, string pat) {
    int n = text.length();
    int m = pat.length();

    vector<int> lps(m, 0);
    computeLPS(pat, lps);

    int i = 0, j = 0;

    while (i < n) {

        if (pat[j] == text[i]) {
            i++;
            j++;
        }

        if (j == m) {
            cout << "Pattern found at index "
                 << i - j << endl;
            j = lps[j - 1];
        }

        else if (i < n && pat[j] != text[i]) {

            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
}

// MAIN FUNCTION

int main() {

    cout << "All Algorithms Implemented Successfully!\n";

    // Example for Quick Sort
    vector<int> arr = {5, 2, 9, 1, 7};

    quickSort(arr, 0, arr.size() - 1);

    cout << "\nQuick Sort Result:\n";
    for (int x : arr)
        cout << x << " ";

    cout << endl;

    int key = 7;
    int pos = binarySearch(arr, key);

    if (pos != -1)
        cout << "Element found at index "
             << pos << endl;
    else
        cout << "Element not found\n";

    return 0;
}

