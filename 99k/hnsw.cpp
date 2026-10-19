#include <bits/stdc++.h>
using namespace std;

vector<float> load_fbin(const string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open " + path);
    vector<float> data(expected_count);
    in.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
    if (!in) throw runtime_error("read failed or file too short: " + path);
    return data;
}

vector<int> load_ibin(const string& path, size_t expected_count) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open " + path);
    vector<int> data(expected_count);
    in.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(int));
    if (!in) throw runtime_error("read failed or file too short: " + path);
    return data;
}

float squared_distance(const float* a, const float* b, size_t dim) {
    float dist = 0;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// Exact-match recall: counts how many of the top-k results appear in the
// ground-truth set for query i.
int count_correct(const vector<int>& res, const vector<int>& match, int i, int k) {
    int c = 0;
    for (int j = 0; j < k && j < (int)res.size(); ++j)
        for (int t = 0; t < k; ++t)
            if (res[j] == match[(size_t)i * k + t]) { c++; break; }
    return c;
}

vector<int> search_layer(const vector<float>& dataset, const vector<vector<int>>& neighbors, int dim, const float* q, int entry, int ef) {
    priority_queue<pair<float, int>> max_heap;
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> min_heap;
    vector<bool> visited(neighbors.size(), false);
    float dist = squared_distance(&dataset[(size_t)entry * dim], q, dim);
    max_heap.emplace(dist, entry);
    min_heap.emplace(dist, entry);
    visited[entry] = true;
    while (!min_heap.empty()) {
        auto [curr_dist, curr_node] = min_heap.top();
        if ((int)max_heap.size() >= ef && curr_dist > max_heap.top().first) break;
        min_heap.pop();
        for (int neighbor : neighbors[curr_node]) {
            if (visited[neighbor]) continue;
            visited[neighbor] = true;
            float neighbor_dist = squared_distance(&dataset[(size_t)neighbor * dim], q, dim);
            if ((int)max_heap.size() < ef || neighbor_dist < max_heap.top().first) {
                max_heap.emplace(neighbor_dist, neighbor);
                min_heap.emplace(neighbor_dist, neighbor);
                if ((int)max_heap.size() > ef) max_heap.pop();
            }
        }
    }
    vector<int> result;
    while (!max_heap.empty()) {
        result.push_back(max_heap.top().second);
        max_heap.pop();
    }
    reverse(result.begin(), result.end());
    return result;
}

void shrink_neighbors(vector<int>& list, const vector<float>& dataset, int dim, int x, int Mmax) {
    if ((int)list.size() <= Mmax) return;
    vector<pair<float, int>> scored;
    scored.reserve(list.size());
    for (int id : list) scored.emplace_back(squared_distance(&dataset[(size_t)id * dim], &dataset[(size_t)x * dim], dim), id);
    sort(scored.begin(), scored.end());
    list.clear();
    for (int i = 0; i < Mmax; ++i) list.push_back(scored[i].second);
}

int draw_level(mt19937& rng, int M) {
    static uniform_real_distribution<double> dist(0.0, 1.0);
    double mL = 1.0 / log((double)M);
    double u = 1.0 - dist(rng);
    int level = (int)(-log(u) * mL);
    return level;
}

void build_hnsw(const vector<float>& dataset, int n, int dim, int M, int ef_construction, const vector<int>& level, vector<vector<vector<int>>>& neighbors, int& entry, int& top) {
    entry = 0;
    top = level[0];
    for (int p = 1; p < n; ++p) {
        int L = level[p];
        int ep = entry;
        const float* q = &dataset[(size_t)p * dim];
        for (int l = top; l > L; --l) {
            auto res = search_layer(dataset, neighbors[l], dim, q, ep, 1);
            ep = res[0];
        }
        for (int l = min(top, L); l >= 0; --l) {
            auto res = search_layer(dataset, neighbors[l], dim, q, ep, ef_construction);
            int Mmax = (l == 0) ? 2 * M : M;
            for (int j = 0; j < M && j < (int)res.size(); ++j) {
                neighbors[l][p].push_back(res[j]);
                neighbors[l][res[j]].push_back(p);
                if ((int)neighbors[l][res[j]].size() > Mmax) shrink_neighbors(neighbors[l][res[j]], dataset, dim, res[j], Mmax);
            }
            ep = res[0];
        }
        if (L > top) {
            top = L;
            entry = p;
        }
    }
}

vector<int> query_hnsw(const vector<float>& dataset, const vector<vector<vector<int>>>& neighbors, int dim, const float* q, int entry, int top, int ef) {
    int ep = entry;
    for (int l = top; l >= 1; --l) {
        auto res = search_layer(dataset, neighbors[l], dim, q, ep, 1);
        ep = res[0];
    }
    return search_layer(dataset, neighbors[0], dim, q, ep, ef);
}

// Usage: test [brute_ms_per_query]
// If you pass the brute-force ms/query from your separate brute run, speedup is printed.
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(0);
    double brute_ms = (argc > 1) ? atof(argv[1]) : 0.0;

    // Set these to match your files.
    const int dim = 128, n = 99000, nq = 1000, k = 1, M = 16;
    vector<float> dataset, query;
    vector<int> match;
    try {
        dataset = load_fbin("dataset.fbin", (size_t)n * dim);
        query = load_fbin("query.fbin", (size_t)nq * dim);
        match = load_ibin("match.ibin", (size_t)nq * k);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    mt19937 rng(42);
    vector<int> level(n);
    int max_level = 0;
    for (int i = 0; i < n; ++i) {
        level[i] = draw_level(rng, M);
        max_level = max(max_level, level[i]);
    }
    vector<vector<vector<int>>> neighbors(max_level + 1, vector<vector<int>>(n));
    int entry, top;
    auto build_start = chrono::high_resolution_clock::now();
    build_hnsw(dataset, n, dim, M, 100, level, neighbors, entry, top);
    auto build_end = chrono::high_resolution_clock::now();
    cout << "HNSW build: " << chrono::duration<double, milli>(build_end - build_start).count() << " ms\n";
    cout << "entry: " << entry << "  top: " << top << "  level[entry]: " << level[entry] << "\n";
    for (int l = 0; l <= max_level; ++l) {
        size_t layer_max_deg = 0;
        long long layer_total_deg = 0;
        int c = 0;
        for (int i = 0; i < n; ++i) {
            if (level[i] < l) continue;
            c++;
            layer_max_deg = max(layer_max_deg, neighbors[l][i].size());
            layer_total_deg += neighbors[l][i].size();
        }
        cout << "layer " << l << ": nodes " << c << "  max deg " << layer_max_deg << "  avg deg " << (double)layer_total_deg / c << "\n";
    }

    vector<vector<int>> results(nq);
    for (int ef : {10, 20, 50, 100, 200}) {
        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < nq; ++i)
            results[i] = query_hnsw(dataset, neighbors, dim, &query[(size_t)i * dim], entry, top, ef);
        auto end = chrono::high_resolution_clock::now();
        double hnsw_ms = chrono::duration<double, milli>(end - start).count() / nq;
        long long correct = 0;
        for (int i = 0; i < nq; ++i) correct += count_correct(results[i], match, i, k);
        cout << "HNSW  ef=" << ef << "  recall@" << k << ": " << (double)correct / ((double)nq * k)
             << "  avg per query: " << hnsw_ms << " ms";
        if (brute_ms > 0) cout << "  speedup: " << brute_ms / hnsw_ms << "x";
        cout << "\n";
    }
    return 0;
}