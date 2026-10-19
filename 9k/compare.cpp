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

int brute_top1(const vector<float>& dataset, int n, int dim, const float* q) {
    float best = numeric_limits<float>::max();
    int best_id = -1;
    for (int i = 0; i < n; ++i) {
        float dist = squared_distance(&dataset[(size_t)i * dim], q, dim);
        if (dist < best) {
            best = dist;
            best_id = i;
        }
    }
    return best_id;
}

double probe_ms(const vector<float>& dataset, const vector<float>& query, int n, int dim, int probe_queries) {
    volatile int sink = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < probe_queries; ++i) sink = brute_top1(dataset, n, dim, &query[(size_t)i * dim]);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count() / probe_queries;
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

vector<vector<int>> build_graph_insert(const vector<float>& dataset, int n, int dim, int M, int ef_construction, int Mmax) {
    vector<vector<int>> neighbors(n);
    for (int p = 1; p < n; ++p) {
        auto res = search_layer(dataset, neighbors, dim, &dataset[(size_t)p * dim], 0, ef_construction);
        for (int j = 0; j < M && j < (int)res.size(); ++j) {
            neighbors[p].push_back(res[j]);
            neighbors[res[j]].push_back(p);
            if ((int)neighbors[res[j]].size() > Mmax) shrink_neighbors(neighbors[res[j]], dataset, dim, res[j], Mmax);
        }
    }
    return neighbors;
}

int draw_level(mt19937& rng, int M) {
    static uniform_real_distribution<double> dist(0.0, 1.0);
    double mL = 1.0 / log((double)M);
    double u = 1.0 - dist(rng);
    return (int)(-log(u) * mL);
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

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(0);
    int dim = 128, n = 9000, nq = 1000, gt_k = 10, M = 16;
    int num_sweeps = argc > 1 ? atoi(argv[1]) : 10;
    int probe_queries = 100;
    const double probe_ref = 2.097, probe_limit = 1.08;
    vector<float> dataset, query;
    vector<int> match;
    try {
        dataset = load_fbin("dataset.fbin", (size_t)n * dim);
        query = load_fbin("query.fbin", (size_t)nq * dim);
        match = load_ibin("match.ibin", (size_t)nq * gt_k);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    cout << "probe pre-build: " << probe_ms(dataset, query, n, dim, probe_queries) << " ms/query\n";

    mt19937 rng(42);
    vector<int> level(n);
    int max_level = 0;
    for (int i = 0; i < n; ++i) {
        level[i] = draw_level(rng, M);
        max_level = max(max_level, level[i]);
    }
    vector<vector<vector<int>>> neighbors(max_level + 1, vector<vector<int>>(n));
    int entry, top;
    auto t0 = chrono::high_resolution_clock::now();
    build_hnsw(dataset, n, dim, M, 100, level, neighbors, entry, top);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "HNSW build: " << chrono::duration<double, milli>(t1 - t0).count() << " ms\n";
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

    t0 = chrono::high_resolution_clock::now();
    auto flat_neighbors = build_graph_insert(dataset, n, dim, M, 100, 32);
    t1 = chrono::high_resolution_clock::now();
    size_t max_deg = 0;
    long long total_deg = 0;
    for (auto& v : flat_neighbors) { max_deg = max(max_deg, v.size()); total_deg += v.size(); }
    cout << "Flat build: " << chrono::duration<double, milli>(t1 - t0).count() << " ms  max degree: " << max_deg << "  avg degree: " << (double)total_deg / n << "\n";

    vector<int> efs = {10, 20, 50, 100, 200};
    double before = probe_ms(dataset, query, n, dim, probe_queries);
    cout << "probe start: " << before << " ms/query\n";

    for (int s = 0; s < num_sweeps; ++s) {
        vector<double> flat_ms(efs.size()), hnsw_ms(efs.size());
        vector<int> flat_correct(efs.size()), hnsw_correct(efs.size());
        for (size_t e = 0; e < efs.size(); ++e) {
            for (int pass = 0; pass < 2; ++pass) {
                bool run_flat = ((pass == 0) == (s % 2 == 0));
                vector<int> found(nq, -1);
                auto start = chrono::high_resolution_clock::now();
                for (int i = 0; i < nq; ++i) {
                    const float* q = &query[(size_t)i * dim];
                    auto res = run_flat ? search_layer(dataset, flat_neighbors, dim, q, 0, efs[e])
                                        : query_hnsw(dataset, neighbors, dim, q, entry, top, efs[e]);
                    if (!res.empty()) found[i] = res[0];
                }
                auto end = chrono::high_resolution_clock::now();
                double ms = chrono::duration<double, milli>(end - start).count() / nq;
                int correct = 0;
                for (int i = 0; i < nq; ++i)
                    if (found[i] == match[(size_t)i * gt_k]) correct++;
                if (run_flat) { flat_ms[e] = ms; flat_correct[e] = correct; }
                else { hnsw_ms[e] = ms; hnsw_correct[e] = correct; }
            }
        }
        double after = probe_ms(dataset, query, n, dim, probe_queries);
        bool ok = before <= probe_ref * probe_limit && after <= probe_ref * probe_limit;
        cout << "sweep " << s << "  probe_before " << before << "  probe_after " << after
             << "  status " << (ok ? "OK" : "SLOW") << "\n";
        for (size_t e = 0; e < efs.size(); ++e) {
            cout << "  ef=" << efs[e]
                 << "  flat: recall@1 " << (double)flat_correct[e] / nq << "  " << flat_ms[e] << " ms"
                 << "   hnsw: recall@1 " << (double)hnsw_correct[e] / nq << "  " << hnsw_ms[e] << " ms"
                 << "   hnsw vs flat " << (hnsw_ms[e] - flat_ms[e]) / flat_ms[e] * 100.0 << " %\n";
        }
        before = after;
    }
    return 0;
}