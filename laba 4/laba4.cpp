#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <cmath>
#include <future>
#include <chrono>
#include <limits>
#include <omp.h>
#include <fstream>

using cd = std::complex<double>;
const double PI = std::acos(-1.0);
const std::size_t PARALLEL_THRESHOLD = 64;

void fft_seq (cd* Xp, std::size_t n, const cd* X);
void fft_omp (cd* Xp, std::size_t n, const cd* X);
void fft_async(cd* Xp, std::size_t n, const cd* X);
void ifft_seq (cd* out, std::size_t n, const cd* in);
void ifft_omp (cd* out, std::size_t n, const cd* in);
void ifft_async(cd* out, std::size_t n, const cd* in);

void fft_seq(cd* Xp, std::size_t n, const cd* X) {
    if (n < 2) { Xp[0] = X[0]; return; }
    std::size_t half = n / 2;
    std::vector<cd> even(half), odd(half);
    for (std::size_t i = 0; i < half; ++i) { even[i] = X[2*i]; odd[i] = X[2*i+1]; }
    std::vector<cd> Xeven(half), Xodd(half);
    fft_seq(Xeven.data(), half, even.data());
    fft_seq(Xodd.data(),  half, odd.data());
    for (std::size_t i = 0; i < half; ++i) { Xp[i] = Xeven[i]; Xp[i+half] = Xodd[i]; }
    cd w = std::exp(cd(0, -2.0 * PI / n));
    for (std::size_t k = 0; k < half; ++k) {
        cd wk = std::pow(w, k);
        cd a = Xp[k], b = Xp[k+half];
        Xp[k]        = a + wk * b;
        Xp[k + half] = a - wk * b;
    }
}

void ifft_seq(cd* output, std::size_t n, const cd* input) {
    std::vector<cd> tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::conj(input[i]);
    fft_seq(output, n, tmp.data());
    for (std::size_t i = 0; i < n; ++i)
        output[i] = std::conj(output[i]) / static_cast<double>(n);
}

void fft_omp(cd* Xp, std::size_t n, const cd* X) {
    if (n < 2) { Xp[0] = X[0]; return; }
    std::size_t half = n / 2;
    std::vector<cd> even(half), odd(half);
    for (std::size_t i = 0; i < half; ++i) { even[i] = X[2*i]; odd[i] = X[2*i+1]; }
    std::vector<cd> Xeven(half), Xodd(half);
    if (n <= PARALLEL_THRESHOLD) {
        fft_seq(Xeven.data(), half, even.data());
        fft_seq(Xodd.data(),  half, odd.data());
    } else {
        #pragma omp task shared(Xeven)
        fft_omp(Xeven.data(), half, even.data());
        #pragma omp task shared(Xodd)
        fft_omp(Xodd.data(), half, odd.data());
        #pragma omp taskwait
    }
    for (std::size_t i = 0; i < half; ++i) { Xp[i] = Xeven[i]; Xp[i+half] = Xodd[i]; }
    cd w = std::exp(cd(0, -2.0 * PI / n));
    for (std::size_t k = 0; k < half; ++k) {
        cd wk = std::pow(w, k);
        cd a = Xp[k], b = Xp[k+half];
        Xp[k]        = a + wk * b;
        Xp[k + half] = a - wk * b;
    }
}

void ifft_omp(cd* output, std::size_t n, const cd* input) {
    std::vector<cd> tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::conj(input[i]);
    #pragma omp parallel
    {
        #pragma omp single
        fft_omp(output, n, tmp.data());
    }
    for (std::size_t i = 0; i < n; ++i)
        output[i] = std::conj(output[i]) / static_cast<double>(n);
}

void fft_async(cd* Xp, std::size_t n, const cd* X) {
    if (n < 2) { Xp[0] = X[0]; return; }
    std::size_t half = n / 2;
    std::vector<cd> even(half), odd(half);
    for (std::size_t i = 0; i < half; ++i) { even[i] = X[2*i]; odd[i] = X[2*i+1]; }
    std::vector<cd> Xeven(half), Xodd(half);
    if (n <= PARALLEL_THRESHOLD) {
        fft_seq(Xeven.data(), half, even.data());
        fft_seq(Xodd.data(),  half, odd.data());
    } else {
        auto f1 = std::async(std::launch::async, fft_async, Xeven.data(), half, even.data());
        auto f2 = std::async(std::launch::async, fft_async, Xodd.data(),  half, odd.data());
        f1.wait(); f2.wait();
    }
    for (std::size_t i = 0; i < half; ++i) { Xp[i] = Xeven[i]; Xp[i+half] = Xodd[i]; }
    cd w = std::exp(cd(0, -2.0 * PI / n));
    for (std::size_t k = 0; k < half; ++k) {
        cd wk = std::pow(w, k);
        cd a = Xp[k], b = Xp[k+half];
        Xp[k]        = a + wk * b;
        Xp[k + half] = a - wk * b;
    }
}

void ifft_async(cd* output, std::size_t n, const cd* input) {
    std::vector<cd> tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = std::conj(input[i]);
    fft_async(output, n, tmp.data());
    for (std::size_t i = 0; i < n; ++i)
        output[i] = std::conj(output[i]) / static_cast<double>(n);
}

bool ulp_close(cd a, cd b) {
    constexpr double eps    = std::numeric_limits<double>::epsilon();
    constexpr double factor = static_cast<double>(1LL << 32);
    auto ok = [&](double x, double y) {
        double ulp = std::abs(std::nextafter(x, std::numeric_limits<double>::infinity()) - x);
        double tol = std::max(factor * ulp, factor * eps);
        return std::abs(x - y) <= tol;
    };
    return ok(a.real(), b.real()) && ok(a.imag(), b.imag());
}

bool check_correctness(const std::vector<cd>& input) {
    std::size_t n = input.size();
    std::vector<cd> spectrum(n), recovered(n);
    fft_seq(spectrum.data(),   n, input.data());
    ifft_seq(recovered.data(), n, spectrum.data());
    for (std::size_t i = 0; i < n; ++i)
        if (!ulp_close(recovered[i], input[i])) return false;
    return true;
}

using Clock = std::chrono::steady_clock;

double measure_seq(const std::vector<cd>& data) {
    std::size_t n = data.size();
    std::vector<cd> out(n);
    auto t1 = Clock::now();
    fft_seq(out.data(), n, data.data());
    auto t2 = Clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

double measure_omp(const std::vector<cd>& data, int threads) {
    std::size_t n = data.size();
    std::vector<cd> out(n);
    omp_set_num_threads(threads);
    double t = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            auto t1 = Clock::now();
            fft_omp(out.data(), n, data.data());
            auto t2 = Clock::now();
            t = std::chrono::duration<double>(t2 - t1).count();
        }
    }
    return t;
}

double measure_async(const std::vector<cd>& data) {
    std::size_t n = data.size();
    std::vector<cd> out(n);
    auto t1 = Clock::now();
    fft_async(out.data(), n, data.data());
    auto t2 = Clock::now();
    return std::chrono::duration<double>(t2 - t1).count();
}

int main() {
    constexpr std::size_t N = 1024;
    std::vector<cd> data(N);
    for (std::size_t i = 0; i < N / 2; ++i) {
        cd v(std::sin(2.0 * PI * i / N), std::cos(2.0 * PI * i / N));
        data[N/2 - i - 1] = v;
        data[N/2 + i]     = v;
    }

    const int W_METHOD  = 10;
    const int W_THREADS =  8;
    const int W_TIME    = 14;
    const int W_SPEEDUP = 10;

    double time_seq = measure_seq(data);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << " FFT benchmark  (n=" << N << ")\n";

    std::cout << std::left
              << std::setw(W_METHOD)  << "method"
              << std::setw(W_THREADS) << "threads"
              << std::setw(W_TIME)    << "time (s)"
              << std::setw(W_SPEEDUP) << "speedup"
              << "\n"
              << std::string(W_METHOD + W_THREADS + W_TIME + W_SPEEDUP, '-')
              << "\n";

    std::cout << std::left
              << std::setw(W_METHOD)  << "seq"
              << std::setw(W_THREADS) << "1"
              << std::setw(W_TIME)    << time_seq
              << std::setw(W_SPEEDUP) << "1.000000"
              << "\n";

    int max_threads = omp_get_max_threads();

    std::vector<std::tuple<std::string,int,double,double>> results;
    results.emplace_back("seq", 1, time_seq, 1.0);

    for (int t = 1; t <= max_threads; ++t) {
        double time_omp = measure_omp(data, t);
        double speedup  = time_seq / time_omp;
        results.emplace_back("omp", t, time_omp, speedup);
        std::cout << std::left
                  << std::setw(W_METHOD)  << "omp"
                  << std::setw(W_THREADS) << t
                  << std::setw(W_TIME)    << time_omp
                  << std::setw(W_SPEEDUP) << speedup
                  << "\n";
    }

    double time_async = measure_async(data);
    double speedup_async = time_seq / time_async;
    results.emplace_back("async", 0, time_async, speedup_async);

    std::cout << std::left
              << std::setw(W_METHOD)  << "async"
              << std::setw(W_THREADS) << "auto"
              << std::setw(W_TIME)    << time_async
              << std::setw(W_SPEEDUP) << speedup_async
              << "\n";

    bool correct = check_correctness(data);
    std::cout << "\ncorrectness (2^32 ULP tolerance): "
              << (correct ? "PASS" : "FAIL") << "\n";

    std::ofstream csv("results.csv");
    csv << "method,threads,time_s,speedup\n";
    for (auto& [method, threads, time, speedup] : results)
        csv << method << "," << threads << ","
            << std::fixed << std::setprecision(9)
            << time << "," << speedup << "\n";
    csv << "async,auto,"
        << std::fixed << std::setprecision(9)
        << time_async << "," << speedup_async << "\n";
    csv.close();

    std::cout << "\nresults saved to results.csv\n";
    return 0;
}
