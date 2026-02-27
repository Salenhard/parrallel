#include <iostream>
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

void fft_seq(cd* Xp, std::size_t n, const cd* X);
void fft_omp(cd* Xp, std::size_t n, const cd* X);
void fft_async(cd* Xp, std::size_t n, const cd* X);

void fft_seq(cd* Xp, std::size_t n, const cd* X) {
    if (n < 2) {
        Xp[0] = X[0];
        return;
    }

    std::size_t half = n / 2;

    std::vector<cd> even(half), odd(half);
    for (std::size_t i = 0; i < half; ++i) {
        even[i] = X[2*i];
        odd[i]  = X[2*i + 1];
    }

    std::vector<cd> Xeven(half), Xodd(half);

    fft_seq(Xeven.data(), half, even.data());
    fft_seq(Xodd.data(),  half, odd.data());

    for (std::size_t i = 0; i < half; ++i) {
        Xp[i]        = Xeven[i];
        Xp[i + half] = Xodd[i];
    }

    cd w = std::exp(cd(0, -2.0 * PI / n));

    for (std::size_t k = 0; k < half; ++k) {
        cd wk = std::pow(w, k);
        cd a = Xp[k];
        cd b = Xp[k + half];

        Xp[k]        = a + wk * b;
        Xp[k + half] = a - wk * b;
    }
}

void fft_omp(cd* Xp, std::size_t n, const cd* X) {
    if (n < 2) {
        Xp[0] = X[0];
        return;
    }

    std::size_t half = n / 2;

    std::vector<cd> even(half), odd(half);
    for (std::size_t i = 0; i < half; ++i) {
        even[i] = X[2*i];
        odd[i]  = X[2*i + 1];
    }

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

    for (std::size_t i = 0; i < half; ++i) {
        Xp[i]        = Xeven[i];
        Xp[i + half] = Xodd[i];
    }

    cd w = std::exp(cd(0, -2.0 * PI / n));

    for (std::size_t k = 0; k < half; ++k) {
        cd wk = std::pow(w, k);
        cd a = Xp[k];
        cd b = Xp[k + half];

        Xp[k]        = a + wk * b;
        Xp[k + half] = a - wk * b;
    }
}

void fft_async(cd* Xp, std::size_t n, const cd* X) {
    if (n < 2) {
        Xp[0] = X[0];
        return;
    }

    std::size_t half = n / 2;

    std::vector<cd> even(half), odd(half);
    for (std::size_t i = 0; i < half; ++i) {
        even[i] = X[2*i];
        odd[i]  = X[2*i + 1];
    }

    std::vector<cd> Xeven(half), Xodd(half);

    if (n <= PARALLEL_THRESHOLD) {
        fft_seq(Xeven.data(), half, even.data());
        fft_seq(Xodd.data(),  half, odd.data());
    } else {
        auto f1 = std::async(std::launch::async, fft_async, Xeven.data(), half, even.data());
        auto f2 = std::async(std::launch::async, fft_async, Xodd.data(),  half, odd.data());
        f1.wait();
        f2.wait();
    }

    for (std::size_t i = 0; i < half; ++i) {
        Xp[i]        = Xeven[i];
        Xp[i + half] = Xodd[i];
    }

    cd w = std::exp(cd(0, -2.0 * PI / n));

    for (std::size_t k = 0; k < half; ++k) {
        cd wk = std::pow(w, k);
        cd a = Xp[k];
        cd b = Xp[k + half];

        Xp[k]        = a + wk * b;
        Xp[k + half] = a - wk * b;
    }
}

void ifft_seq(cd* output, std::size_t n, const cd* input) {
    std::vector<cd> tmp(n);
    for (std::size_t i = 0; i < n; ++i)
        tmp[i] = std::conj(input[i]);

    fft_seq(output, n, tmp.data());

    for (std::size_t i = 0; i < n; ++i)
        output[i] = std::conj(output[i]) / static_cast<double>(n);
}

void ifft_omp(cd* output, std::size_t n, const cd* input) {
    std::vector<cd> tmp(n);
    for (std::size_t i = 0; i < n; ++i)
        tmp[i] = std::conj(input[i]);

    #pragma omp parallel
    {
        #pragma omp single
        fft_omp(output, n, tmp.data());
    }

    for (std::size_t i = 0; i < n; ++i)
        output[i] = std::conj(output[i]) / static_cast<double>(n);
}

void ifft_async(cd* output, std::size_t n, const cd* input) {
    std::vector<cd> tmp(n);
    for (std::size_t i = 0; i < n; ++i)
        tmp[i] = std::conj(input[i]);

    fft_async(output, n, tmp.data());

    for (std::size_t i = 0; i < n; ++i)
        output[i] = std::conj(output[i]) / static_cast<double>(n);
}

bool check_correctness(const std::vector<cd>& input) {
    std::size_t n = input.size();
    std::vector<cd> tmp(n), res(n);

    fft_seq(tmp.data(), n, input.data());
    ifft_seq(res.data(), n, tmp.data());

    double tol = 1e-10;

    for (std::size_t i = 0; i < n; ++i)
        if (std::abs(res[i] - input[i]) > tol)
            return false;

    return true;
}

int main() {
    std::size_t n = 1024;
    std::vector<cd> data(n);

    for (std::size_t i = 0; i < n / 2; ++i) {
        cd value(
            std::sin(2.0 * PI * i / n),
            std::cos(2.0 * PI * i / n)
        );
        data[n/2 - i - 1] = value;
        data[n/2 + i]     = value;
    }

    std::vector<cd> out(n);

    auto t1 = std::chrono::steady_clock::now();
    fft_seq(out.data(), n, data.data());
    auto t2 = std::chrono::steady_clock::now();
    double time_seq = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "fft_seq: " << time_seq << "\n";

    int max_threads = omp_get_max_threads();

    std::ofstream csv("results.csv");
    csv << "method,t,time,acceleration\n";

    for (int t = 1; t <= max_threads; ++t) {

        omp_set_num_threads(t);

        #pragma omp parallel
        {
            #pragma omp single
            {
                t1 = std::chrono::steady_clock::now();
                fft_omp(out.data(), n, data.data());
                t2 = std::chrono::steady_clock::now();
            }
        }

        double time_parallel =
            std::chrono::duration<double>(t2 - t1).count();

        double acceleration = time_seq / time_parallel;

        std::cout << "OMP threads: " << t
                  << " time: " << time_parallel
                  << " speedup: " << acceleration << "\n";

        csv << "omp,"
            << t << ","
            << time_parallel << ","
            << acceleration << "\n";
    }

    for (int t = 1; t <= max_threads; ++t) {

        omp_set_num_threads(t);

        t1 = std::chrono::steady_clock::now();
        fft_async(out.data(), n, data.data());
        t2 = std::chrono::steady_clock::now();

        double time_async =
            std::chrono::duration<double>(t2 - t1).count();

        double acceleration = time_seq / time_async;

        std::cout << "ASYNC threads: " << t
                  << " time: " << time_async
                  << " speedup: " << acceleration << "\n";

        csv << "async,"
            << t << ","
            << time_async << ","
            << acceleration << "\n";
    }

    csv.close();

    std::cout << "correct: "
              << (check_correctness(data) ? "true" : "false")
              << "\n";

    return 0;
}
