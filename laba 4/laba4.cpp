#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <omp.h>
#include <fstream>
const double PI = 3.14159265358979323846;

std::vector<std::complex<double>> calculate_w(size_t n) {
    std::vector<std::complex<double>> w(n);
    for (size_t k = 0; k < n; k++) {
        double angle = 2 * PI * k / n;
        w[k] = std::polar(1.0, angle);
    }
    return w;
}

void dft_generic(const std::vector<std::complex<double>>& input, std::vector<std::complex<double>>& output, const std::vector<std::complex<double>>& w) {
    size_t n = input.size();
    output.resize(n);
    for (size_t k = 0; k < n; k++) {
        output[k] = std::complex<double>(0.0, 0.0);
        for (size_t j = 0; j < n; j++) {
            output[k] += input[j] * w[j * k % n];
        }
    }
}

void dft(const std::vector<std::complex<double>>& time, std::vector<std::complex<double>>& spectrum, size_t n) {
    std::vector<std::complex<double>> w = calculate_w(n);
    dft_generic(time, spectrum, w);
}

void idft(const std::vector<std::complex<double>>& spectrum, std::vector<std::complex<double>>& time, size_t n) {
    std::vector<std::complex<double>> w = calculate_w(n);
    for (auto& val : w) {
        val = std::conj(val) / static_cast<double>(n);
    }
    dft_generic(spectrum, time, w);
}

void fft_generic(const std::complex<double>* in, std::complex<double>* out, size_t n, std::complex<double> w, size_t s) 
{
    if (n == 1) {
        out[0] = in[0];
        return;
    }

    const size_t half = n / 2;
    
    fft_generic(in,       out,       half, w * w, 2 * s);
    fft_generic(in + s,   out + half, half, w * w, 2 * s);
    
    std::complex<double> w_k = 1.0;
    
    for (size_t k = 0; k < half; ++k) {
        std::complex<double> even = out[k];
        std::complex<double> odd  = out[k + half] * w_k;
        
        out[k]       = even + odd;
        out[k + half] = even - odd;

        w_k *= w;
    }
}

void fft(std::complex<double>* data, size_t n) {
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("n must be power of 2");
    }
    
    std::complex<double>* temp = new std::complex<double>[n];

    
    std::complex<double> w = std::exp(std::complex<double>(0, -2.0 * PI / n));
    
    fft_generic(data, temp, n, w, 1);
    std::copy(temp, temp + n, data);
    delete[] temp;
}


void ifft(std::complex<double>* data, size_t n) {
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("n must be power of 2");
    }
    
    std::complex<double>* temp = new std::complex<double>[n];
    
    std::complex<double> w = std::exp(std::complex<double>(0, 2.0 * PI / n));
    
    fft_generic(data, temp, n, w, 1);
    
    for (size_t i = 0; i < n; ++i) {
        data[i] = temp[i] / static_cast<double>(n);
    }
    delete[] temp;
}

void fft_omp_recursive(const std::complex<double> *in,
                       std::complex<double> *out,
                       size_t n,
                       std::complex<double> w,
                       size_t s,
                       int depth = 0)
{
    if (n == 1) {
        out[0] = in[0];
        return;
    }

    const size_t half = n / 2;
    
    if (depth < 4 && n > 1024) {
        #pragma omp task shared(in, out)
        fft_omp_recursive(in, out, half, w * w, 2 * s, depth + 1);
        
        #pragma omp task shared(in, out)
        fft_omp_recursive(in + s, out + half, half, w * w, 2 * s, depth + 1);
        
        #pragma omp taskwait
    } else {
        fft_omp_recursive(in, out, half, w * w, 2 * s, depth + 1);
        fft_omp_recursive(in + s, out + half, half, w * w, 2 * s, depth + 1);
    }
    
    std::complex<double> w_k = 1.0;
    
    if (half > 512) {
        #pragma omp parallel for firstprivate(w_k)
        for (size_t k = 0; k < half; ++k) {
            w_k = std::pow(w, static_cast<double>(k));
            std::complex<double> even = out[k];
            std::complex<double> odd = out[k + half] * w_k;
            out[k] = even + odd;
            out[k + half] = even - odd;
        }
    } else {
        for (size_t k = 0; k < half; ++k) {
            std::complex<double> even = out[k];
            std::complex<double> odd = out[k + half] * w_k;
            out[k] = even + odd;
            out[k + half] = even - odd;
            w_k *= w;
        }
    }
}

void fft_parallel(std::complex<double>* data, size_t n) {
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("n must be power of 2");
    }
    
    std::complex<double>* temp = new std::complex<double>[n];
    std::complex<double> w = std::exp(std::complex<double>(0, -2.0 * PI / n));
    
    #pragma omp parallel
    {
        #pragma omp single
        fft_omp_recursive(data, temp, n, w, 1, 0);
    }
    
    std::copy(temp, temp + n, data);
    delete[] temp;
}

void ifft_parallel(std::complex<double>* data, size_t n) {
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("n must be power of 2");
    }
    
    std::complex<double>* temp = new std::complex<double>[n];
    std::complex<double> w = std::exp(std::complex<double>(0, 2.0 * PI / n));
    
    #pragma omp parallel
    {
        #pragma omp single
        fft_omp_recursive(data, temp, n, w, 1, 0);
    }

    for (size_t i = 0; i < n; ++i) {
        data[i] = temp[i] / static_cast<double>(n);
    }

    std::copy(temp, temp + n, data);
    delete[] temp;
}

void run_dft(const std::vector<std::complex<double>>& input, 
             std::vector<std::complex<double>>& output) {
    size_t n = input.size();
    const double pi = std::acos(-1.0);
    
    // Предвычисление поворотных множителей для DFT
    std::vector<std::complex<double>> w(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = std::exp(std::complex<double>(0, -2.0 * pi * i / n));
    }
    
    dft_generic(input, output, w);
}

void run_sequential(std::vector<std::complex<double>>& data) {
    size_t n = data.size();
    std::vector<std::complex<double>> temp(n);
    double pi = std::acos(-1.0);
    std::complex<double> w = std::exp(std::complex<double>(0, -2.0 * pi / n));
    fft_generic(data.data(), temp.data(), n, w, 1);
    data = temp;
}

void run_omp(std::vector<std::complex<double>>& data) {
    size_t n = data.size();
    std::vector<std::complex<double>> temp(n);
    double pi = std::acos(-1.0);
    std::complex<double> w = std::exp(std::complex<double>(0, -2.0 * pi / n));
    
    #pragma omp parallel
    {
        #pragma omp single
        fft_omp_recursive(data.data(), temp.data(), n, w, 1, 0);
    }
    data = temp;
}

double max_error(const std::vector<std::complex<double>>& a, 
                 const std::vector<std::complex<double>>& b) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}
int main() {
    const size_t N = 1 << 20;          // 2^20
    const int repetitions = 5;
    const int max_threads = omp_get_max_threads();

    std::cout << "FFT Benchmark (N = " << N
              << " = 2^" << static_cast<int>(std::log2(N)) << ")\n";
    std::cout << "Max available threads: " << max_threads << "\n\n";

    // ---------- генерация входных данных ----------
    std::vector<std::complex<double>> original(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        original[i] = {dist(rng), dist(rng)};
    }

    // ---------- baseline (sequential FFT) ----------
    std::vector<std::complex<double>> baseline(N);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        auto copy = original;
        run_sequential(copy);
        if (i == 0)
            baseline = copy;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double seq_time =
        std::chrono::duration<double, std::milli>(end - start).count()
        / repetitions;

    // ---------- CSV ----------
    std::ofstream csv("results.csv");
    csv << "T,time_ms,speedup,efficiency\n";
    csv << 1 << "," << seq_time << ",1.0,100\n";

    // ---------- таблица ----------
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::setw(10) << "Threads"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Efficiency\n";
    std::cout << std::string(70, '-') << "\n";

    std::cout << std::setw(10) << "1 (seq)"
              << std::setw(15) << std::fixed << std::setprecision(2) << seq_time
              << std::setw(15) << "1.00x"
              << std::setw(14) << "100%\n";

    // ---------- OMP прогоны ----------
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);

        // warm-up
        auto warmup = original;
        run_omp(warmup);

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            auto copy = original;
            run_omp(copy);
        }
        end = std::chrono::high_resolution_clock::now();

        double time =
            std::chrono::duration<double, std::milli>(end - start).count()
            / repetitions;

        // проверка корректности для 1 потока
        if (t == 1) {
            double err = max_error(baseline, warmup);
            std::cout << "[Check: max error vs sequential = "
                      << std::scientific << err << "]\n";
        }

        double speedup = seq_time / time;
        double efficiency = (speedup / t) * 100.0;

        std::cout << std::setw(10) << t
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << time
                  << std::setw(14) << speedup << "x"
                  << std::setw(14) << static_cast<int>(efficiency) << "%";

        if (t == max_threads)
            std::cout << "  <-- max";
        std::cout << "\n";

        // CSV
        csv << t << ","
            << time << ","
            << speedup << ","
            << efficiency << "\n";
    }

    std::cout << std::string(70, '-') << "\n";
    csv.close();

    std::cout << "CSV saved to results.csv\n";
    return 0;
}

