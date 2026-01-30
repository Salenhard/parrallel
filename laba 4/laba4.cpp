#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

const double PI = 3.14159265358979323846;

std::vector<std::complex<double>> calculate_w(size_t n) {
    std::vector<std::complex<double>> w(n);
    for (size_t k = 0; k < n; k++) {
        double angle = 2 * PI * k / n;
        w[k] = std::polar(1.0, angle);
    }
    return w;
}

// Generic DFT function
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
    for (auto& val : w) 
        val = std::conj(val) / static_cast<double>(n);
    
    dft_generic(spectrum, time, w);
}

void fft_generic(std::vector<std::complex<double>>& a,
                 const std::vector<std::complex<double>>& w)
{
    const std::size_t n = a.size();
    if (n <= 2) {                      
        std::vector<std::complex<double>> tmp;
        dft_generic(a, tmp, w);         
        a.swap(tmp);
        return;
    }

    const std::size_t m = n / 2;
    std::vector<std::complex<double>> even(m), odd(m);

    for (std::size_t i = 0; i < m; ++i) {
        even[i] = a[2 * i];
        odd [i] = a[2 * i + 1];
    }

    /* рекурсивные вызовы — каждый со своим w */
    fft_generic(even, calculate_w(m));
    fft_generic(odd , calculate_w(m));

    /* объединяем по схеме Cooley–Tukey */
    for (std::size_t k = 0; k < m; ++k) {
        const std::complex<double> t = w[k * (n / n)] * odd[k]; // w[k] для N-точки
        a[k]     = even[k] + t;
        a[k + m] = even[k] - t;
    }
}

std::vector<std::complex<double>> fft(const std::vector<double>& x)
{
    const std::size_t n = x.size();               
    std::vector<std::complex<double>> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = x[i];

    fft_generic(a, calculate_w(n));
    return a;                                     
}

std::vector<double> ifft(const std::vector<std::complex<double>>& spectrum)
{
    const std::size_t n = spectrum.size();
    std::vector<std::complex<double>> a = spectrum;

    auto w = calculate_w(n);
    for (auto& v : w) v = std::conj(v);

    fft_generic(a, w);                 

    std::vector<double> out(n);
    const double scale = 1.0 / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (a[i] * scale).real();

    return out;
}

int main() {
    std::vector<double> x = {1, 2, 3, 4};
    size_t n = x.size();
    std::vector<std::complex<double>> X(n), time(n), x_reconstructed(n);

    for (size_t i = 0; i < n; ++i)
        time[i] = std::complex<double>(x[i], 0.0);
    

    std::vector<std::complex<double>> out1 = fft(x);
    std::vector<double> out2 = ifft(X);

    std::cout << "Original: ";
    for (double val : x) 
        std::cout << val << " ";
    
    std::cout << "\nFFT: ";
    for (const auto& val : out1) 
        std::cout << val << " ";
    
    std::cout << "\nReconstructed: ";
    for (const auto& val : out2) 
        std::cout << std::real(val) << " ";
    
    std::cout << std::endl;

    return 0;
}