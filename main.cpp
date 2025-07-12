#include <iostream>

typedef float float32;
typedef double float64;
#if HAVE_LONGDOUBLE
typedef long double float128;
#endif

namespace __main__ {
    float64 square(float64);

    float64 mult(float64, float64);
}

int main() {
    std::cout << "square: " << __main__::square(8) << std::endl;
    std::cout << "mult: " << __main__::mult(8.0, 8.0) << std::endl;
}
