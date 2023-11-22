#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int N;
int SIZE;
// в методичке просят для STEPS=20  'Следует выполнить около 20 шагов по времени'
int K = 20;
double T = 0.01;
double L = 1.0;
int STEPS = 20;
double Lx = 1.0;
double Ly = 1.0;
double Lz = 1.0; 
double Hx = 0.0;
double Hy = 0.0;
double Hz = 0.0;
double Tau = 0.0;
double A2 = 0.0;
double* u_prev;
double* u_current;
double* u_next;

// добавляем вершину справа для оси где период условие.  (итого n+2 вершин для одной оси)
int get_index(int i, int j, int k)
{
    return i * (N + 2) * (N + 2) + j * (N + 2) + k;
}

void createJson(const double *u, double t, const char *path) {
        
        std::ofstream fout(path);
        fout << "{" << '\n';
        fout << "\t\"u\":" << '\n';
        fout << "\t[" << '\n';
        for (int i = 0; i < SIZE; i++) {
            fout << "\t\t" << u[i];
            if (i != SIZE - 1)
                fout << ", " << '\n';
            else
                fout << '\n';
        }
        fout << "\t]" << '\n';
        fout << "}" << '\n';

        fout.close();
}

double u_analytical(double x, double y, double z, double t)
{
    double ax = 2 * M_PI / Lx;
    double ay = 4 * M_PI / Ly;
    double az = 6 * M_PI / Lz;

    double res = sin(ax * x) * sin(ay * y) * sin(az * z) * cos(M_PI * sqrt( (4/(Lx*Lx)) + (16 / (Ly * Ly)) + (36 / (Lz * Lz))) * t);

    return res;
}

double phi(double x, double y, double z)
{
    return u_analytical(x, y, z, 0);
}

double delta_h(int i, int j, int k, double* matrix)
{
    double delta_x = (matrix[get_index(i - 1, j, k)] - 2 * matrix[get_index(i, j, k)] + matrix[get_index(i + 1, j, k)]) / (Hx * Hx);
    double delta_y = (matrix[get_index(i, j - 1, k)] - 2 * matrix[get_index(i, j, k)] + matrix[get_index(i, j + 1, k)]) / (Hy * Hy);
    double delta_z = (matrix[get_index(i, j, k - 1)] - 2 * matrix[get_index(i, j, k)] + matrix[get_index(i, j, k + 1)]) / (Hz * Hz);

    return delta_x + delta_y + delta_z;
}



int main(int argc, char** argv)
{
    // для ввода параметров из терминал например ./prac2 128 20 0.01 pi 
    for (size_t i = 0; i < argc; i++)
    {
        // std::cout << argv[i] << "\n"; 
    }
    // чтобы получить значение количества потоков
    int nthreads = 1; 
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
        std::cout << "number of threads" << nthreads << "\n";
    }
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    STEPS = K;
    T = atof(argv[3]);
    std::string L_format = argv[4]; // pi - M_PI, 1 - 1.0
    if (L_format == "pi" )
        L = M_PI;
    else
        L = 1;
    Lx = L;
    Ly = L;
    Lz = L;

    Hx = Lx / (N);
    Hy = Ly / (N);
    Hz = Lz / (N);
    Tau = T / K;
    A2 = 1;

    // добавляем вершину справа для оси где период условие.
    SIZE = (N + 2) * (N + 2) * (N + 2);
    u_prev = new double[SIZE];
    u_current = new double[SIZE];
    u_next = new double[SIZE];

    double start_time, end_time;
    
    // std::cout << "запускаем таймер\n";  
    start_time = omp_get_wtime();


    // std::cout<< N << " " << K << " " << T << " " << L << " " << Lx << " " << Ly << " " << Lz << "\n"; 

    #pragma omp parallel for collapse(3)
    for (int i = 1; i <= N; i++) 
        for (int j = 1; j <= N; j++)
            for (int k = 1; k <= N; k++)
                u_prev[get_index(i, j, k)] = phi(i * Hx, j * Hy, k * Hz);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j <= N; j++) {
            u_prev[get_index(0, i, j)] = u_prev[get_index(N, i, j)];
            u_prev[get_index(N + 1, i, j)] = u_prev[get_index(1, i, j)];

            u_prev[get_index(i, 0, j)] = u_prev[get_index(i, N, j)];
            u_prev[get_index(i, N + 1, j)] = u_prev[get_index(i, 1, j)];

            u_prev[get_index(i, j, 0)] = u_prev[get_index(i, j, N )];
            u_prev[get_index(i, j, N + 1)] = u_prev[get_index(i, j, 1)];
        }
    }   


    #pragma omp parallel for collapse(3)
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            for (int k = 1; k <= N; k++)
                u_current[get_index(i, j, k)] = u_prev[get_index(i, j, k)] + Tau * Tau * A2 / 2 * delta_h(i, j, k, u_prev);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j <= N; j++) {
            u_current[get_index(0, i, j)] = u_current[get_index(N, i, j)]; 
            u_current[get_index(N + 1, i, j)] = u_current[get_index(1, i, j)];

            u_current[get_index(i, 0, j)] = u_current[get_index(i, N, j)];
            u_current[get_index(i, N + 1, j)] = u_current[get_index(i, 1, j)];

            u_current[get_index(i, j, 0)] = u_current[get_index(i, j, N )];
            u_current[get_index(i, j, N + 1)] = u_current[get_index(i, j, 1)];
        }
    }


    for (int step = 2; step <= STEPS; step++) {


        #pragma omp parallel for collapse(3)
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++)
                    for (int k = 1; k <= N; k++)
                        u_next[get_index(i, j, k)] = 2 * u_current[get_index(i, j, k)] - u_prev[get_index(i, j, k)] + Tau * Tau * A2 * delta_h(i, j, k, u_current);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= N; i++) {
            for (int j = 0; j <= N; j++) {
                u_next[get_index(0, i, j)] = u_next[get_index(N, i, j)]; 
                u_next[get_index(N + 1, i, j)] = u_next[get_index(1, i, j)];

                u_next[get_index(i, 0, j)] = u_next[get_index(i, N, j)];
                u_next[get_index(i, N + 1, j)] = u_next[get_index(i, 1, j)];

                u_next[get_index(i, j, 0)] = u_next[get_index(i, j, N)];
                u_next[get_index(i, j, N + 1)] = u_next[get_index(i, j, 1)];
            }
        }

        double* tmp = u_prev;
        u_prev = u_current;
        u_current = u_next;
        u_next = tmp;
        // при выходе из цикла u_20 (последнее) хранится в u_current, u_19 (пред последнее) в u_prev
    }
  
    end_time = omp_get_wtime();


    double error = 0;
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= N; j++)
            for (int k = 0; k <= N; k++) {
                    error = std::max(error, std::fabs(u_current[get_index(i, j, k)] - u_analytical(i * Hx, j * Hy, k * Hz, STEPS * Tau)));
            }



    std::string calculated_filename = "calculated_for_" + std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(T) + "_" + L_format + ".json";
    createJson(u_current, STEPS * Tau, calculated_filename.c_str());

   
    // сохранения всех значений матрицы u_аналитической (u_20) – 
    // Важно – значения u_analytical на шаге u_20 будем записывать в матрицу u_next, чтобы не испортить значения хранящиеся в u_current (u_20)
    #pragma omp parallel for collapse(3)
        for (int i = 0; i <= N; i++)
            for (int j = 0; j <= N; j++)
                for (int k = 0; k <= N; k++)
                    u_next[get_index(i, j, k)] = u_analytical(i * Hx, j * Hy, k * Hz, STEPS * Tau);

    std::string analytical_filename = "analytical_for_" + std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(T) + "_" + L_format + ".json";
    createJson(u_next, STEPS * Tau, analytical_filename.c_str());


    // сохранения погрешностей для всех значений разности двух матриц u_посчитанной и u_аналитической (u_20)
    #pragma omp parallel for collapse(3)
        for (int i = 0; i <= N; i++)
            for (int j = 0; j <= N; j++)
                for (int k = 0; k <= N; k++)               
                    u_prev[get_index(i, j, k)] = std::fabs(u_current[get_index(i, j, k)] - u_next[get_index(i, j, k)]);
    std::string error_filename = "errors_for_" + std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(T) + "_" + L_format + ".json";
    createJson(u_prev, STEPS * Tau, error_filename.c_str());



    std::string filename = "stats_for_" + std::to_string(N) + "_" + std::to_string(K) + "_" + std::to_string(T) + "_" + std::to_string(nthreads) + "_" + L_format + ".txt";
    std::ofstream fout(filename, std::ios_base::app);
    fout << N << " " << K << " " << T << " " << nthreads << " " << error << " " << end_time - start_time << std::endl;
    fout.close();


    free(u_prev);
    free(u_current);
    free(u_next);

    return 0;
}