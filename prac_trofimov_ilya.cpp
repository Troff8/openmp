#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include <cstddef> 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int N;
int SIZE;
// 'Следует выполнить около 20 шагов по времени'
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

// mpi params

int RESULT_SIZES_TAG = 1;
int RESULT_TAG = 2;

int GridSizeX = 1;
int GridSizeY = 1;
int GridSizeZ = 1;
int BlockIdxX = 1;
int BlockIdxY = 1; 
int BlockIdxZ = 1;
int BlockSizeX = 1;
int BlockSizeY = 1;
int BlockSizeZ = 1;
int Rank = 0;
int NumProc = 1;

int buf_x_size = 0;
int buf_y_size = 0;
int buf_z_size = 0;

typedef struct {
    int x, y, z;
} GridSize;


struct buffers_sms {
    double* send_x_0;
    double* send_x_n;
    double* send_y_0;
    double* send_y_n;
    double* send_z_0;
    double* send_z_n;
    double* recv_x_0;
    double* recv_x_n;
    double* recv_y_0;
    double* recv_y_n;
    double* recv_z_0;
    double* recv_z_n;
};

int id_to_rank(int x, int y, int z)
{
    return z * (GridSizeX * GridSizeY) + y * GridSizeX + x;
}
int get_index(int i, int j, int k)
{
    return i * BlockSizeY * BlockSizeZ + j * BlockSizeZ + k;
}

void init_bufs(buffers_sms* bufs)
{
    buf_x_size = (BlockSizeY - 2) * (BlockSizeZ - 2);
    buf_y_size = (BlockSizeX - 2) * (BlockSizeZ - 2);
    buf_z_size = (BlockSizeX - 2) * (BlockSizeY - 2);

    if (GridSizeX != 1) {
        bufs->send_x_0 = new double[buf_x_size];
        bufs->send_x_n = new double[buf_x_size];
        bufs->recv_x_0 = new double[buf_x_size];
        bufs->recv_x_n = new double[buf_x_size];
    } else {
        bufs->send_x_0 = nullptr;
        bufs->send_x_n = nullptr;
        bufs->recv_x_0 = nullptr;
        bufs->recv_x_n = nullptr;
    }

    if (GridSizeY != 1) {
        bufs->send_y_0 = new double[buf_y_size];
        bufs->send_y_n = new double[buf_y_size];
        bufs->recv_y_0 = new double[buf_y_size];
        bufs->recv_y_n = new double[buf_y_size];
    } else {
        bufs->send_y_0 = nullptr;
        bufs->send_y_n = nullptr;
        bufs->recv_y_0 = nullptr;
        bufs->recv_y_n = nullptr;
    }

    if (GridSizeZ != 1) {
        bufs->send_z_0 = new double[buf_z_size];
        bufs->send_z_n = new double[buf_z_size];
        bufs->recv_z_0 = new double[buf_z_size];
        bufs->recv_z_n = new double[buf_z_size];
    } else {
        bufs->send_z_0 = nullptr;
        bufs->send_z_n = nullptr;
        bufs->recv_z_0 = nullptr;
        bufs->recv_z_n = nullptr;
    }


}

void wait_all_requests(double* matrix, struct buffers_sms* bufs,MPI_Request* request_x,MPI_Request* request_y,MPI_Request* request_z){
    if (request_x != NULL) {
        // std::cout << "rec x"<< Rank << "\n";
        //  fflush(stdout);
        MPI_Status* statuses = new MPI_Status[4];
        MPI_Waitall(4, request_x, statuses);
        
        delete[] statuses;
        delete[] request_x;

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; ++j)
        {
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(0, j, k)] = bufs->recv_x_0[(j - 1) * (BlockSizeZ - 2) + (k - 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; ++j){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(BlockSizeX - 1, j, k)] = bufs->recv_x_n[(j - 1) * (BlockSizeZ - 2) + (k - 1)];
            }
        }
    }

    if (request_y != NULL) {
        MPI_Status* statuses = new MPI_Status[4];
        MPI_Waitall(4, request_y, statuses);
        delete[] statuses;
        delete[] request_y;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; i++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                matrix[get_index(i, 0, k)] = bufs->recv_y_0[(i - 1) * (BlockSizeZ - 2) + (k - 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; i++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                matrix[get_index(i, BlockSizeY - 1, k)] = bufs->recv_y_n[(i - 1) * (BlockSizeZ - 2) + (k - 1)];
            }
        }
    }

    if (request_z != NULL) {
        MPI_Status* statuses = new MPI_Status[4];
        MPI_Waitall(4, request_z, statuses);
        delete[] statuses;
        delete[] request_z;

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; j++){
            for (int k = 1; k < BlockSizeY - 1; k++){
                matrix[get_index(j, k, 0)] = bufs->recv_z_0[(j - 1) * (BlockSizeY - 2) + (k - 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; j++){
            for (int k = 1; k < BlockSizeY - 1; k++){
                matrix[get_index(j, k, BlockSizeZ - 1)] = bufs->recv_z_n[(j - 1) * (BlockSizeY - 2) + (k - 1)];
            }
        }
    }
}
MPI_Request*  update_halo_os_x(double* matrix, struct buffers_sms* bufs,MPI_Request* request){
    if (GridSizeX != 1) {
        //  std::cout << "update x"<< Rank << "\n";
        //  fflush(stdout);
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; j++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                bufs->send_x_0[(j - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(1, j, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; j++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                bufs->send_x_n[(j - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(BlockSizeX - 2, j, k)];
            }
        }
        request = new MPI_Request[4];
        MPI_Isend(
            bufs->send_x_0, buf_x_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX == 0 ? GridSizeX - 1 : BlockIdxX - 1, BlockIdxY, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 0
        );

        MPI_Isend(
            bufs->send_x_n, buf_x_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX == GridSizeX - 1 ? 0 : BlockIdxX + 1, BlockIdxY, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 1
        );

        MPI_Irecv(
            bufs->recv_x_n, buf_x_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX == GridSizeX - 1 ? 0 : BlockIdxX + 1, BlockIdxY, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 2
        );

        MPI_Irecv(
            bufs->recv_x_0, buf_x_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX == 0 ? GridSizeX - 1 : BlockIdxX - 1, BlockIdxY, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 3
        );
    }
    else {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; j++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                matrix[get_index(0, j, k)] = matrix[get_index(BlockSizeX - 2, j, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; j++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                matrix[get_index(BlockSizeX - 1, j, k)] = matrix[get_index(1, j, k)];
            }
        }
    }
    return request;
}
MPI_Request* update_halo_os_y(double* matrix, struct buffers_sms* bufs,MPI_Request* request){
   if (GridSizeY != 1) {

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; i++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                bufs->send_y_0[(i - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(i, 1, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; i++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                bufs->send_y_n[(i - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(i, BlockSizeY - 2, k)];
            }
        }
        request = new MPI_Request[4];
        MPI_Isend(
            bufs->send_y_0, buf_y_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY == 0 ? GridSizeY - 1 : BlockIdxY - 1, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 0
        );

        MPI_Isend(
            bufs->send_y_n, buf_y_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY == GridSizeY - 1 ? 0 : BlockIdxY + 1, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 1
        );

        MPI_Irecv(
            bufs->recv_y_n, buf_y_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY == GridSizeY - 1 ? 0 : BlockIdxY + 1, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 2
        );

        MPI_Irecv(
            bufs->recv_y_0, buf_y_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY == 0 ? GridSizeY - 1 : BlockIdxY - 1, BlockIdxZ),
            0, MPI_COMM_WORLD, request + 3
        );
    }
    else {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; i++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                matrix[get_index(i, 0, k)] = matrix[get_index(i, BlockSizeY - 2, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; i++){
            for (int k = 1; k < BlockSizeZ - 1; k++){
                matrix[get_index(i, BlockSizeY - 1, k)] = matrix[get_index(i, 1, k)];
            }
        }
    }
    return request;
}
MPI_Request* update_halo_os_z(double* matrix, struct buffers_sms* bufs, MPI_Request* request) {
    if (GridSizeZ != 1) {

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; j++){
            for (int k = 1; k < BlockSizeY - 1; k++){
                bufs->send_z_0[(j - 1) * (BlockSizeY - 2) + (k - 1)] = matrix[get_index(j, k, 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; j++){
            for (int k = 1; k < BlockSizeY - 1; k++){
                bufs->send_z_n[(j - 1) * (BlockSizeY - 2) + (k - 1)] = matrix[get_index(j, k, BlockSizeZ - 2)];
            }
        }
        request = new MPI_Request[4];
        MPI_Isend(
            bufs->send_z_0, buf_z_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY, BlockIdxZ == 0 ? GridSizeZ - 1 : BlockIdxZ - 1),
            0, MPI_COMM_WORLD, request + 0
        );

        MPI_Isend(
            bufs->send_z_n, buf_z_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY, BlockIdxZ == GridSizeZ - 1 ? 0 : BlockIdxZ + 1),
            0, MPI_COMM_WORLD, request + 1
        );

        MPI_Irecv(
            bufs->recv_z_n, buf_z_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY, BlockIdxZ == GridSizeZ - 1 ? 0 : BlockIdxZ + 1),
            0, MPI_COMM_WORLD, request + 2
        );

        MPI_Irecv(
            bufs->recv_z_0, buf_z_size, MPI_DOUBLE,
            id_to_rank(BlockIdxX, BlockIdxY, BlockIdxZ == 0 ? GridSizeZ - 1 : BlockIdxZ - 1),
            0, MPI_COMM_WORLD, request + 3
        );
    }
    else {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; j++){
            for (int k = 1; k < BlockSizeY - 1; k++){
                matrix[get_index(j, k, 0)] = matrix[get_index(j, k, BlockSizeZ - 2)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; j++){
            for (int k = 1; k < BlockSizeY - 1; k++){
                matrix[get_index(j, k, BlockSizeZ - 1)] = matrix[get_index(j, k, 1)];
            }
        }
    }
    return request;
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

    int global_i = (x + BlockIdxX * (N / GridSizeX) + std::min(BlockIdxX, N % GridSizeX));
    int global_j = (y + BlockIdxY * (N / GridSizeY) + std::min(BlockIdxY, N % GridSizeY));
    int global_k = (z + BlockIdxZ * (N / GridSizeZ) + std::min(BlockIdxZ, N % GridSizeZ));

    x = global_i * Hx;
    y = global_j * Hy;
    z = global_k * Hz;
    t = t * Tau;


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
double calc_diff(double* u_an, double* u_error, const double* u_calculated, int step)
{
    double max_error = 0.0;

    #pragma omp parallel for reduction(max: max_error)
        for (int i = 0; i < BlockSizeX; ++i) {
            for (int j = 0; j < BlockSizeY; ++j) {
                for (int k = 0; k < BlockSizeZ; ++k) {
                    u_an[get_index(i, j, k)] = u_analytical(i, j, k, step);

                    double error = 0.0;
                    // error equals zero on edges:
                    if (i != 0 && j != 0 && k != 0 && i != BlockSizeX - 1 && j != BlockSizeY - 1 && k != BlockSizeZ - 1)
                        error = fabs(u_an[get_index(i, j, k)] - u_calculated[get_index(i, j, k)]);

                    u_error[get_index(i, j, k)] = error;
                    if (error > max_error)
                        max_error = error;
                }
            }
        }

    return max_error;
}

int max(int a, int b)
{
    if (a >= b)
        return a;

    return b;
}

int min(int a, int b)
{
    if (a <= b)
        return a;

    return b;
}

int diff(int a, int b, int c)
{
    return max(max(a, b), c) - min(min(a, b), c);
}

int get_least_delimeter(int start, int n)
{
    for (int i = start; i < n; ++i)
        if (n % i == 0)
            return i;

    return n;
}

int init_grid_next_next(int* a, int* b, int n)
{
    if (*a == n)
        return -1;

    int del = get_least_delimeter((*a) + 1, n);
    *a = del;
    *b = n / *a;

    return 0;
}

int init_grid_next(int* a, int* b, int* c, int n)
{
    if (*a == n)
        return -1;

    int res = init_grid_next_next(b, c, n / *a);
    if (res == -1) {
        int del = get_least_delimeter((*a) + 1, n);
        *a = del;
        *b = 1;
        *c = n / *a;

        int res = init_grid_next_next(b, c, n / *a);
    }

    return 0;
}

void init_grid(int n)
{
    int tmp_x = 1;
    int tmp_y = 1;
    int tmp_z = n;

    int res_x = 1;
    int res_y = 1;
    int res_z = n;

    while (1) {
        int res = init_grid_next(&tmp_x, &tmp_y, &tmp_z, n);
        if (res == -1) {
            GridSizeX = res_z;
            GridSizeY = res_y;
            GridSizeZ = res_x;
            return;
        }

        if (diff(tmp_x, tmp_y, tmp_z) < diff(res_x, res_y, res_z)) {
            res_x = tmp_x;
            res_y = tmp_y;
            res_z = tmp_z;
        }
    }
}
void calcualte_size_local_matrixe(){
    BlockSizeX = N / GridSizeX + 2;
    if (BlockIdxX < N % GridSizeX){
        BlockSizeX += 1;
    }

    BlockSizeY = N / GridSizeY + 2;
    if (BlockIdxY < N % GridSizeY){
        BlockSizeY += 1;
    }

    BlockSizeZ = N / GridSizeZ + 2;
    if (BlockIdxZ < N % GridSizeZ){
        BlockSizeZ += 1;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    init_grid(NumProc);
    BlockIdxZ = Rank / (GridSizeX * GridSizeY);
    int tempRank = Rank % (GridSizeX * GridSizeY);
    BlockIdxY = tempRank / GridSizeX;
    tempRank = tempRank % GridSizeX;
    BlockIdxX = tempRank;

    if (Rank == 0){
        std::cout << "Grid for proc - " << NumProc << ", X - " << GridSizeX
              << ", Y - " << GridSizeY << ", Z - " << GridSizeZ << std::endl;

         fflush(stdout);
    }

    N = atoi(argv[1]);
    K = 20;
    STEPS = K;
    T = 0.01;
    std::string L_format = argv[2];
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

    if (Rank == 0) {
        std::cout << "Params \n" << std::endl;
        std::cout << "Size grid - " << N << std::endl;
    }

    calcualte_size_local_matrixe();
    double* u_prev;
    double* u_current;
    double* u_next;
    u_prev = new double[BlockSizeX * BlockSizeY * BlockSizeZ];
    u_current = new double[BlockSizeX * BlockSizeY * BlockSizeZ];
    u_next = new double[BlockSizeX * BlockSizeY * BlockSizeZ];
    // printf("%u %u \n",Rank,u_prev);
    fflush(stdout);
    struct buffers_sms update_struct;
    init_bufs(&update_struct);
    // printf("%u %u \n",update_struct.recv_x_0,Rank);
    
    double start_time = MPI_Wtime();

    int start_z = 0;
    int over_z = BlockSizeZ;
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < BlockSizeX; i++)
        for (int j = 0; j < BlockSizeY; j++)
            for (int k = start_z; k < over_z; k++)
                u_prev[get_index(i, j, k)] = phi(i, j, k);

    
    // std::cout << "After u0" << std::endl;

    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < BlockSizeY; i++) {
    //     for (int j = 0; j < BlockSizeZ; j++) {
    //         u_prev[get_index(0, i, j)] = u_prev[get_index(BlockSizeX, i, j)];
    //         u_prev[get_index(BlockSizeX + 1, i, j)] = u_prev[get_index(1, i, j)];
    //     }
    // }
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < BlockSizeX; i++) {
    //     for (int j = 0; j < BlockSizeZ; j++) {
    //         u_prev[get_index(i, 0, j)] = u_prev[get_index(i, BlockSizeY, j)];
    //         u_prev[get_index(i, BlockSizeY + 1, j)] = u_prev[get_index(i, 1, j)];
    //     }
    // }
    // #pragma omp parallel for collapse(2)   
    // for (int i = 0; i < BlockSizeX; i++) {
    //     for (int j = 0; j < BlockSizeY; j++) {
    //         u_prev[get_index(i, j, 0)] = u_prev[get_index(i, j, BlockSizeZ)];
    //         u_prev[get_index(i, j, BlockSizeZ + 1)] = u_prev[get_index(i, j, 1)];
    //     }
    // }

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < BlockSizeX - 1; i++)
        for (int j = 1; j < BlockSizeY - 1 ; j++)
            for (int k = 1; k < BlockSizeZ - 1; k++){
                    // std::cout << "I" << i << "J" << j << "K" << k<< std::endl;
                    // std::cout << BlockSizeX <<BlockSizeY << BlockSizeZ << std::endl;
                    // fflush(stdout);
                    u_current[get_index(i, j, k)] = u_prev[get_index(i, j, k)] + Tau * Tau * A2 / 2 * delta_h(i, j, k, u_prev);
                    // printf("%u %u \n",Rank,u_prev);
            }

    // printf("%u %u \n",Rank,u_prev);
    fflush(stdout);
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < BlockSizeY; i++) {
    //     for (int j = 0; j < BlockSizeZ; j++) {
    //         u_current[get_index(0, i, j)] = u_current[get_index(BlockSizeX, i, j)];
    //         u_current[get_index(BlockSizeX + 1, i, j)] = u_current[get_index(1, i, j)];
    //     }
    // }
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < BlockSizeX; i++) {
    //     for (int j = 0; j < BlockSizeZ; j++) {
    //         u_current[get_index(i, 0, j)] = u_current[get_index(i, BlockSizeY, j)];
    //         u_current[get_index(i, BlockSizeY + 1, j)] = u_current[get_index(i, 1, j)];
    //     }
    // }
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < BlockSizeX; i++) {
    //     for (int j = 0; j < BlockSizeY; j++) {
    //         u_current[get_index(i, j, 0)] = u_current[get_index(i, j, BlockSizeZ)];
    //         u_current[get_index(i, j, BlockSizeZ + 1)] = u_current[get_index(i, j, 1)];
    //     }
    // }

    // std::cout << "Before update halo" << std::endl;
    MPI_Request* requests_x = NULL;
    MPI_Request* requests_y = NULL;
    MPI_Request* requests_z = NULL;
    requests_x = update_halo_os_x(u_current,&update_struct,requests_x); 
    requests_y = update_halo_os_y(u_current,&update_struct,requests_y); 
    requests_z = update_halo_os_z(u_current,&update_struct,requests_z);
    // wait all
    wait_all_requests(u_current,&update_struct,requests_x,requests_y,requests_z);

    // std::cout << "After update halo" << std::endl;

    double global_max_error = 0.0;
    for (int step = 2; step <= STEPS; step++) {

        #pragma omp parallel for collapse(3)
            for (int i = 1; i < BlockSizeX - 1; i++)
                for (int j = 1; j < BlockSizeY - 1; j++)
                    for (int k = 1; k < BlockSizeZ - 1; k++)
                        u_next[get_index(i, j, k)] = 2 * u_current[get_index(i, j, k)] - u_prev[get_index(i, j, k)] + Tau * Tau * A2 * delta_h(i, j, k, u_current);
           
            if (step != K) {
                MPI_Request* requests_x = NULL;
                MPI_Request* requests_y = NULL;
                MPI_Request* requests_z = NULL;
                requests_x = update_halo_os_x(u_next, &update_struct, requests_x);
                requests_y = update_halo_os_y(u_next, &update_struct, requests_y);
                requests_z = update_halo_os_z(u_next, &update_struct, requests_z);
                // wait all
                wait_all_requests(u_next, &update_struct, requests_x, requests_y, requests_z);


            }
            if(step == K){
                double error = calc_diff(u_prev, u_current, u_next, K);
                MPI_Reduce(&error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            }
            else{
                // при выходе из цикла u_20 (последнее) хранится в u_current, u_19 (пред последнее) в u_prev
                double* tmp = u_prev;
                u_prev = u_current;
                u_current = u_next;
                u_next = tmp;
            }

            // #pragma omp parallel for collapse(2)
            //     for (int i = 0; i < BlockSizeY; i++) {
            //         for (int j = 0; j < BlockSizeZ; j++) {
            //             u_next[get_index(0, i, j)] = u_next[get_index(BlockSizeX, i, j)];
            //             u_next[get_index(BlockSizeX + 1, i, j)] = u_next[get_index(1, i, j)];
            //         }
            //     }
            // #pragma omp parallel for collapse(2)
            //     for (int i = 0; i < BlockSizeX; i++) {
            //         for (int j = 0; j < BlockSizeZ; j++) {
            //             u_next[get_index(i, 0, j)] = u_next[get_index(i, BlockSizeY, j)];
            //             u_next[get_index(i, BlockSizeY + 1, j)] = u_next[get_index(i, 1, j)];
            //         }
            //     }
            // #pragma omp parallel for collapse(2)
            //     for (int i = 0; i < BlockSizeX; i++) {
            //         for (int j = 0; j < BlockSizeY; j++) {
            //             u_next[get_index(i, j, 0)] = u_next[get_index(i, j, BlockSizeZ)];
            //             u_next[get_index(i, j, BlockSizeZ + 1)] = u_next[get_index(i, j, 1)];
            //         }
            //     }
    }
    double end_time =  MPI_Wtime();
    double cuurent_time = end_time - start_time;
    double max_time = 0.0;
    MPI_Reduce(&cuurent_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(Rank == 0){ 
        printf("error = %.10f\n", global_max_error);
        std::cout << "time - " << max_time << "\n";
        fflush(stdout);
    }

    // std::cout << "debug"<< "\n";

    // printf("%u %u \n",Rank,u_prev);
    // fflush(stdout);

    delete[] u_prev;
    delete [] u_current;
    delete [] u_next;

    if (update_struct.send_x_0 != nullptr){
        // std::cout << "debug x0"<< Rank<< "\n";
        // fflush(stdout);
        delete[] update_struct.send_x_0;
    }
    if (update_struct.send_x_n != nullptr){
        // std::cout << "debug xn"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.send_x_n;
    }
    if (update_struct.send_y_0 != nullptr){
        // std::cout << "debug y0"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.send_y_0;
    }
    if (update_struct.send_y_n != nullptr){
        // std::cout << "debug yn"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.send_y_n;
    }
    if (update_struct.send_z_0 != nullptr){
        // std::cout << "debug z0"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.send_z_0;
    }
    if (update_struct.send_z_n != nullptr){
        // std::cout << "debug zn"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.send_z_n;
    }

    if (update_struct.recv_x_0 != nullptr){
        // std::cout << "debug x0 rec"<< Rank << "\n";
        // printf("%u %u \n",update_struct.recv_x_0,Rank);
        // fflush(stdout);
        delete[] update_struct.recv_x_0;
    }
    if (update_struct.recv_x_n != nullptr){
        //         std::cout << "debug xn rec"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.recv_x_n;
    }
    if (update_struct.recv_y_0 != nullptr){
        //         std::cout << "debug y0 rec"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.recv_y_0;
    }
    if (update_struct.recv_y_n != nullptr){
        //         std::cout << "debug yn rec"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.recv_y_n;
    }
    if (update_struct.recv_z_0 != nullptr){
        //         std::cout << "debug z0 rec"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.recv_z_0;
    }
    if (update_struct.recv_z_n != nullptr){
        //         std::cout << "debug zn rec"<< Rank << "\n";
        // fflush(stdout);
        delete[] update_struct.recv_z_n;
    }

    // fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    std::cout << "Finish" << std::endl;

    return 0;
}