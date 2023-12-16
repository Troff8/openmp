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

// mpi parameters
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

void alloc_update_bufs(buffers_sms* bufs)
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

    if (BlockIdxZ != 0) {
        bufs->send_z_0 = new double[buf_z_size];
        bufs->recv_z_0 = new double[buf_z_size];
    } else {
        bufs->send_z_0 = nullptr;
        bufs->recv_z_0 = nullptr;
    }

    if (BlockIdxZ != GridSizeZ - 1) {
        bufs->send_z_n = new double[buf_z_size];
        bufs->recv_z_n = new double[buf_z_size];
    } else {
        bufs->send_z_n = nullptr;
        bufs->recv_z_n = nullptr;
    }
}

void wait_all_requests(double* matrix, struct buffers_sms* bufs,MPI_Request* request_x,MPI_Request* request_y,MPI_Request* request_z){
    if (request_x != nullptr) {
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

    if (request_y != nullptr) {
        MPI_Status* statuses = new MPI_Status[4];
        MPI_Waitall(4, request_y, statuses);
        delete[] statuses;
        delete[] request_y;

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; ++i){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(i, 0, k)] = bufs->recv_y_0[(i - 1) * (BlockSizeZ - 2) + (k - 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; ++i){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(i, BlockSizeY - 1, k)] = bufs->recv_y_n[(i - 1) * (BlockSizeZ - 2) + (k - 1)];
            }
        }
    }

    if (request_z != nullptr) {
        MPI_Status* statuses = new MPI_Status[4];
        MPI_Waitall(4, request_z, statuses);
        delete[] statuses;
        delete[] request_z;

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; ++j){
            for (int k = 1; k < BlockSizeY - 1; ++k){
                matrix[get_index(j, k, 0)] = bufs->recv_z_0[(j - 1) * (BlockSizeY - 2) + (k - 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; ++j){
            for (int k = 1; k < BlockSizeY - 1; ++k){
                matrix[get_index(j, k, BlockSizeZ - 1)] = bufs->recv_z_n[(j - 1) * (BlockSizeY - 2) + (k - 1)];
            }
        }
    }
}
void update_halo_os_x(double* matrix, struct buffers_sms* bufs,MPI_Request* request){
    if (GridSizeX != 1) {

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; ++j){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                bufs->send_x_0[(j - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(1, j, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; ++j){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                bufs->send_x_n[(j - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(BlockSizeX - 2, j, k)];
            }
        }
 
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
        for (int j = 1; j < BlockSizeY - 1; ++j){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(0, j, k)] = matrix[get_index(BlockSizeX - 2, j, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeY - 1; ++j){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(BlockSizeX - 1, j, k)] = matrix[get_index(1, j, k)];
            }
        }
    }
}
void update_halo_os_y(double* matrix, struct buffers_sms* bufs,MPI_Request* request){
   if (GridSizeY != 1) {

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; ++i){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                bufs->send_y_0[(i - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(i, 1, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; ++i){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                bufs->send_y_n[(i - 1) * (BlockSizeZ - 2) + (k - 1)] = matrix[get_index(i, BlockSizeY - 2, k)];
            }
        }

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
        for (int i = 1; i < BlockSizeX - 1; ++i){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(i, 0, k)] = matrix[get_index(i, BlockSizeY - 2, k)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < BlockSizeX - 1; ++i){
            for (int k = 1; k < BlockSizeZ - 1; ++k){
                matrix[get_index(i, BlockSizeY - 1, k)] = matrix[get_index(i, 1, k)];
            }
        }
    }
}
void update_halo_os_z(double* matrix, struct buffers_sms* bufs, MPI_Request* request) {
    if (GridSizeX != 1) {

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; ++j){
            for (int k = 1; k < BlockSizeY - 1; ++k){
                bufs->send_z_0[(j - 1) * (BlockSizeY - 2) + (k - 1)] = matrix[get_index(j, k, 1)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; ++j){
            for (int k = 1; k < BlockSizeY - 1; ++k){
                bufs->send_z_n[(j - 1) * (BlockSizeY - 2) + (k - 1)] = matrix[get_index(j, k, BlockSizeZ - 2)];
            }
        }

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
        for (int j = 1; j < BlockSizeX - 1; ++j){
            for (int k = 1; k < BlockSizeY - 1; ++k){
                matrix[get_index(j, k, 0)] = matrix[get_index(j, k, BlockSizeZ - 2)];
            }
        }

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < BlockSizeX - 1; ++j){
            for (int k = 1; k < BlockSizeY - 1; ++k){
                matrix[get_index(j, k, BlockSizeZ - 1)] = matrix[get_index(j, k, 1)];
            }
        }
    }
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
    int global_k = (z + BlockIdxZ * ((N - 1) / GridSizeZ) + std::min(BlockIdxZ, (N - 1) % GridSizeZ));

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


void init_grid(int n)
{
    int tmp_x = 1;
    int tmp_y = 1;
    int tmp_z = n;

    int res_x = 1;
    int res_y = 1;
    int res_z = n;

    bool loop = true;
    while (loop) {
        if (tmp_x == n) {
            GridSizeX = res_z;
            GridSizeY = res_y;
            GridSizeZ = res_x;
            return;
        }

        // Find the least denominator
        int start = tmp_x + 1;
        for (int i = start; i < n; ++i)
            if (n % i == 0) {
                tmp_x = i;
                break;
            }

        // Get next grid
        if (tmp_x != n) {
            tmp_y = 1;
            tmp_z = n / tmp_x;

            if (tmp_x != n) {
                int sub_start = tmp_y + 1;
                for (int i = sub_start; i < tmp_z; ++i)
                    if (tmp_z % i == 0) {
                        tmp_y = i;
                        break;
                    }
            }
        }

        // Calculate diff3
        int current_diff = tmp_x;
        if (tmp_y > current_diff)
            current_diff = tmp_y;
        if (tmp_z > current_diff)
            current_diff = tmp_z;

        int best_diff = res_x;
        if (res_y > best_diff)
            best_diff = res_y;
        if (res_z > best_diff)
            best_diff = res_z;

        if (current_diff < best_diff) {
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
void combined(double* result_matrix, const double* matrix)
{
    if (Rank != 0) {
        int block_sizes[3] = { BlockSizeX - 2, BlockSizeY - 2, BlockSizeZ - 2 };
        MPI_Send(block_sizes, 3, MPI_INT, 0, RESULT_SIZES_TAG, MPI_COMM_WORLD);

        int buf_size = (BlockSizeX - 2) * (BlockSizeY - 2) * (BlockSizeZ - 2);
        double* buf = new double [buf_size];

        #pragma omp parallel for
        for (int i = 1; i < BlockSizeX - 1; ++i)
            for (int j = 1; j < BlockSizeY - 1; ++j)
                for (int k = 1; k < BlockSizeZ - 1; ++k)
                    buf[(i - 1) * (BlockSizeY - 2) * (BlockSizeZ - 2) + (j - 1) * (BlockSizeZ - 2) + k - 1] = 
                        matrix[get_index(i, j, k)];

        MPI_Send(buf, buf_size, MPI_DOUBLE, 0, RESULT_TAG, MPI_COMM_WORLD);

        delete[] buf;
    }
    else { 

        for (int i = 0; i < BlockSizeX - 1; ++i) {
            for (int j = 0; j < BlockSizeY - 1; ++j) {
                for (int k = 1; k < BlockSizeZ - 1; ++k) {
                    result_matrix[i * (N + 1) * (N + 1) + j * (N + 1) + k] = matrix[get_index(i, j, k)];
                }
            }
        }

        int buf_size = (N / GridSizeX + 1) * (N / GridSizeY + 1) * (N / GridSizeZ + 1);
        double* buf = new double [buf_size];
        
        MPI_Status status;
        for (int proc = 1; proc < NumProc; ++proc) {
            int block_sizes[3];
            int res = MPI_Recv(block_sizes, 3, MPI_INT, proc, RESULT_SIZES_TAG, MPI_COMM_WORLD, &status);

            int total_block_size = block_sizes[0] * block_sizes[1] * block_sizes[2];
            res = MPI_Recv(buf, total_block_size, MPI_DOUBLE, proc, RESULT_TAG, MPI_COMM_WORLD, &status);

            int proc_idx_x, proc_idx_y, proc_idx_z;
            proc_idx_z = proc / (GridSizeX * GridSizeY);
            proc = proc % (GridSizeX * GridSizeY);
            proc_idx_y = proc / GridSizeX;
            proc = proc % GridSizeX;
            proc_idx_x = proc;

            #pragma omp parallel for
            for (int i = 0; i < block_sizes[0]; ++i) {
                for (int j = 0; j < block_sizes[1]; ++j) {
                    for (int k = 0; k < block_sizes[2]; ++k) {
                        int global_i = (i + 1 + proc_idx_x * (N / GridSizeX) + std::min(proc_idx_x, N % GridSizeX));
                        int global_j = (j + 1 + proc_idx_y * (N  / GridSizeY) + std::min(proc_idx_y, N % GridSizeY));
                        int global_k = (k + 1 + proc_idx_z * ((N - 1) / GridSizeZ) + std::min(proc_idx_z, (N - 1) % GridSizeZ));

                        double elem = buf[i * block_sizes[1] * block_sizes[2] + j * block_sizes[2] + k];

                        result_matrix[global_i * (N + 1) * (N + 1) + global_j * (N + 1) + global_k] = elem;

                        if (global_i == N)
                            result_matrix[global_j * (N + 1) + global_k] = elem;

                        if (global_j == N)
                            result_matrix[global_i * (N + 1) * (N + 1) + global_k] = elem;

                        if (global_k == N)
                            result_matrix[global_i * (N + 1) * (N + 1) + global_j * (N + 1)] = elem;


                        if (global_i == N && global_j == N && global_k == N)
                            result_matrix[global_k] = elem;
                    }
                }
            }
        }
        delete[] buf;
    }
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    init_grid(NumProc);
    BlockIdxZ = Rank / (GridSizeX * GridSizeY);
    Rank = Rank % (GridSizeX * GridSizeY);
    BlockIdxY = Rank / GridSizeX;
    Rank = Rank % GridSizeX;
    BlockIdxX = Rank;

    if (Rank == 0){
        std::cout << "Grid for proc - " << NumProc << "x - " << GridSizeX
              << ", Y - " << GridSizeY << ", Z - " << GridSizeZ << std::endl;
    }

    int nthreads = 1; 
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
        // std::cout << "number of threads" << nthreads << "\n";
    }
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    STEPS = K;
    T = atof(argv[3]);
    std::string L_format = argv[4];
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

    calcualte_size_local_matrixe();
    double* u_prev;
    double* u_current;
    double* u_next;
    u_prev = new double[BlockSizeX * BlockSizeY * BlockSizeZ];
    u_current = new double[BlockSizeX * BlockSizeY * BlockSizeZ];
    u_next = new double[BlockSizeX * BlockSizeY * BlockSizeZ];

    struct buffers_sms update_struct;
    alloc_update_bufs(&update_struct);
    

    double start_time = MPI_Wtime();


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

    MPI_Request* requests_x = new MPI_Request[4];
    update_halo_os_x(u_current,&update_struct,requests_x); 
    MPI_Request* requests_y = new MPI_Request[4];
    update_halo_os_y(u_current,&update_struct,requests_y); 
    MPI_Request* requests_z = new MPI_Request[4];
    update_halo_os_z(u_current,&update_struct,requests_z);
    // wait all
    wait_all_requests(u_current,&update_struct,requests_x,requests_y,requests_z);


    for (int step = 2; step <= STEPS; step++) {
        if (step != K){
            MPI_Request* requests_x = new MPI_Request[4];
            update_halo_os_x(u_current,&update_struct,requests_x);
            MPI_Request* requests_y = new MPI_Request[4];
            update_halo_os_y(u_current,&update_struct,requests_y);
            MPI_Request* requests_z = new MPI_Request[4];
            update_halo_os_z(u_current,&update_struct,requests_z);
            // wait all
            wait_all_requests(u_current,&update_struct,requests_x,requests_y,requests_z);
        }
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
  
    double end_time =  MPI_Wtime();

    bool flag = false;
    if(Rank == 0){ // ??
        int result_matrix_size = (N + 1) * (N + 1) * (N + 1);
        double* result_matrix;
        combined(result_matrix,u_next); // ??
        std::cout << "time" << end_time - start_time << "\n";
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
    }


    delete[] u_prev;
    delete [] u_current;
    delete [] u_next;
    if (update_struct.send_x_0 != nullptr)
        delete[] update_struct.send_x_0;
    if (update_struct.send_x_n != nullptr)
        delete[] update_struct.send_x_n;
    if (update_struct.send_y_0 != nullptr)
        delete[] update_struct.send_y_0;
    if (update_struct.send_y_n != nullptr)
        delete[] update_struct.send_y_n;
    if (update_struct.send_z_0 != nullptr)
        delete[] update_struct.send_z_0;
    if (update_struct.send_z_n != nullptr)
        delete[] update_struct.send_z_n;

    if (update_struct.recv_x_0 != nullptr)
        delete[] update_struct.recv_x_0;
    if (update_struct.recv_x_n != nullptr)
        delete[] update_struct.recv_x_n;
    if (update_struct.recv_y_0 != nullptr)
        delete[] update_struct.recv_y_0;
    if (update_struct.recv_y_n != nullptr)
        delete[] update_struct.recv_y_n;
    if (update_struct.recv_z_0 != nullptr)
        delete[] update_struct.recv_z_0;
    if (update_struct.recv_z_n != nullptr)
        delete[] update_struct.recv_z_n;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}