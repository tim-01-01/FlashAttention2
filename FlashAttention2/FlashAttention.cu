#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// основные параметры конфигурации
const int N = 8192;   // длина последовательности
const int D = 64;   // размерность скрытого состояния

// настройки для FlashAttention 1.0
const int Bc = 32;   // размер блока по колонкам для K, V
const int Br = 32;   // размер блока по строкам для Q

// настройки для FlashAttention 2.0
#define FA2_BR 128   // число строк на блок потоков
#define FA2_BC 32   // число строк K/V на тайл
#define FA2_WARP_SIZE 32   // размер варпа
#define MAX_D 128   // максимально поддерживаемая размерность для статических массивов

// 1. функция Attention на CPU
void attentionCPU(const float* Q, const float* K, const float* V, float* O, int seq_len, int head_dim) {
    float* S = (float*)malloc(seq_len * seq_len * sizeof(float));
    float* P = (float*)malloc(seq_len * seq_len * sizeof(float));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // шаг 1. S = Q * K^T * scale
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; ++k) {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            S[i * seq_len + j] = sum * scale;
        }
    }

    // Шаг 2. P = softmax(S) по строкам
    for (int i = 0; i < seq_len; ++i) {
        float max_val = S[i * seq_len];
        for (int j = 1; j < seq_len; ++j) {
            max_val = std::max(max_val, S[i * seq_len + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            P[i * seq_len + j] = std::exp(S[i * seq_len + j] - max_val);
            sum_exp += P[i * seq_len + j];
        }

        for (int j = 0; j < seq_len; ++j) {
            P[i * seq_len + j] /= sum_exp;
        }
    }

    // Шаг 3. O = P * V
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < head_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len; ++k) {
                sum += P[i * seq_len + k] * V[k * head_dim + j];
            }
            O[i * head_dim + j] = sum;
        }
    }

    free(S);
    free(P);
}

// 2. Flash Attention 1.0
__global__ void flashAttentionKernel(const float* Q, const float* K, const float* V, float* O,
    int seq_len, int head_dim, float scale) {

    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tx;

    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem;
    float* s_K = s_Q + (Br * head_dim);
    float* s_V = s_K + (Bc * head_dim);

    float m_old = -__int_as_float(0x7F800000);
    float d_old = 0.0f;

    float r_O[MAX_D];
    for (int d = 0; d < head_dim; ++d) {
        r_O[d] = 0.0f;
    }

    if (row < seq_len) {
        for (int d = 0; d < head_dim; ++d) {
            s_Q[tx * head_dim + d] = Q[row * head_dim + d];
        }
    }
    __syncthreads();

    int Tc = (seq_len + Bc - 1) / Bc;

    for (int j = 0; j < Tc; ++j) {
        int elements_to_load = Bc * head_dim;
        int threads_in_block = blockDim.x;

        for (int idx = tx; idx < elements_to_load; idx += threads_in_block) {
            int k_row = (j * Bc) + (idx / head_dim);
            int k_col = idx % head_dim;

            if (k_row < seq_len && k_col < head_dim) {
                s_K[idx] = K[k_row * head_dim + k_col];
                s_V[idx] = V[k_row * head_dim + k_col];
            }
            else {
                s_K[idx] = -__int_as_float(0x7F800000);
                s_V[idx] = 0.0f;
            }
        }
        __syncthreads();

        if (row < seq_len) {
            float r_S[Bc];
            float m_tile = -INFINITY;

            for (int c = 0; c < Bc; ++c) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    sum += s_Q[tx * head_dim + d] * s_K[c * head_dim + d];
                }
                r_S[c] = sum * scale;
                m_tile = fmaxf(m_tile, r_S[c]);
            }

            float m_new = fmaxf(m_old, m_tile);
            float sum_exp_tile = 0.0f;
            for (int c = 0; c < Bc; ++c) {
                sum_exp_tile += std::exp(r_S[c] - m_new);
            }

            float d_new = d_old * std::exp(m_old - m_new) + sum_exp_tile;
            float alpha = std::exp(m_old - m_new);

            for (int d = 0; d < head_dim; ++d) {
                float pv_sum = 0.0f;
                for (int c = 0; c < Bc; ++c) {
                    pv_sum += std::exp(r_S[c] - m_new) * s_V[c * head_dim + d];
                }
                r_O[d] = r_O[d] * alpha * (d_old / d_new) + (pv_sum / d_new);
            }

            m_old = m_new;
            d_old = d_new;
        }
        __syncthreads();
    }

    if (row < seq_len) {
        for (int d = 0; d < head_dim; ++d) {
            O[row * head_dim + d] = r_O[d];
        }
    }
}

// 3. Flash Attention 2.0
__global__ void flashAttention2Kernel(const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len, int head_dim)
{
    // параллелизация по блокам строк Q
    int rowBlockIdx = blockIdx.x;
    int rowStart = rowBlockIdx * FA2_BR;
    int rowEnd = min(rowStart + FA2_BR, seq_len);

    // разбиение работы внутри блока между варпами
    int tid = threadIdx.x;
    int warpId = tid / FA2_WARP_SIZE;
    int lane = tid % FA2_WARP_SIZE;
    constexpr int rowsPerWarp = 32;
    int warpFirstRow = rowStart + warpId * rowsPerWarp;
    int myRow = warpFirstRow + lane;
    bool active = (myRow < rowEnd);

    // загрузка строки Q в регистры потока
    float q_row[MAX_D];
    if (active) {
        for (int d = 0; d < head_dim; ++d)
            q_row[d] = Q[myRow * head_dim + d];
    }

    // состояние онлайн софтмакса в регистрах
    float m = -1e20f;
    float l = 0.0f;
    float o_unscaled[MAX_D];
    if (active) {
        for (int d = 0; d < head_dim; ++d)
            o_unscaled[d] = 0.0f;
    }

    // тайлы для матриц K и V в шеред мемори
    __shared__ float K_tile[FA2_BC * MAX_D];
    __shared__ float V_tile[FA2_BC * MAX_D];

    int numColBlocks = (seq_len + FA2_BC - 1) / FA2_BC;
    const int D4 = head_dim / 4;
    const bool dDivisibleBy4 = (head_dim % 4 == 0);

    // главный цикл по блокам колонок K и V
    for (int colBlock = 0; colBlock < numColBlocks; ++colBlock) {
        int colStart = colBlock * FA2_BC;
        int colEnd = min(colStart + FA2_BC, seq_len);
        int validCols = colEnd - colStart;

        // векторизованная загрузка K_tile и V_tile через float4
        int totalThreads = blockDim.x;
        int idx = tid;
        if (dDivisibleBy4) {
            int totalFloat4 = validCols * D4;
            while (idx < totalFloat4) {
                int elemIdx = idx * 4;
                int kRow = elemIdx / head_dim;
                int kCol = elemIdx % head_dim;
                int globalRow = colStart + kRow;

                float4 kVec = *reinterpret_cast<const float4*>(&K[globalRow * head_dim + kCol]);
                float4 vVec = *reinterpret_cast<const float4*>(&V[globalRow * head_dim + kCol]);

                reinterpret_cast<float4*>(K_tile)[idx] = kVec;
                reinterpret_cast<float4*>(V_tile)[idx] = vVec;
                idx += totalThreads;
            }
        }
        else {
            while (idx < validCols * head_dim) {
                int kRow = idx / head_dim;
                int kCol = idx % head_dim;
                int globalRow = colStart + kRow;
                K_tile[idx] = K[globalRow * head_dim + kCol];
                V_tile[idx] = V[globalRow * head_dim + kCol];
                idx += totalThreads;
            }
        }
        __syncthreads();

        // вычисления внутри тайла
        if (active) {
            float scores[FA2_BC];
            float blockMax = -1e20f;
            for (int k = 0; k < validCols; ++k) {
                float dot = 0.0f;
                const float* kRowPtr = &K_tile[k * head_dim];
#pragma unroll
                for (int d = 0; d < head_dim; ++d)
                    dot += q_row[d] * kRowPtr[d];
                dot /= sqrtf(float(head_dim));
                scores[k] = dot;
                if (dot > blockMax) blockMax = dot;
            }

            // обновление коэфф онлайн софтмакса
            float newMax = (m > blockMax) ? m : blockMax;
            float expDiffM = expf(m - newMax);

            // сокращение флопс, вычисление экспонент 1 раз за итерацию тайла
            float p[FA2_BC];
            float blockExpSum = 0.0f;
            for (int k = 0; k < validCols; ++k) {
                p[k] = expf(scores[k] - newMax);
                blockExpSum += p[k];
            }
            float newL = expDiffM * l + blockExpSum;

            // накопление промежуточного значения O
            for (int d = 0; d < head_dim; ++d) {
                float accum = expDiffM * o_unscaled[d];
                for (int k = 0; k < validCols; ++k) {
                    accum += p[k] * V_tile[k * head_dim + d];
                }
                o_unscaled[d] = accum;
            }

            m = newMax;
            l = newL;
        }
        __syncthreads();
    }

    if (active) {
        float invL = 1.0f / l;
        for (int d = 0; d < head_dim; ++d)
            O[myRow * head_dim + d] = o_unscaled[d] * invL;
    }
}

// функция для макс абсолют погрешности
float getMaxError(const float* ref, const float* test, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_err = std::max(max_err, std::fabs(ref[i] - test[i]));
    }
    return max_err;
}

int main() {
    setlocale(LC_ALL, "Russian");
    SetConsoleOutputCP(1251);

    size_t matrix_size = N * D * sizeof(float);

    std::cout << "=== Запуск ===" << std::endl;
    std::cout << "Размер контекста (N): " << N << ", Размерность скрытого состояния (D): " << D << "\n" << std::endl;

    // выделение памяти на хосте для CPU
    float* h_Q = (float*)malloc(matrix_size);
    float* h_K = (float*)malloc(matrix_size);
    float* h_V = (float*)malloc(matrix_size);
    float* h_O_CPU = (float*)malloc(matrix_size);
    float* h_O_FA1 = (float*)malloc(matrix_size);
    float* h_O_FA2 = (float*)malloc(matrix_size);

    // инициализация фиксированным сидом для воспроизводимости
    srand(42);
    for (int i = 0; i < N * D; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 1. вычисления и замеры на CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    attentionCPU(h_Q, h_K, h_V, h_O_CPU, N, D);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;

    // выделение памяти на GPU
    float* d_Q, * d_K, * d_V, * d_O_FA1, * d_O_FA2;
    cudaMalloc((void**)&d_Q, matrix_size);
    cudaMalloc((void**)&d_K, matrix_size);
    cudaMalloc((void**)&d_V, matrix_size);
    cudaMalloc((void**)&d_O_FA1, matrix_size);
    cudaMalloc((void**)&d_O_FA2, matrix_size);

    cudaMemcpy(d_Q, h_Q, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrix_size, cudaMemcpyHostToDevice);

    // создание событий CUDA для замеров времени
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // 2. вычисления и замеры времени на Flash Attention 1.0
    size_t shared_mem_size_fa1 = (Br * D * sizeof(float)) + (Bc * D * sizeof(float)) + (Bc * D * sizeof(float));
    int threadsPerBlock_fa1 = Br;
    int blocks_fa1 = (N + Br - 1) / Br;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    cudaEventRecord(start_event);
    flashAttentionKernel << <blocks_fa1, threadsPerBlock_fa1, shared_mem_size_fa1 >> > (d_Q, d_K, d_V, d_O_FA1, N, D, scale);
    cudaEventRecord(stop_event);
    cudaDeviceSynchronize();
    float duration_gpu_fa1 = 0;
    cudaEventElapsedTime(&duration_gpu_fa1, start_event, stop_event);

    // 3. вычисления и замеры времени на Flash Attention 2.0
    int numRowBlocks_fa2 = (N + FA2_BR - 1) / FA2_BR;
    int threadsPerBlock_fa2 = FA2_BR;

    cudaEventRecord(start_event);
    flashAttention2Kernel << <numRowBlocks_fa2, threadsPerBlock_fa2 >> > (d_Q, d_K, d_V, d_O_FA2, N, D);
    cudaEventRecord(stop_event);
    cudaDeviceSynchronize();
    float duration_gpu_fa2 = 0;
    cudaEventElapsedTime(&duration_gpu_fa2, start_event, stop_event);

    // копирование рез обратно на хост
    cudaMemcpy(h_O_FA1, d_O_FA1, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_FA2, d_O_FA2, matrix_size, cudaMemcpyDeviceToHost);

    // расчет погрешностей
    float max_err_fa1 = getMaxError(h_O_CPU, h_O_FA1, N * D);
    float max_err_fa2 = getMaxError(h_O_CPU, h_O_FA2, N * D);

    // вывод на консоль
    std::cout << "==================================================" << std::endl;
    std::cout << "1) Скорость вычислений (Время выполнения):" << std::endl;
    std::cout << "   CPU (Наивный Attention):  " << duration_cpu.count() << " мс" << std::endl;
    std::cout << "   GPU (Flash Attention 1.0): " << duration_gpu_fa1 << " мс" << std::endl;
    std::cout << "   GPU (Flash Attention 2.0): " << duration_gpu_fa2 << " мс" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "2) Ускорение относительно CPU:" << std::endl;
    std::cout << "   Flash Attention 1.0 быстрее CPU в: " << duration_cpu.count() / duration_gpu_fa1 << " раз(а)" << std::endl;
    std::cout << "   Flash Attention 2.0 быстрее CPU в: " << duration_cpu.count() / duration_gpu_fa2 << " раз(а)" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "3) Сравнение версий GPU алгоритмов:" << std::endl;
    std::cout << "   Flash Attention 2.0 быстрее Flash Attention 1.0 в: " << duration_gpu_fa1 / duration_gpu_fa2 << " раз(а)" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "4) Максимальная абсолютная погрешность в сравнении с CPU:" << std::endl;
    std::cout << "   Макс. ошибка Flash Attention 1.0: " << max_err_fa1 << std::endl;
    std::cout << "   Макс. ошибка Flash Attention 2.0: " << max_err_fa2 << std::endl;
    std::cout << "==================================================" << std::endl;

    // освобождение памяти
    free(h_Q); free(h_K); free(h_V); free(h_O_CPU); free(h_O_FA1); free(h_O_FA2);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O_FA1); cudaFree(d_O_FA2);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);

    return 0;
}