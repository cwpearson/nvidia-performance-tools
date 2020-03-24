__global__ void kernel(float *a, float *b, int n) {
    *a = *b;
}

int main(void) {
    float *a, *b;
    cudaMalloc(&a, 10 * sizeof(float));
    cudaMalloc(&b, 10 * sizeof(float));
    kernel<<<1,1>>>(a,b,10);
    cudaDeviceSynchronize();
}