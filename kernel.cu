#include <cusparse.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
if (cooRowIndex)        cudaFree(cooRowIndex);   \
            	printf("TEST\n");	\
if (cooColIndex)        cudaFree(cooColIndex);   \
if (cooVal)             cudaFree(cooVal);        \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
	if (descr)              cusparseDestroyMatDescr(descr);\
    if (handle)             cusparseDestroy(handle); \
	cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)


int main()
{
	cudaError_t cudaStat1, cudaStat2, cudaStat3;
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	cusparseMatDescr_t descr = 0, descrA2 = 0;

	//int NNZ = 6629222;	// auto
	//int sizeOfMatrix = 448695;
	int NNZ = 16313034; // britain
	int sizeOfMatrix = 7733822;
	//int NNZ = 25165738;	// delaunay
	//int sizeOfMatrix = 4194304;

	int *cooRowIndexHostPtr = 0;
	int *cooColIndexHostPtr = 0;
	float *cooValHostPtr = 0;
	int *cooRowIndex = 0;
	int *cooColIndex = 0;
	float *cooVal = 0;
	int *csrRowPtr = 0;
	int *csrRowPtrA2 = 0;
	int *csrColIndA2 = 0;
	float *csrValA2 = 0;

	cooRowIndexHostPtr = (int*)malloc(NNZ * sizeof(cooRowIndexHostPtr[0]));
	cooColIndexHostPtr = (int*)malloc(NNZ * sizeof(cooColIndexHostPtr[0]));
	cooValHostPtr = (float*)malloc(NNZ * sizeof(cooValHostPtr[0]));
	if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)) {
		CLEANUP("Host malloc failed (matrix)");
		return 1;
	}

	FILE* fp;
	int buff_int = 0;

	printf("Scanning file...\n");
	//fp = fopen("auto_A.txt", "r");	//auto
	fp = fopen("britain_A.txt", "r");	//britain
	//fp = fopen("delaunay_A.txt", "r");	//delaunay

	for (int i = 0; i < NNZ; i++)
	{
		fscanf(fp, "%d", &cooRowIndexHostPtr[i]);
		fscanf(fp, "%d", &cooColIndexHostPtr[i]);
		fscanf(fp, "%f", &cooValHostPtr[i]);
	}
	fclose(fp);

	printf("Scan completed!\n");
	printf("%d\t%d\t%.2f\n", cooRowIndexHostPtr[NNZ - 1], cooColIndexHostPtr[NNZ - 1], cooValHostPtr[NNZ - 1]);

	cudaStat1 = cudaMalloc((void**)& cooRowIndex, NNZ * sizeof(cooRowIndex[0]));
	cudaStat2 = cudaMalloc((void**)& cooColIndex, NNZ * sizeof(cooColIndex[0]));
	cudaStat3 = cudaMalloc((void**)& cooVal, NNZ * sizeof(cooVal[0]));
	if ((cudaStat1 != cudaSuccess) ||
		(cudaStat2 != cudaSuccess) ||
		(cudaStat3 != cudaSuccess) ) {
		CLEANUP("Device malloc failed");
		return 1;
	}
	cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr,
		(size_t)(NNZ * sizeof(cooRowIndex[0])),
		cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr,
		(size_t)(NNZ * sizeof(cooColIndex[0])),
		cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(cooVal, cooValHostPtr,
		(size_t)(NNZ * sizeof(cooVal[0])),
		cudaMemcpyHostToDevice);
	if ((cudaStat1 != cudaSuccess) ||
		(cudaStat2 != cudaSuccess) ||
		(cudaStat3 != cudaSuccess) ) {
		CLEANUP("Memcpy from Host to Device failed");
		return 1;
	}

	/* initialize cusparse library */ 
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		CLEANUP("CUSPARSE Library initialization failed");
		return 1;
	}

	/* create and setup matrix descriptor */
	status = cusparseCreateMatDescr(&descr);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		CLEANUP("Matrix descriptor initialization failed");
		return 1;
	}
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);

	cudaStat1 = cudaMalloc((void**)& csrRowPtr, ((size_t)sizeOfMatrix + 1) * sizeof(csrRowPtr[0]));
	if (cudaStat1 != cudaSuccess) {
		CLEANUP("Device malloc failed (csrRowPtr)");
		return 1;
	}
	status = cusparseXcoo2csr(handle, cooRowIndex, NNZ, sizeOfMatrix,
		csrRowPtr, CUSPARSE_INDEX_BASE_ONE);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		CLEANUP("Conversion from COO to CSR format failed");
		return 1;
	}
	printf("A converted to CSR.\n");


	/*int devId;
	cudaDeviceProp prop;
	cudaError_t cudaStat;
	cudaStat = cudaGetDevice(&devId);
	if (cudaSuccess != cudaStat) {
		CLEANUP("cudaGetDevice failed");
		printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
		return 1;
	}
	cudaStat = cudaGetDeviceProperties(&prop, devId);
	if (cudaSuccess != cudaStat) {
		CLEANUP("cudaGetDeviceProperties failed");
		printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
		return 1;
	}*/

	
	// assume matrices A, B and D are ready.
	int baseC, nnzC;
	csrgemm2Info_t info = NULL;
	size_t bufferSize;
	void* buffer = NULL;
	// nnzTotalDevHostPtr points to host memory
	int* nnzTotalDevHostPtr = &nnzC;
	float alpha = 1.0;
	float beta = 0.0;
	cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
	printf("TEST\n");
	// step 1: create an opaque structure
	cusparseCreateCsrgemm2Info(&info);
	printf("TEST1\n");

	// step 2: allocate buffer for csrgemm2Nnz and csrgemm2
	cusparseScsrgemm2_bufferSizeExt(handle, sizeOfMatrix, sizeOfMatrix, sizeOfMatrix, &alpha,
		descr, NNZ, csrRowPtr, cooColIndex,
		descr, NNZ, csrRowPtr, cooColIndex,
		&beta,
		descr, NNZ, csrRowPtr, cooColIndex,
		info,
		&bufferSize);
	cudaMalloc(&buffer, bufferSize);
	printf("TEST2\n");

	// step 3: compute csrRowPtrC
	cudaMalloc((void**)& csrRowPtrA2, sizeof(int)* ((size_t)sizeOfMatrix + 1));
	printf("TEST3\n");
	/*cusparseXcsrgemm2Nnz(handle, sizeOfMatrix, sizeOfMatrix, sizeOfMatrix,
		descr, NNZ, csrRowPtr, cooColIndex,
		descr, NNZ, csrRowPtr, cooColIndex,
		descr, NNZ, csrRowPtr, cooColIndex,
		descr, csrRowPtrA2, nnzTotalDevHostPtr,
		info, buffer);
	*/


	if (NULL != nnzTotalDevHostPtr) {
		nnzC = *nnzTotalDevHostPtr;
	}
	else {
		cudaMemcpy(&nnzC, csrRowPtrA2 + sizeOfMatrix, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseC, csrRowPtrA2, sizeof(int), cudaMemcpyDeviceToHost);
		nnzC -= baseC;
	}

	nnzC = 27968324;

	// step 4: finish sparsity pattern and value of C
	cudaMalloc((void**)& csrColIndA2, sizeof(int)* nnzC);
	cudaMalloc((void**)& csrValA2, sizeof(double)* nnzC);
	// Remark: set csrValC to null if only sparsity pattern is required.
	printf("TEST4\n");
	cusparseScsrgemm2(handle, sizeOfMatrix, sizeOfMatrix, sizeOfMatrix, &alpha,
		descr, NNZ, cooVal, csrRowPtr, cooColIndex,
		descr, NNZ, cooVal, csrRowPtr, cooColIndex,
		&beta,
		descr, NNZ, cooVal, csrRowPtr, cooColIndex,
		descr, csrValA2, csrRowPtrA2, csrColIndA2,
		info, buffer);

	// step 5: destroy the opaque structure
	cusparseDestroyCsrgemm2Info(info);





	/* destroy matrix descriptor */
	status = cusparseDestroyMatDescr(descr);
	descr = 0;
	if (status != CUSPARSE_STATUS_SUCCESS) {
		CLEANUP("Matrix descriptor destruction failed");
		return 1;
	}

	/* destroy handle */
	status = cusparseDestroy(handle);
	handle = 0;
	if (status != CUSPARSE_STATUS_SUCCESS) {
		CLEANUP("CUSPARSE Library release of resources failed");
		return 1;
	}

	//CLEANUP("The end.\n");

	if (cooRowIndexHostPtr) free(cooRowIndexHostPtr); 
	if (cooColIndexHostPtr) free(cooColIndexHostPtr); 
	if (cooValHostPtr)      free(cooValHostPtr);     
	if (cooRowIndex)        cudaFree(cooRowIndex);   
	printf("TEST\n");
	if (cooColIndex)        cudaFree(cooColIndex);   
	if (cooVal)             cudaFree(cooVal);        
	if (csrRowPtr)          cudaFree(csrRowPtr);     
	if (descr)              cusparseDestroyMatDescr(descr); 
	if (handle)             cusparseDestroy(handle); 
	cudaDeviceReset();

}
