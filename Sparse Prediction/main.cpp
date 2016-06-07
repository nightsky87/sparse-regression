#include <arrayfire.h>
#include <stdio.h>
#include <stdlib.h>

using namespace af;

array dcaSolveMatrix(array Xk, array Xu, array E0);

#define USE_TANH_L1
//#define USE_SQRT_L1
#define NUM_CG_ITER 200

int main()
{
	FILE *fh;

	int i;

	unsigned int numData;
	double *pBlock, *pBorder;

	fopen_s(&fh, "Data/cluster_01.dat", "rb");
	fread_s(&numData, sizeof(unsigned int), sizeof(unsigned int), 1, fh);

	pBorder = (double *)_aligned_malloc(sizeof(double) * numData * 25, 64);
	pBlock  = (double *)_aligned_malloc(sizeof(double) * numData * 64, 64);

	fread_s(pBorder, sizeof(double) * numData * 25, sizeof(double), numData * 25, fh);
	fread_s(pBlock, sizeof(double) * numData * 64, sizeof(double), numData * 64, fh);

	// Copy the data to ArrayFire arrays
	array Xk(numData, 25, pBorder);
	array Xu(numData, 64, pBlock);
	
	// Adjust the coefficient magnitude for numeric stability
	Xk = Xk;
	Xu = Xu;

	// Initialize the output matrix
	array R = solve(Xk, Xu);


	// Test the solution
	array E0 = Xu - matmul(Xk, solve(Xk, Xu));

	double l1_0 = norm(E0, AF_NORM_MATRIX_1);
	double l2_0 = norm(E0, AF_NORM_EUCLID);
	double ld_0 = l1_0 - l2_0;

	printf("Least-squares solution:\n");
	printf("  l1: %.1f\n", l1_0);
	printf("  l2: %.1f\n", l2_0);
	printf("  ld: %.1f\n", ld_0);

	R = dcaSolveMatrix(Xk, Xu, R);

	array E1 = Xu - matmul(Xk, R);

	double l1_1 = norm(E1, AF_NORM_MATRIX_1);
	double l2_1 = norm(E1, AF_NORM_EUCLID);
	double ld_1 = l1_1 - l2_1;

	printf("Final solution:\n");
	printf("  l1: %.1f\n", l1_1);
	printf("  l2: %.1f\n", l2_1);
	printf("  ld: %.1f\n", ld_1);

	system("PAUSE");
	return 0;
}

array dcaSolveMatrix(array Xk, array Xu, array R)
{
	int i, n;
	int numData = Xu.dims(0);

	// Calculate the error at the starting point
	array E = Xu - matmul(Xk, R);

	// Calculate the gradient of the l2 term
	array G_l2 = matmul(-Xk, E / tile(sqrt(sum(E * E)), numData), AF_MAT_TRANS, AF_MAT_NONE);

	// Calculate the gradient of the l1 term
#ifdef USE_TANH_L1
	array S = tanh(E);
#else
#ifdef USE_SQRT_L1
	array S = E / sqrt(E * E + 1);
#else
	array S = ((E > 0) - (E < 0)).as(f64);
#endif
#endif
	array G_l1 = matmul(-Xk, S, AF_MAT_TRANS, AF_MAT_NONE);

	// Combine the gradients to find the gradient of the objective function
	array G_obj = G_l1 - G_l2;

	// Initialize the search direction
	array S_dir = -G_obj;
	array S_norm = S_dir / tile(sqrt(sum(S_dir * S_dir)), 25);

	array R_best = R;
	array best_norm = sum(abs(G_obj));

	// Iteratively refine the solution
	for (n = 0; n < NUM_CG_ITER; n++)
	{
		// Precalculate the error correction term of the line search
		array E_corr = matmul(Xk, S_norm);

		// Initialize the step size
		array alpha_min = constant(0, 1, 64);
		array alpha_max = constant(pow(0.5, 17), 1, 64);

		// Search for the upper limit of the bracketing line search
		array alpha = alpha_max;
		for (i = 0; i < 33; i++)
		{
			// Double the step size
			alpha = 2 * alpha;

			// Calculate the gradient of the objective function
			array E_new = E - tile(alpha, numData) * E_corr;
#ifdef USE_TANH_L1
			array S_new = tanh(E_new);
#else
#ifdef USE_SQRT_L1
			array S_new = E_new / sqrt(E_new * E_new + 1);
#else
			array S_new = ((E_new > 0) - (E_new < 0)).as(f64);
#endif
#endif
			array G_l1_new = matmul(-Xk, S_new, AF_MAT_TRANS, AF_MAT_NONE);
			array G_obj_new = G_l1_new - G_l2;

			// Calculate the dot product with respect to the search direction
			array G_dot = sum(-S_norm * G_obj_new);

			// Update the upper limit if the current point is still consistent with the search direction
			array alpha_mask = (G_dot > 0).as(f64);
			alpha_max = alpha_max * (1 - alpha_mask) + alpha * alpha_mask;
		}
		alpha_max = 2 * alpha_max;

		// Perform a bisection search for the minimum along the search direction
		for (i = 0; i < 25; i++)
		{
			// Locate the midpoint
			alpha = (alpha_min + alpha_max) / 2;

			// Calculate the gradient of the objective function
			array E_new = E - tile(alpha, numData) * E_corr;
#ifdef USE_TANH_L1
			array S_new = tanh(E_new);
#else
#ifdef USE_SQRT_L1
			array S_new = E_new / sqrt(E_new * E_new + 1);
#else
			array S_new = ((E_new > 0) - (E_new < 0)).as(f64);
#endif
#endif
			array G_l1_new = matmul(-Xk, S_new, AF_MAT_TRANS, AF_MAT_NONE);
			array G_obj_new = G_l1_new - G_l2;

			// Calculate the dot product with respect to the search direction
			array G_dot = sum(-S_norm * G_obj_new, 0);

			// Update the boundaries
			array alpha_mask = (G_dot > 0).as(f64);
			alpha_min = alpha_min * (1 - alpha_mask) + alpha * alpha_mask;
			alpha_max = alpha_max * alpha_mask + alpha * (1 - alpha_mask);
		}

		// Calculate the final midpoint
		alpha = (alpha_min + alpha_max) / 2;

		// Update the vector using the results from the line search
		R = R + tile(alpha, 25) * S_norm;

		// Calculate the error at the new point
		E = Xu - matmul(Xk, R);

		// Calculate the gradient of the l2 term
		G_l2 = matmul(-Xk, E / tile(sqrt(sum(E * E)), numData), AF_MAT_TRANS, AF_MAT_NONE);

		// Calculate the gradient of the l1 term
#ifdef USE_TANH_L1
		S = tanh(E);
#else
#ifdef USE_SQRT_L1
		S = E / sqrt(E * E + 1);
#else
		S = ((E > 0) - (E < 0)).as(f64);
#endif
#endif
		G_l1 = matmul(-Xk, S, AF_MAT_TRANS, AF_MAT_NONE);

		// Combine the gradients to find the gradient of the objective function
		array G_old = G_obj;
		G_obj = G_l1 - G_l2;

		//af_print(sum(sum(abs(G_obj))) / sum(sum(abs(G_old))));

		// Calculate the Polak-Ribiere factor for updating the direction
		array beta = sum(G_obj * (G_obj - G_old)) / sum(G_old * G_old);
		beta = beta * (beta > 0);

		// Update the search direction
		S_dir = -G_obj + tile(beta, 25) * S_dir;
		S_norm = S_dir / tile(sqrt(sum(S_dir * S_dir)), 25);

		// Calculate the norm of the current solutions
		array cur_norm = sum(abs(G_obj));

		// Keep track of the best solution
		array G_mask = (cur_norm < best_norm).as(f64);
		//af_print(join(0, R_best.row(0), R.row(0)));
		R_best = R * tile(G_mask, 25) + R_best * (1 - tile(G_mask, 25));
		best_norm = sum(abs(G_obj)) * G_mask + best_norm * (1 - G_mask);
	}

	return R_best;
}
