#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define pi 3.14

void
calc_displaces(int *ranks, int *displaces, int world_size, int rank, int n)
{
	if(rank == 0) {
		for (int i = 0; i < world_size; i++)
			ranks[i] = n / world_size;
		for (int i = 0; i < n % (world_size); i++)
			ranks[i] += 1;
		displaces[0] = 0;
		for (int i = 1; i < world_size; i++)
			displaces[i] = displaces[i - 1] + ranks[i - 1];
	}
	MPI_Bcast(ranks, world_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(displaces, world_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void
tridiag_matrix_alg(double *a, double *b, double *c, double *f, int n, double *x)
{
	double *alpha = malloc(sizeof(double) * n);
	double *beta = malloc(sizeof(double) * n);

	alpha[0] = 0;
	beta[0] = 0;
	alpha[1] = - b[0] / c[0];
	beta[1] = f[0] / c[0];
	/* Forward. */
	for (int i = 0; i < n - 1; i++) {
		alpha[i + 1] = -b[i] / (a[i] * alpha[i] + c[i]);
		beta[i + 1] = (f[i] - a[i] * beta[i]) /
			      (alpha[i] * a[i] + c[i]);
	}
	x[n - 1] = (f[n - 1] - a[n - 1] * beta[n - 1]) /
		   (a[n - 1] * alpha[n - 1] + c[n - 1]);
	/* Backward. */
	for (int i = n - 2; i >= 0; i--)
		x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1];

	free(alpha);
	free(beta);
}

/**
 * Смотри "Параллельный вариант метода прогонки", чтобы понять,
 * че происходит.
 */
void
tridiag_matrix_alg_par(double *A, double *B, double *C, double *F, int *ranks,
		       int *displaces, int rank, int world_size, double *x_res)
{
	int current_rank = ranks[rank];
	int size = ranks[rank] * sizeof(double);
	double *a = malloc(size);
	double *b = malloc(size);
	double *c = malloc(size);
	double *f = malloc(size);
	double *g = malloc(size);
	double *x = malloc(size);

	MPI_Scatterv(A, ranks, displaces, MPI_DOUBLE, a, current_rank, MPI_DOUBLE, 0,
		     MPI_COMM_WORLD);
	MPI_Scatterv(B, ranks, displaces, MPI_DOUBLE, b, current_rank, MPI_DOUBLE, 0,
		     MPI_COMM_WORLD);
	MPI_Scatterv(C, ranks, displaces, MPI_DOUBLE, c, current_rank, MPI_DOUBLE, 0,
		     MPI_COMM_WORLD);
	MPI_Scatterv(F, ranks, displaces, MPI_DOUBLE, f, current_rank, MPI_DOUBLE, 0,
		     MPI_COMM_WORLD);

	double *d = NULL;
	/* Gauss forward elimination. */
	if (rank != 0) {
		d = malloc(size);
		d[0] = a[0];
	}
	for (int i = 0; i < current_rank - 1; i++) {
		c[i + 1] -= a[i + 1] * b[i] / c[i];
		f[i + 1] -= a[i + 1] * f[i] / c[i];
		if (rank != 0)
			d[i + 1] = -a[i + 1] * d[i] / c[i];
	}
	/* Gauss backward elimination. */
	g[current_rank - 2] = b[current_rank - 2];

	for (int i = current_rank - 3; i >= 0; i--) {
		f[i] -= f[i + 1] * b[i] / c[i + 1];
		g[i] = -g[i + 1] * b[i] / c[i + 1];
		if (rank != 0)
			d[i] -= d[i + 1] * b[i] / c[i + 1];
	}
	/* Exchange values. */
	MPI_Status status;
	MPI_Request requests[2];
	double *send_frac, *recv_frac;
	send_frac = malloc(3 * sizeof(double));
	recv_frac = malloc(3 * sizeof(double));
	if (rank != 0) {
		send_frac[0] = d[0] / c[0];
		send_frac[1] = f[0] / c[0];
		send_frac[2] = g[0] / c[0];
		/* Change to Isend. */
		MPI_Isend(send_frac, 3, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
			  &requests[0]);
	}
	if (rank != world_size - 1) {
		/* Change to Irecv. */
		MPI_Irecv(recv_frac, 3, MPI_DOUBLE, rank + 1, MPI_ANY_TAG,
			  MPI_COMM_WORLD, &requests[1]);
		MPI_Wait(&requests[1], &status);
	}

	double g0 = 0.0;
	/* Elimination. */
	if (rank != world_size - 1) {
		c[current_rank - 1] -= recv_frac[0] * b[current_rank - 1];
		f[current_rank - 1] -= recv_frac[1] * b[current_rank - 1];
		g0 = -recv_frac[2] * b[current_rank - 1];
	}

	double *a_temp, *b_temp, *c_temp, *f_temp, *x_temp;
	int temp_size = world_size * sizeof(double);
	if (rank == 0) {
		a_temp = malloc(temp_size);
		b_temp = malloc(temp_size);
		c_temp = malloc(temp_size);
		f_temp = malloc(temp_size);
	}
	x_temp = malloc(temp_size);

	MPI_Gather(&c[current_rank - 1], 1, MPI_DOUBLE, c_temp, 1, MPI_DOUBLE, 0,
		   MPI_COMM_WORLD);
	MPI_Gather(&f[current_rank - 1], 1, MPI_DOUBLE, f_temp, 1, MPI_DOUBLE, 0,
		   MPI_COMM_WORLD);
	MPI_Gather(&g0, 1, MPI_DOUBLE, b_temp, 1, MPI_DOUBLE, 0,
		   MPI_COMM_WORLD);

	if (rank > 0) {
		MPI_Gather(&d[current_rank - 1], 1, MPI_DOUBLE, a_temp, 1, MPI_DOUBLE, 0,
			   MPI_COMM_WORLD);
	} else {
		double dd = 0.0;
		MPI_Gather(&dd, 1, MPI_DOUBLE, a_temp, 1, MPI_DOUBLE, 0,
			   MPI_COMM_WORLD);
	}

	if (rank == 0) {
		tridiag_matrix_alg(a_temp, b_temp, c_temp, f_temp, world_size,
				   x_temp);
	}

	int *rcounts = malloc(sizeof(int) * world_size);
	int *displs = malloc(sizeof(int) * world_size);
	displs[0] = 0;
	rcounts[0] = 0;
	for (int i = 1; i < world_size; i++) {
		rcounts[i] = 2;
		displs[i] = i - 1;
	}

	double *x_to_send = malloc(sizeof(double) * 2);
	MPI_Scatterv(x_temp, rcounts, displs, MPI_DOUBLE, x_to_send,
		     rcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		x[current_rank - 1] = x_temp[0];
		for (int i = current_rank - 2; i >= 0; i--)
			x[i] = (f[i] - g[i] * x_temp[0]) / (c[i]);
	} else {
		x[ranks[rank] - 1] = x_to_send[1];
		for (int i = ranks[rank] - 2; i >=0; i--){
			x[i] = (f[i] - g[i] * x_to_send[1] -
				d[i] * x_to_send[0]) / c[i];
		}
	}

	MPI_Gatherv(x, current_rank, MPI_DOUBLE, x_res, ranks, displaces,
		    MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(a);
	free(b);
	free(c);
	free(f);
	if (rank != 0)
		free(d);
	free(g);
	free(x);
}

int main()
{
	MPI_Init(NULL, NULL);

	/* Get the number of processes. */
	int world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* Get the rank of process. */
	int world_rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	double final_time = 1000;

	/* Number of time steps. */
	int K = 0;
	/* Number of x and y steps. */
	int N = 0;

	if (world_rank == 0) {
		K = 1000;
		N = 1000;
	}
	MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int *ranks = malloc(sizeof(int) * world_size);
	int *displaces = malloc(sizeof(int) * world_size);
	calc_displaces(ranks, displaces, world_size, world_rank, N);

	double t_step = final_time / K, x_y_step = 1. / N;

	double into_F = 1. - t_step / (x_y_step * x_y_step);

	FILE *f;
	if ((f = fopen("output.txt", "w")) == NULL) {
		getchar();
		return 0;
	}

	double **u_k = malloc(sizeof(double *) * N);
	double **u_k_plus_half = malloc(sizeof(double *) * N);
	for (int j = 0; j < N; j++) {
		u_k[j] = malloc(sizeof(double) * N);
		/*
		 * Fill zeroth layer with initial conditions.
		 */
		for (int i = 0; i < N; i++)
			u_k[j][i] = sin(i * x_y_step) * (1. + 2. * j *
				    x_y_step * cos(i * x_y_step));
		u_k_plus_half[j] = malloc(sizeof(double) * N);
	}

	/*
	 * Заранее инициализируем массивы, потому что они
	 * одинаковые во всех итерациях, за исключением одного
	 * элемента A[N -1].
	 */
	double *A = malloc(sizeof(double) * N);
	double *C = malloc(sizeof(double) * N);
	double *F = malloc(sizeof(double) * N);
	A[0] = 0.;
	C[0] = 1.;
	for (int i = 1; i < N - 2; i++) {
		A[i] = 0.5 * t_step / (x_y_step * x_y_step);
		C[i] = 1. + t_step / (x_y_step * x_y_step);
	}
	C[N - 1] = 1.;

	for (int k = 0; k < K; k++) {
		/* First system. */
		A[N - 1] = 0.;
		 /* Fill intermediate (k + 1/2)th layer. */
		for(int j = 1; j < N - 2; j++) {
			F[0] = 0.;
			for (int a = 1; a < N - 2; a++) {
				F[a] = A[1] * (u_k[j - 1][a] + u_k[j + 1][a])
				       + into_F * u_k[j][a] - 0.5 * t_step *
				       cos(pi * j * x_y_step) * exp((k + 0.5) *
				       t_step); /* F_k +1/2 */
			}
			F[N - 1] = 0.;
			tridiag_matrix_alg_par(A, A, C, F, ranks, displaces,
					       world_rank, world_size,
					       u_k_plus_half[j]);
		}

		/*
		 * Fill zeroth and last layers according to
		 * boundary conditions. When y = 0 or y = 1.
		 */
		for (int i = 0; i < N; i++) {
			u_k_plus_half[0][i] = sin(i * x_y_step);
			u_k_plus_half[N - 1][i] = u_k_plus_half[N - 2][i] +
						  x_y_step * sin(2 * i * x_y_step);
		}

		/* Second system. */
		A[N - 1] = -1.;
		 /* Fill (k + 1)th layer. */
		double *tmp_arr = malloc(sizeof(double) * N);
		for(int i = 1; i < N - 1; i++) {
			F[0] = sin(i * x_y_step);
			for (int a = 1; a < N - 2; a++) {
				/* 14. */
				F[a] = A[1] * (u_k_plus_half[a][i - 1] +
				       u_k_plus_half[a][i + 1]) + into_F *
				       u_k_plus_half[a][i] - 0.5 * t_step *
				       cos(pi * a * x_y_step) *
				       exp((k + 1.) * t_step); /* F_k + 1 */
			}
			F[N - 1] = x_y_step * sin(2 * i * x_y_step);
			tridiag_matrix_alg_par(A, A, C, F, ranks, displaces,
					       world_rank, world_size, tmp_arr);
			for (int a = 0; a < N; a++)
				u_k[a][i] = tmp_arr[a];
		}
		free(tmp_arr);

		/*
		 * Fill zeroth and last layers according to
		 * boundary conditions. When x = 0 or x = 1.
		 */
		for (int j = 0; j < N; j++) {
			u_k[j][0] = 0.;
			u_k[j][N - 1] = 0.;
		}
	}

	free(A);
	free(C);
	free(F);
	for (int j = 0; j < N; j++) {
		free(u_k[j]);
		free(u_k_plus_half[j]);
	}
	free(u_k);
	free(u_k_plus_half);

	MPI_Finalize();
}
