#include "test.h"

void print_test_matrix_results(Matrix *a, Matrix *b, Matrix *c) {
    printf("Results.\n");
    printf("Matrix A:\n");
    m_full_print(a);
    printf("Matrix B:\n");
    m_full_print(b);
    printf("Matrix C:\n");
    m_full_print(c);
    printf("\n");
    printf("\n");
}

void test_matrix() {
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    printf("Initializing Matrices... ");
    Matrix a;
    Matrix b;
    Matrix c;
    Matrix mult_result;
    Matrix copy_result;
    m_identity(&a, 5);
    m_init(&b, 5, 6);
    m_init(&c, 6, 5);
    printf("Done.\n");

    print_test_matrix_results(&a, &b, &c);

    printf("Filling Matrix B and C... ");
    int p = 1;
    for (int i = 0; i < (&b)->rows; i++) {
        for (int j = 0; j < (&b)->cols; j++) {
            m_set(&b, i, j, p);
            p++;
        }
    }
    p = 1;
    for (int i = 0; i < (&c)->rows; i++) {
        for (int j = 0; j < (&c)->cols; j++) {
            m_set(&c, i, j, p);
            p++;
        }
    }
    printf("Done.\n");
    print_test_matrix_results(&a, &b, &c);

    printf("Multiplying B and C into mult_result... ");
    m_mult(&b, &c, &mult_result);
    printf("Done\n");

    printf("Result of mult_result:\n");
    m_full_print(&mult_result);
    printf("\n");

    printf("Copying mult_result into copy_result... ");
    m_copy(&mult_result, &copy_result);
    printf("Done\n");

    printf("Result of copy_result:\n");
    m_full_print(&copy_result);
    printf("\n");

    printf("Transposing copy_result... ");
    printf("Done\n");

    printf("Result of copy_result:\n");
    m_full_print(m_transpose(&copy_result, &mult_result));
    printf("\n");

    m_free_memory(&a);
    m_free_memory(&b);
    m_free_memory(&c);
    m_free_memory(&mult_result);
    m_free_memory(&copy_result);

    printf("All tests for Matrix module done.\n");
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Completed in %f seconds. \n", cpu_time_used);
}