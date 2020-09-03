#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "matrix.h"
#include "more_math.h"

static const int num_inputs = 2;
static const int num_hidden = 8;
static const int num_outputs = 1;
static const int EPOCHS = 10000;
static const double LR = 0.1;
static const int num_training = 28;
static const int NUMBER_OF_TESTS = 100000;

double cost(double expected_output, double predicted_output)
{
    return 0.5 * pow((expected_output - predicted_output), 2);
}

double predict(double i1, double i2, Matrix *hw, Matrix *hb, Matrix *ow, Matrix *ob)
{
    Matrix single_point;
    m_init(&single_point, num_outputs, num_inputs);
    m_set(&single_point, 0, 0, i1);
    m_set(&single_point, 0, 1, i2);

    Matrix result_hidden;
    m_mult(&single_point, hw, &result_hidden);
    m_add(&result_hidden, hb);
    m_map(&result_hidden, sigmoid);

    Matrix result_output;
    m_mult(&result_hidden, ow, &result_output);
    m_add(&result_output, ob);
    m_map(&result_output, sigmoid);

    double predicted_output = m_get(&result_output, 0, 0);

    m_free_memory(&single_point);
    m_free_memory(&result_hidden);
    m_free_memory(&result_output);

    return predicted_output;
}

/*
    The fuzzy-XOR function is expected to behave in the following manner. Given
    inputs between 0.0 and 1.0, we assume everything below 0.5 is 0 and everything
    equal to or above 0.5 as 1. Then, the expected output is the same as the
    conventional XOR understanding.(0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0. This
    function returns the average error across the total number of tests.
*/
void random_test(Matrix *hw, Matrix *hb, Matrix *ow, Matrix *ob, bool verbose)
{
    int correct_predictions = 0;
    double total_error = 0.0;
    for (int trial = 0; trial < NUMBER_OF_TESTS; trial++)
    {
        double i1 = rand_weight();
        double i2 = rand_weight();

        double expected = i1 + i2;

        if ((i1 < 0.5 && i2 < 0.5) || (i1 > 0.5 && i2 > 0.5))
        {
            expected = 0.0;
        }
        else
        {
            expected = 1.0;
        }

        double predicted_result = predict(i1, i2, hw, hb, ow, ob);

        total_error += 0.5 * pow((expected - predicted_result), 2);

        double rounded_predicted = -1.0;

        if (predicted_result < 0.5)
        {
            rounded_predicted = 0.0;
        }
        else
        {
            rounded_predicted = 1.0;
        }

        if (double_equals(expected, rounded_predicted))
        {
            correct_predictions++;
        }
        else
        {
            if (verbose)
            {
                printf("XOR of (%f, %f) (Expected: %d): %f\n", i1, i2, ((int)(expected)), predicted_result);
            }
        }
    }

    double accuracy = ((double)(correct_predictions)) / ((double)(NUMBER_OF_TESTS));
    printf("Passed %d out of %d tests. (Accuracy = %f)\n", correct_predictions, NUMBER_OF_TESTS, accuracy);
    printf("Total error over %d random tests: %f\n", NUMBER_OF_TESTS, total_error);
    printf("Average error per trial: %f\n", total_error / ((double)(NUMBER_OF_TESTS)));
}

int main()
{
    srand(time(NULL));
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    //Initializing training sets
    Matrix training_inputs;
    m_init(&training_inputs, num_training, 2);
    m_set(&training_inputs, 1, 1, 1.0);
    m_set(&training_inputs, 2, 0, 1.0);
    m_set(&training_inputs, 3, 0, 1.0);
    m_set(&training_inputs, 3, 1, 1.0);

    m_set(&training_inputs, 4, 0, 0.2);
    m_set(&training_inputs, 4, 1, 0.2);
    m_set(&training_inputs, 5, 0, 0.2);
    m_set(&training_inputs, 5, 1, 0.8);
    m_set(&training_inputs, 6, 0, 0.8);
    m_set(&training_inputs, 6, 1, 0.2);
    m_set(&training_inputs, 7, 0, 0.8);
    m_set(&training_inputs, 7, 1, 0.8);

    m_set(&training_inputs, 8, 0, 0.4);
    m_set(&training_inputs, 8, 1, 0.4);
    m_set(&training_inputs, 9, 0, 0.4);
    m_set(&training_inputs, 9, 1, 0.6);
    m_set(&training_inputs, 10, 0, 0.6);
    m_set(&training_inputs, 10, 1, 0.4);
    m_set(&training_inputs, 11, 0, 0.6);
    m_set(&training_inputs, 11, 1, 0.6);

    m_set(&training_inputs, 12, 0, 0.1);
    m_set(&training_inputs, 12, 1, 0.1);
    m_set(&training_inputs, 13, 0, 0.1);
    m_set(&training_inputs, 13, 1, 0.9);
    m_set(&training_inputs, 14, 0, 0.9);
    m_set(&training_inputs, 14, 1, 0.1);
    m_set(&training_inputs, 15, 0, 0.9);
    m_set(&training_inputs, 15, 1, 0.9);

    m_set(&training_inputs, 16, 0, 0.3);
    m_set(&training_inputs, 16, 1, 0.3);
    m_set(&training_inputs, 17, 0, 0.3);
    m_set(&training_inputs, 17, 1, 0.7);
    m_set(&training_inputs, 18, 0, 0.7);
    m_set(&training_inputs, 18, 1, 0.3);
    m_set(&training_inputs, 19, 0, 0.7);
    m_set(&training_inputs, 19, 1, 0.7);

    m_set(&training_inputs, 20, 0, 0.45);
    m_set(&training_inputs, 20, 1, 0.45);
    m_set(&training_inputs, 21, 0, 0.45);
    m_set(&training_inputs, 21, 1, 0.55);
    m_set(&training_inputs, 22, 0, 0.55);
    m_set(&training_inputs, 22, 1, 0.45);
    m_set(&training_inputs, 23, 0, 0.55);
    m_set(&training_inputs, 23, 1, 0.55);

    m_set(&training_inputs, 24, 0, 0.49);
    m_set(&training_inputs, 24, 1, 0.49);
    m_set(&training_inputs, 25, 0, 0.49);
    m_set(&training_inputs, 25, 1, 0.51);
    m_set(&training_inputs, 26, 0, 0.51);
    m_set(&training_inputs, 26, 1, 0.49);
    m_set(&training_inputs, 27, 0, 0.51);
    m_set(&training_inputs, 27, 1, 0.51);

    Matrix training_outputs;
    m_init(&training_outputs, num_training, 1);
    m_set(&training_outputs, 1, 0, 1.0);
    m_set(&training_outputs, 2, 0, 1.0);
    m_set(&training_outputs, 5, 0, 1.0);
    m_set(&training_outputs, 6, 0, 1.0);
    m_set(&training_outputs, 9, 0, 1.0);
    m_set(&training_outputs, 10, 0, 1.0);
    m_set(&training_outputs, 13, 0, 1.0);
    m_set(&training_outputs, 14, 0, 1.0);
    m_set(&training_outputs, 17, 0, 1.0);
    m_set(&training_outputs, 18, 0, 1.0);
    m_set(&training_outputs, 21, 0, 1.0);
    m_set(&training_outputs, 22, 0, 1.0);
    m_set(&training_outputs, 25, 0, 1.0);
    m_set(&training_outputs, 26, 0, 1.0);
    // m_set(&training_outputs, 3, 0, 1.0);

    Matrix hidden_weights;
    m_init(&hidden_weights, num_inputs, num_hidden);
    for (int i = 0; i < (&hidden_weights)->rows; i++)
    {
        for (int j = 0; j < (&hidden_weights)->cols; j++)
        {
            m_set(&hidden_weights, i, j, rand_weight());
        }
    }

    Matrix hidden_bias;
    m_init(&hidden_bias, 1, num_hidden);
    for (int i = 0; i < (&hidden_bias)->rows; i++)
    {
        for (int j = 0; j < (&hidden_bias)->cols; j++)
        {
            m_set(&hidden_bias, i, j, rand_weight());
        }
    }

    Matrix output_weights;
    m_init(&output_weights, num_hidden, num_outputs);
    for (int i = 0; i < (&output_weights)->rows; i++)
    {
        for (int j = 0; j < (&output_weights)->cols; j++)
        {
            m_set(&output_weights, i, j, rand_weight());
        }
    }

    Matrix output_bias;
    m_init(&output_bias, 1, num_outputs);
    m_set(&output_bias, 0, 0, rand_weight());

    //Iterate through epochs
    for (int n = 0; n < EPOCHS; n++)
    {
        //Forward pass
        Matrix inputs;
        m_copy(&training_inputs, &inputs);

        Matrix in_h;
        m_mult(&inputs, &hidden_weights, &in_h);
        m_add(&in_h, &hidden_bias);

        Matrix out_h;
        m_copy(&in_h, &out_h);
        m_map(&out_h, sigmoid);

        Matrix in_o;
        m_mult(&out_h, &output_weights, &in_o);
        m_add(&in_o, &output_bias);

        Matrix out_o;
        m_copy(&in_o, &out_o);
        m_map(&out_o, sigmoid);

        //Backward pass (back propagation)
        Matrix error;
        m_copy(&out_o, &error);
        m_subtract(&error, &training_outputs);

        Matrix derr_douto;
        m_copy(&error, &derr_douto);

        Matrix douto_dino;
        m_copy(&out_o, &douto_dino);
        m_map(&douto_dino, d_sigmoid);

        Matrix derr_dino;
        m_copy(&derr_douto, &derr_dino);
        m_hadamard(&derr_dino, &douto_dino);

        Matrix output_weights_trans;
        m_transpose(&output_weights, &output_weights_trans);

        Matrix error_hidden_layer;
        m_mult(&derr_dino, &output_weights_trans, &error_hidden_layer);

        Matrix douth_dinh;
        m_copy(&out_h, &douth_dinh);
        m_map(&douth_dinh, d_sigmoid);

        Matrix derr_dinh;
        m_copy(&error_hidden_layer, &derr_dinh);
        m_hadamard(&derr_dinh, &douth_dinh);

        Matrix input_trans;
        m_transpose(&inputs, &input_trans);

        Matrix d_hidden_layer;
        m_mult(&input_trans, &derr_dinh, &d_hidden_layer);

        Matrix out_h_trans;
        m_transpose(&out_h, &out_h_trans);

        Matrix d_output_layer;
        m_mult(&out_h_trans, &derr_dino, &d_output_layer);

        //Updating weights and biases
        m_scalar_mult(&d_output_layer, LR);
        m_subtract(&output_weights, &d_output_layer);

        Matrix d_output_bias;
        m_colsum(&derr_dino, &d_output_bias);
        m_scalar_mult(&d_output_bias, LR);
        m_subtract(&output_bias, &d_output_bias);

        m_scalar_mult(&d_hidden_layer, LR);
        m_subtract(&hidden_weights, &d_hidden_layer);

        Matrix d_hidden_bias;
        m_colsum(&derr_dinh, &d_hidden_bias);
        m_scalar_mult(&d_hidden_bias, LR);
        m_subtract(&hidden_bias, &d_hidden_bias);

        m_free_memory(&inputs);
        m_free_memory(&in_h);
        m_free_memory(&out_h);
        m_free_memory(&in_o);
        m_free_memory(&out_o);
        m_free_memory(&error);
        m_free_memory(&derr_douto);
        m_free_memory(&douto_dino);
        m_free_memory(&derr_dino);
        m_free_memory(&output_weights_trans);
        m_free_memory(&error_hidden_layer);
        m_free_memory(&douth_dinh);
        m_free_memory(&derr_dinh);
        m_free_memory(&input_trans);
        m_free_memory(&d_hidden_layer);
        m_free_memory(&out_h_trans);
        m_free_memory(&d_output_layer);
        m_free_memory(&d_output_bias);
        m_free_memory(&d_hidden_bias);
    }

    printf("\n========== Final Weights and Biases ==========\n");
    printf("Hidden Weights\n");
    m_full_print(&hidden_weights);
    printf("Hidden Biases\n");
    m_full_print(&hidden_bias);
    printf("Output Weights\n");
    m_full_print(&output_weights);
    printf("Output Biases\n");
    m_full_print(&output_bias);
    printf("\n========== Predictions on input data ==========\n");
    printf("Prediction for (0, 0) (Expected: 0): %f\n", predict(0.0, 0.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0, 1) (Expected: 1): %f\n", predict(0.0, 1.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (1, 0) (Expected: 1): %f\n", predict(1.0, 0.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (1, 1) (Expected: 0): %f\n", predict(1.0, 1.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("\n");

    printf("Prediction for (0.2, 0.2) (Expected: 0): %f\n", predict(0.2, 0.2, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.2, 0.8) (Expected: 1): %f\n", predict(0.2, 0.8, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.8, 0.2) (Expected: 1): %f\n", predict(0.8, 0.2, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.8, 0.8) (Expected: 0): %f\n", predict(0.8, 0.8, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("\n");
    printf("Prediction for (0.3, 0.3) (Expected: 0): %f\n", predict(0.3, 0.3, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.3, 0.7) (Expected: 1): %f\n", predict(0.3, 0.7, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.7, 0.3) (Expected: 1): %f\n", predict(0.7, 0.3, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.7, 0.7) (Expected: 0): %f\n", predict(0.7, 0.7, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("\n");
    printf("Prediction for (0.4, 0.4) (Expected: 0): %f\n", predict(0.4, 0.4, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.4, 0.6) (Expected: 1): %f\n", predict(0.4, 0.6, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.6, 0.4) (Expected: 1): %f\n", predict(0.6, 0.4, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.6, 0.6) (Expected: 0): %f\n", predict(0.6, 0.6, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("\n");
    printf("\n========== Predictions on random data ==========\n");

    random_test(&hidden_weights, &hidden_bias, &output_weights, &output_bias, false);

    printf("\n");
    printf("xor_nn.py has finished.\n");
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Completed in %f seconds. \n", cpu_time_used);
}