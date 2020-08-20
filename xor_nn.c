#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "matrix.h"
#include "more_math.h"

static const int num_inputs = 2;
static const int num_hidden = 2;
static const int num_outputs = 1;
static const int EPOCHS = 10000;
static const double LR = 0.1;

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

int main()
{
    srand(time(NULL));

    //Initializing training sets
    Matrix training_inputs;
    m_init(&training_inputs, 4, 2);
    m_set(&training_inputs, 1, 1, 1.0);
    m_set(&training_inputs, 2, 0, 1.0);
    m_set(&training_inputs, 3, 0, 1.0);
    m_set(&training_inputs, 3, 1, 1.0);

    Matrix training_outputs;
    m_init(&training_outputs, 4, 1);
    m_set(&training_outputs, 1, 0, 1.0);
    m_set(&training_outputs, 2, 0, 1.0);

    Matrix hidden_weights;
    m_init(&hidden_weights, num_inputs, num_hidden);
    m_set(&hidden_weights, 0, 0, 0.1);
    m_set(&hidden_weights, 0, 1, 0.2);
    m_set(&hidden_weights, 1, 0, 0.3);
    m_set(&hidden_weights, 1, 1, 0.4);

    Matrix hidden_bias;
    m_init(&hidden_bias, 1, num_hidden);
    m_set(&hidden_bias, 0, 0, 0.5);
    m_set(&hidden_bias, 0, 1, 0.6);

    Matrix output_weights;
    m_init(&output_weights, num_hidden, num_outputs);
    m_set(&output_weights, 0, 0, 0.1);
    m_set(&output_weights, 1, 0, 0.2);

    Matrix output_bias;
    m_init(&output_bias, 1, num_outputs);
    m_set(&output_bias, 0, 0, 0.3);

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
        //Some weird behavior with this one...
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

    printf("\n========== FLAG ==========\n");
    m_full_print(&hidden_weights);
    m_full_print(&hidden_bias);
    m_full_print(&output_weights);
    m_full_print(&output_bias);

    Matrix test_hw;
    m_init(&test_hw, 2, 2);
    m_set(&test_hw, 0, 0, 3.1382);
    m_set(&test_hw, 0, 1, 5.4453);
    m_set(&test_hw, 1, 0, 3.1573);
    m_set(&test_hw, 1, 1, 5.5619);
    Matrix test_hb;
    m_init(&test_hb, 1, 2);
    m_set(&test_hb, 0, 0, -4.7832);
    m_set(&test_hb, 0, 1, -2.1433);
    Matrix test_ow;
    m_init(&test_ow, 2, 1);
    m_set(&test_ow, 0, 0, -7.0657);
    m_set(&test_ow, 1, 0, 6.5386);
    Matrix test_ob;
    m_init(&test_ob, 1, 1);
    m_set(&test_ob, 0, 0, -2.8911);

    printf("Prediction for (0, 0) (Expected: 0): %f\n", predict(0.0, 0.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0, 1) (Expected: 1): %f\n", predict(0.0, 1.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (1, 0) (Expected: 1): %f\n", predict(1.0, 0.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (1, 1) (Expected: 0): %f\n", predict(1.0, 1.0, &hidden_weights, &hidden_bias, &output_weights, &output_bias));

    printf("Prediction for (0.2, 0.2) (Expected: 0): %f\n", predict(0.2, 0.2, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.2, 0.8) (Expected: 1): %f\n", predict(0.2, 0.8, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.8, 0.2) (Expected: 1): %f\n", predict(0.8, 0.2, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    printf("Prediction for (0.8, 0.8) (Expected: 0): %f\n", predict(0.8, 0.8, &hidden_weights, &hidden_bias, &output_weights, &output_bias));
    //TEST Already Trained Parameters
    // printf("Prediction for (0, 0) (Expected: 0): %f\n", predict(0.0, 0.0, &test_hw, &test_hb, &test_ow, &test_ob));
    // printf("Prediction for (0, 1) (Expected: 1): %f\n", predict(0.0, 1.0, &test_hw, &test_hb, &test_ow, &test_ob));
    // printf("Prediction for (1, 0) (Expected: 1): %f\n", predict(1.0, 0.0, &test_hw, &test_hb, &test_ow, &test_ob));
    // printf("Prediction for (1, 1) (Expected: 0): %f\n", predict(1.0, 1.0, &test_hw, &test_hb, &test_ow, &test_ob));
}