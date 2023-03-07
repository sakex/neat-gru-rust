#include "neat_gru_lib/neat_gru.h"
#include <cassert>
#include <iostream>

void TestC()
{
    const char path[] = "topology_test.json";
    const char bad_path[] = "fake.json";

    const NeatGruResult result = load_network_from_file_f64(path);
    assert(result.status == NeatGruStatus::NeatGruStatusSucess);

    const NeatGruResult result_not_found = load_network_from_file_f64(bad_path);
    assert(result_not_found.status == NeatGruStatus::NeatGruStatusMissingFile);

    const double input_1[] = {0.5, 0.5, 0.1, -0.2};
    const double input_2[] = {-0.5, -0.5, -0.1, 0.2};

    double output_1[2];
    double output_2[2];
    double output_3[2];

    compute_network_f64(result.network, 4, input_1, output_1);
    compute_network_f64(result.network, 4, input_2, output_2);
    compute_network_f64(result.network, 4, input_1, output_3);

    assert(output_1[0] != output_2[0] || output_1[1] != output_2[1]);
    // Because of GRU gates, giving the same input twice won't yield the same output.
    assert(output_1[0] != output_3[0] || output_1[1] != output_3[1]);

    reset_network_f64(result.network);
    compute_network_f64(result.network, 4, input_1, output_3);
    // After resetting, giving the same input sequence should yield the same results

    assert(output_1[0] == output_3[0] && output_1[1] == output_3[1]);
    free_network_f64(result.network);
}

void TestCpp()
{
    using namespace NeatGru;
    const std::string path = "topology_test.json";
    std::string bad_path = "fake.json";

    NeuralNetwork<double> network = NeuralNetwork<double>::FromFile(path);
    try
    {
        const NeuralNetwork<double> bad_result = NeuralNetwork<double>::FromFile(bad_path);
        assert(false);
    }
    catch (NeatGruException e)
    {
        assert(std::string(e.what()) == "MissingFile");
    }
    const double input_1[] = {0.5, 0.5, 0.1, -0.2};
    const double input_2[] = {-0.5, -0.5, -0.1, 0.2};

    double output_1[2];
    double output_2[2];
    double output_3[2];

    network.Compute(4, input_1, output_1);
    network.Compute(4, input_2, output_2);
    network.Compute(4, input_1, output_3);

    assert(output_1[0] != output_2[0] || output_1[1] != output_2[1]);
    // Because of GRU gates, giving the same input twice won't yield the same output.
    assert(output_1[0] != output_3[0] || output_1[1] != output_3[1]);

    network.Reset();
    network.Compute(4, input_1, output_3);
    // After resetting, giving the same input sequence should yield the same results

    assert(output_1[0] == output_3[0] && output_1[1] == output_3[1]);
}

int main()
{
    std::cout << "TESTING C FFIs" << std::endl;
    TestC();
    std::cout << "C FFIs passed\n" << std::endl;
    std::cout << "TESTING C++ FFIs" << std::endl;
    TestCpp();
    std::cout << "C++ FFIs passed" << std::endl;

    return 0;
}
