enum NeatGruStatus
{
    Sucess,
    InvalidString,
    MissingFile,
    FailedToReadFile,
    InvalidFile,
};

typedef struct
{
    NeatGruStatus status;
    struct NeuralNetworkErased *network;
} NeatGruResult;

extern "C" NeatGruResult load_network_from_file_f32(const char *const file_path);

extern "C" NeatGruResult load_network_from_file_f64(const char *const file_path);

extern "C" void compute_network_f32(
    struct NeuralNetworkErased *network,
    const long input_size,
    const float *inputs,
    float *outputs);

extern "C" void compute_network_f64(
    struct NeuralNetworkErased *network,
    const long input_size,
    const double *inputs,
    double *outputs);

extern "C" void reset_network_f32(
    struct NeuralNetworkErased *network);

extern "C" void reset_network_f64(
    struct NeuralNetworkErased *network);

#if __cplusplus
namespace NeatGru
{
    class NeuralNetwork
    {
    };
}
#endif