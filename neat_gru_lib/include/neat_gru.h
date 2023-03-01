typedef enum
{
    NeatGruStatusSucess,
    NeatGruStatusInvalidString,
    NeatGruStatusMissingFile,
    NeatGruStatusFailedToReadFile,
    NeatGruStatusInvalidFile,
} NeatGruStatus;

typedef struct
{
    NeatGruStatus status;
    struct NeuralNetworkErased *network;
} NeatGruResult;

#ifdef __cplusplus
extern "C"
{
#endif
    NeatGruResult load_network_from_file_f32(const char *const file_path);

    NeatGruResult load_network_from_file_f64(const char *const file_path);

    void compute_network_f32(
        struct NeuralNetworkErased *network,
        const long input_size,
        const float *inputs,
        float *outputs);

    void compute_network_f64(
        struct NeuralNetworkErased *network,
        const long input_size,
        const double *inputs,
        double *outputs);

    void reset_network_f32(
        struct NeuralNetworkErased *network);

    void reset_network_f64(
        struct NeuralNetworkErased *network);

    void free_network_f32(
        struct NeuralNetworkErased *network);

    void free_network_f64(
        struct NeuralNetworkErased *network);

#ifdef __cplusplus
} // extern "C".

#include <string>
#include <exception>

namespace NeatGru
{

    namespace
    {
        template <typename T>
        class NeuralNetworkImpl
        {
        public:
            static NeatGruResult CLoadFromFile(std::string const &path);
            static void CCompute(struct NeuralNetworkErased *neural_network,
                                 const long input_size,
                                 const T *inputs,
                                 T *outputs);
            static void CReset(struct NeuralNetworkErased *neural_network);
            static void CFree(struct NeuralNetworkErased *network);
        };

        template <>
        class NeuralNetworkImpl<float>
        {
        public:
            static NeatGruResult CLoadFromFile(std::string const &path)
            {
                return load_network_from_file_f32(path.c_str());
            }

            static void CCompute(struct NeuralNetworkErased *neural_network,
                                 const long input_size,
                                 const float *inputs,
                                 float *outputs)
            {
                compute_network_f32(neural_network, input_size, inputs, outputs);
            }

            static void CReset(struct NeuralNetworkErased *neural_network)
            {
                reset_network_f32(neural_network);
            }

            static void CFree(struct NeuralNetworkErased *network)
            {
                free_network_f32(network);
            }
        };

        template <>
        class NeuralNetworkImpl<double>
        {
        public:
            static NeatGruResult CLoadFromFile(std::string const &path)
            {
                return load_network_from_file_f64(path.c_str());
            }

            static void CCompute(struct NeuralNetworkErased *neural_network,
                                 const long input_size,
                                 const double *inputs,
                                 double *outputs)
            {
                compute_network_f64(neural_network, input_size, inputs, outputs);
            }

            static void CReset(struct NeuralNetworkErased *neural_network)
            {
                reset_network_f64(neural_network);
            }

            static void CFree(struct NeuralNetworkErased *network)
            {
                free_network_f64(network);
            }
        };
    }

    class NeatGruException : public std::exception
    {
    public:
        NeatGruException(NeatGruStatus status) : _status(status)
        {
        }

        const char *what() const noexcept override
        {

            const char *error_msg;
            switch (_status)
            {
            case NeatGruStatus::NeatGruStatusInvalidString:
            {
                error_msg = "InvalidString";
                break;
            }
            case NeatGruStatus::NeatGruStatusMissingFile:
            {
                error_msg = "MissingFile";
                break;
            }
            case NeatGruStatus::NeatGruStatusFailedToReadFile:
            {
                error_msg = "FailedToReadFile";
                break;
            }
            case NeatGruStatus::NeatGruStatusInvalidFile:
            {
                error_msg = "InvalidFile";
                break;
            }
            default:
                error_msg = "UnknownError";
            }
            return error_msg;
        }

    private:
        const NeatGruStatus _status;
    };

    template <typename T>
    class NeuralNetwork
    {
    public:
        // Not using string_view for backward compatibility.
        static NeuralNetwork<T> FromFile(std::string const &path)
        {
            const auto result = NeuralNetworkImpl<T>::CLoadFromFile(path);
            if (result.status != NeatGruStatus::NeatGruStatusSucess)
            {
                throw NeatGruException(result.status);
            }
            return NeuralNetwork<T>(result.network);
        }

        void Compute(const long input_size, const T *inputs, T *outputs)
        {
            NeuralNetworkImpl<T>::CCompute(_neural_network, input_size, inputs, outputs);
        }

        void Reset()
        {
            NeuralNetworkImpl<T>::CReset(_neural_network);
        }

        ~NeuralNetwork<T>()
        {
            NeuralNetworkImpl<T>::CFree(_neural_network);
        }

    private:
        NeuralNetwork<T>(NeuralNetwork<T> &) = delete;
        NeuralNetwork<T>(struct NeuralNetworkErased *neural_network) : _neural_network(neural_network) {}
        struct NeuralNetworkErased *_neural_network;
    };
}
#endif
