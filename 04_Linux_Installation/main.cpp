#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <iostream> 


int main() {
    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("../vae_encoder.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    interpreter->SetNumThreads(20);

    // Resize input tensors, if desired.
    interpreter->AllocateTensors();
    std::cout << "11111"<< std::endl;

    float* input = interpreter->typed_input_tensor<float>(0);
    // Fill `input`.

    const int batch_size = 1;
    const int height = 512;
    const int width = 512;
    const int channels = 3;

    float input_data[batch_size * height * width * channels];
    for (int i = 0; i < batch_size * height * width * channels; ++i) {
        input_data[i] = 1;  
    }

    std::cout << "1.2" << std::endl;
    std::memcpy(input, input_data, batch_size * height * width * channels * sizeof(float));

    std::cout << "2222222" << std::endl;
    interpreter->Invoke();
    std::cout << "over....." << std::endl;
    float* output = interpreter->typed_output_tensor<float>(0);

    float output_data[2*4*64*64];
    std::memcpy(output_data, output, 2*4*64*64 * sizeof(float));
    
    for (int i=2*4*64*64-4; i < 2*4*64*64; i++){
        std::cout << "i:" << i << " output_data:" << output_data[i]<< std::endl;
    }
    return 0;



}