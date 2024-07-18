#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <iostream> // Add this line for std::cout

class TfliteModelBase {
public:
    TfliteModelBase(const char* model_path) {
        model = tflite::FlatBufferModel::BuildFromFile(model_path);
        if (!model) {
            // Handle error, model loading failed
        }
        interpreter = std::make_unique<tflite::Interpreter>(*model);
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            // Handle error, tensor allocation failed
        }
    }

    std::vector<int> GetInputTensorShape(int index) const {
        TfLiteIntArray* dims = interpreter->tensor(index)->dims;
        return std::vector<int>(dims->data, dims->data + dims->size);
    }

    void SetInputTensor(int index, const float* data) {
        int input_index = interpreter->inputs()[index];
        interpreter->typed_tensor<float>(input_index)[0] = *data;
    }

    void Invoke() {
        if (interpreter->Invoke() != kTfLiteOk) {
            // Handle error, inference invocation failed
        }
    }

    const float* GetOutputTensor(int index) const {
        int output_index = interpreter->outputs()[index];
        return interpreter->typed_output_tensor<float>(output_index);
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

class TfliteVAEEncoder : public TfliteModelBase {
public:
    TfliteVAEEncoder(const char* model_path) : TfliteModelBase(model_path) {}

    std::vector<float> RunTflite(const float* inputs, int input_size) {
        // Assuming there's only one input tensor
        SetInputTensor(0, inputs);
        Invoke();

        // Get output tensor size
        int output_size = interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float);
        std::vector<float> outputs(output_size);
        std::memcpy(outputs.data(), GetOutputTensor(0), output_size * sizeof(float));
        return outputs;
    }

};


int main() {
    
    const char* model_path = "../vae_encoder.tflite";
    TfliteVAEEncoder encoder(model_path);

    // Prepare input data (example using hardcoded input, adjust as needed)
    const float sample[] = { /* your input data here */ };

    // Run inference
    std::vector<float> outputs = encoder.RunTflite(sample, sizeof(sample) / sizeof(float));

    // Process outputs as needed
    // Example: print the first output value
    if (!outputs.empty()) {
        std::cout << "Output value: " << outputs[0] << std::endl;
    }

    return 0;
}

// Load the model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);

// Build the interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
interpreter->AllocateTensors();

float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

interpreter->Invoke();

float* output = interpreter->typed_output_tensor<float>(0);
