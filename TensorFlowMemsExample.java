package com.example.MemsTensorFlow;


import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import java.nio.FloatBuffer;

public class TensorFlowMemsExample {
    public static void main(String[] args) {
        // Loading model
        SavedModelBundle aimodel = SavedModelBundle.load("Model_saved_directory");

        // Input data
        string[][] inputMems = {{"https://i.ibb.co/6HKvxqm/Euro-Mems.jpg", 
                                 "https://i.ibb.co/hHT61PW/Linguistic-Mem.jpg", 
                                 "https://i.ibb.co/b5NTXh4/Linguistic-Model.jpg", 
                                 "https://i.ibb.co/9ThFqmB/AI-training-model.jpg",
                                 "https://i.ibb.co/2vDpH3k/AI-training-element.jpg",
                                 "https://i.ibb.co/9ThFqmB/AI-training-model.jpg",
                                 "https://i.ibb.co/519qCkx/AI-languages-model.jpg",
                                 "https://i.ibb.co/g3yTzJZ/AI-training-element.jpg",
                                 "https://i.ibb.co/9ThFqmB/AI-training-model.jpg",
                                 "https://i.ibb.co/Vg5vDfG/AI-clarifying-model.jpg",
                                 "https://i.ibb.co/YTT301J/AI-languages-clarifying-model.jpg",
                                 "https://i.ibb.co/cCBXGDg/AI-training-model.jpg",
                                 "https://i.ibb.co/87Tm8GN/AGI-training-analytics.jpg",
                                 "https://i.ibb.co/Ybhf6cx/AI-language-training.jpg",
                                 "https://i.ibb.co/hFdBKGQ/AI-Languages-Training-Element.jpg",
                                 "https://i.ibb.co/tHmY031/AI-language-training-model.jpg",
                                 "https://i.ibb.co/2d5LbgV/AI-emotional-model.jpg",
                                 "https://i.ibb.co/WfhLjHd/AI-Training-Model.jpg",
                                 "https://i.ibb.co/wB9wJQx/AI-Languages-Model.jpg",
                                 "https://i.ibb.co/89P0G1W/AI-Training-model.jpg",
                                 "https://i.ibb.co/nmSc2sq/AGI-languages-training.jpg",
                                 "https://i.ibb.co/5YPTmKV/AGI-training.jpg"
                               }};
        Tensor<Float> inputTensor = Tensor.create(inputMems);

        // Performing inference
        Tensor<?> output = aimodel.session().runner()
            .feed("inputnodename", inputTensor)
            .fetch("outputnodename")
            .run()
            .get(0);

        // Output of the result
        FloatBuffer outputBuffer = FloatBuffer.allocate((int) output.shape()[1]);
        output.writeTo(outputBuffer);
        System.out.println("Generated Result: " + outputBuffer.get(0));
    }
}
